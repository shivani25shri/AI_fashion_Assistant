# hf_fashion.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import tempfile
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline as hf_pipeline

# ── HF Space client (not-lain/background-removal) ─────────────────────────────
try:
    from gradio_client import Client as HFClient, handle_file
    HAS_HF_CLIENT = True
except Exception:
    HAS_HF_CLIENT = False

HF_SPACE_ID  = "not-lain/background-removal"
HF_API_PNG   = "/png"
HF_API_IMAGE = "/image"

# ── Models ────────────────────────────────────────────────────────────────────
def load_models(device: int = -1) -> Tuple[None, Any]:
    """
    Returns (None, segformer_pipeline). device=-1 for CPU, >=0 for CUDA index.
    """
    seg = hf_pipeline(
        "image-segmentation",
        model="mattmdjaga/segformer_b2_clothes",
        device=device
    )
    return None, seg

# ── Background removal via HF Space (BiRefNet) ────────────────────────────────
def _open_result_as_image(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGBA")
    s = str(x)
    if s.startswith(("http://", "https://")):
        import requests
        r = requests.get(s, stream=True, timeout=30)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            for ch in r.iter_content(8192):
                tmp.write(ch)
            s = tmp.name
    return Image.open(s).convert("RGBA")

def remove_background_hf(img: Image.Image, hf_token: Optional[str] = None) -> Image.Image:
    """
    Call not-lain/background-removal. If anything fails (SSL/no client/etc),
    return an RGBA copy so downstream segmentation still runs.
    """
    if not HAS_HF_CLIENT:
        return img.convert("RGBA")
    try:
        client = HFClient(HF_SPACE_ID, hf_token=hf_token)
    except Exception:
        return img.convert("RGBA")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.convert("RGB").save(tmp.name, "PNG")
        in_path = tmp.name

    # Preferred: /png (returns a transparent PNG)
    try:
        out = client.predict(handle_file(in_path), api_name=HF_API_PNG)
        return _open_result_as_image(out)
    except Exception:
        pass

    # Fallback: /image
    try:
        out = client.predict(Image.open(in_path).convert("RGB"), api_name=HF_API_IMAGE)
        cand = out[0] if isinstance(out, (list, tuple)) else out
        return _open_result_as_image(cand)
    except Exception:
        return img.convert("RGBA")

# ── SegFormer helpers ─────────────────────────────────────────────────────────
def map_label_to_section(label: str) -> Optional[str]:
    l = label.lower().replace("_","-").strip()
    if l in {"upper-clothes","upperclothes","t-shirt","shirt","coat","jacket",
             "outerwear","blazer","dress","top","hoodie","sweater","cardigan","blouse"}:
        return "topwear"
    if l in {"pants","trousers","jeans","skirt","shorts","leggings"}:
        return "bottomwear"
    if any(k in l for k in ("left-shoe","right-shoe","shoe","sneaker","boot","heel","loafer","sandal","sock","socks")):
        return "footwear"
    if any(k in l for k in ("bag","handbag","backpack","hat","cap","belt","scarf",
                            "glasses","sunglasses","watch","wallet","necklace","earring")):
        return "accessories"
    return None

def _as_binary(mask_like: Any) -> Optional[np.ndarray]:
    """
    Normalize any SegFormer mask (PIL or ndarray, float 0–1 or uint8) to a
    binary uint8 {0,255}. Reject degenerate near-empty/near-full masks.
    """
    try:
        if isinstance(mask_like, Image.Image):
            im = mask_like
        elif isinstance(mask_like, np.ndarray):
            arr = mask_like
            if arr.dtype.kind == "f":
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            elif arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            im = Image.fromarray(arr)
        else:
            return None

        if "A" in im.getbands():
            m = np.array(im.getchannel("A"))
        else:
            m = np.array(im.convert("L"))
            _, m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        m = (m > 0).astype(np.uint8) * 255
        cov = m.mean() / 255.0
        if cov < 0.00001 or cov > 0.999:
            return None
        return m
    except Exception:
        return None

def _collect_masks(seg_outputs: List[Dict[str, Any]]) -> Dict[str, List[np.ndarray]]:
    groups: Dict[str, List[np.ndarray]] = {}
    for out in seg_outputs:
        sec = map_label_to_section(out.get("label",""))
        if not sec:
            continue
        m = _as_binary(out.get("mask"))
        if m is None:
            continue
        groups.setdefault(sec, []).append(m)
    return groups

# ── Section-aware selection (tuned) ───────────────────────────────────────────
def _select_components(mask: np.ndarray, section: str) -> np.ndarray:
    """
    Choose plausible components by area and a minimum bbox dimension ratio.
    Fallback: keep largest N components if filters remove everything.
    """
    binm = (mask > 0).astype(np.uint8)
    H, W = binm.shape
    area_img = H * W

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
    comps = []
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        ar = a / area_img
        min_dim_ratio = min(w / W, h / H)
        comps.append((i, a, ar, min_dim_ratio))

    if not comps:
        return (binm * 255).astype(np.uint8)

    # Tuned thresholds
    if section == "footwear":
        # allow small but real shoes; keep up to 2
        min_ar, max_ar, keep, min_dim = 0.0005, 0.25, 2, 0.025
    elif section == "accessories":
        min_ar, max_ar, keep, min_dim = 0.0004, 0.25, 3, 0.02
    else:  # top/bottom
        min_ar, max_ar, keep, min_dim = 0.01, 0.85, 1, 0.10

    filtered = [c for c in comps if (min_ar <= c[2] <= max_ar and c[3] >= min_dim)]
    filtered.sort(key=lambda t: t[1], reverse=True)
    if not filtered:
        filtered = sorted(comps, key=lambda t: t[1], reverse=True)[:keep]

    out = np.zeros_like(binm, dtype=np.uint8)
    for i, *_ in filtered[:keep]:
        out[labels == i] = 255

    # de-speckle, then light close to keep edges crisp
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    k = 3 if section in ("footwear","accessories") else 5
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((k,k), np.uint8), 1)
    return out

def _tight_crop_rgba(rgba: Image.Image, pad: int = 6):
    a = np.array(rgba.split()[-1])
    ys, xs = np.where(a > 0)
    if not (ys.size and xs.size):
        return rgba, [0,0,rgba.width,rgba.height]
    y0,y1 = ys.min(), ys.max()
    x0,x1 = xs.min(), xs.max()
    x0 = max(x0 - pad, 0); y0 = max(y0 - pad, 0)
    x1 = min(x1 + pad, rgba.width); y1 = min(y1 + pad, rgba.height)
    return rgba.crop((x0,y0,x1,y1)), [int(x0),int(y0),int(x1),int(y1)]

# ── Compose with BG alpha (for precise edges) ─────────────────────────────────
def _compose_with_bg_rgba(pre_rgba: Image.Image, mask_u8: np.ndarray) -> Image.Image:
    # AND SegFormer mask with BiRefNet's alpha → very sharp edges
    m1 = Image.fromarray(mask_u8, "L").resize(pre_rgba.size, Image.NEAREST)
    m2 = pre_rgba.split()[-1]
    comb = Image.fromarray(
        np.minimum(np.array(m1, dtype=np.uint8), np.array(m2, dtype=np.uint8)),
        "L"
    )
    out = Image.new("RGBA", pre_rgba.size, (0,0,0,0))
    out.paste(pre_rgba.convert("RGB"), mask=comb)
    return out

# ── Main: HF BG → SegFormer on BG-removed ONLY → section selection ───────────
def segment_image_with_hf(seg_pipeline, img: Image.Image, hf_token: Optional[str]) -> Dict[str, Dict]:
    """
    1) Remove background via HF Space → RGBA cut-out.
    2) Run SegFormer ONLY on the BG-removed RGB.
    3) Section-aware selection; compose with BG alpha for crisp edges; tight crop.
    Returns dict with:
        - "full": {"rgba": RGBA cut-out, "bbox": [x0,y0,x1,y1]}
        - per section ("topwear", "bottomwear", "footwear", "accessories")
    """
    # 1) precise background removal (RGBA)
    pre = remove_background_hf(img, hf_token=hf_token)
    rgb_bg = pre.convert("RGB")

    # 2) segment ONLY the BG-removed image
    outs_bg = seg_pipeline(rgb_bg)
    groups = _collect_masks(outs_bg)

    # 3) section-aware selection → compose with BG alpha → tight crop
    results: Dict[str, Dict] = {}
    results["full"] = {"rgba": pre, "bbox": [0, 0, pre.width, pre.height]}

    for sec, masks in groups.items():
        if not masks:
            continue
        union = np.maximum.reduce(masks)
        sel = _select_components(union, sec)

        cov = sel.mean() / 255.0
        floor = 0.0006 if sec in ("footwear","accessories") else 0.005
        if cov < floor or cov > 0.98:
            continue

        rgba = _compose_with_bg_rgba(pre, sel)
        crop, bbox = _tight_crop_rgba(rgba, pad=6)
        results[sec] = {"rgba": crop, "bbox": bbox}

    return results
