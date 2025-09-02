# hf_fashion.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import os, tempfile
import numpy as np
from PIL import Image
import cv2

from transformers import pipeline as hf_pipeline

# Hugging Face Space client (v1+)
try:
    from gradio_client import Client as HFClient, handle_file
    HAS_HF_CLIENT = True
except Exception:
    HAS_HF_CLIENT = False

HF_SPACE_ID  = "not-lain/background-removal"
HF_API_PNG   = "/png"
HF_API_IMAGE = "/image"

# ──────────────────────────────────────────────────────────────────────────────
# Models
def load_models(device: int = -1) -> Tuple[None, any]:
    seg = hf_pipeline("image-segmentation",
                      model="mattmdjaga/segformer_b2_clothes",
                      device=device)
    return None, seg

# ──────────────────────────────────────────────────────────────────────────────
# Background removal (HF Space; tolerant)
def _open_result_as_image(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGBA")
    s = str(x)
    if s.startswith(("http://","https://")):
        import requests
        r = requests.get(s, stream=True, timeout=30)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            for ch in r.iter_content(8192): tmp.write(ch)
            s = tmp.name
    return Image.open(s).convert("RGBA")

def remove_background_hf(img: Image.Image, hf_token: Optional[str] = None) -> Image.Image:
    """
    Try not-lain/background-removal Space. If anything fails (including SSL),
    return an RGBA copy so downstream segmentation still runs.
    """
    # If gradio_client isn't available, just continue
    if not HAS_HF_CLIENT:
        return img.convert("RGBA")

    # Try to construct the client (this is where your SSL error happened)
    try:
        client = HFClient(HF_SPACE_ID, hf_token=hf_token)
    except Exception:
        # Could not even reach the Space (SSL/proxy/DNS/etc.)
        return img.convert("RGBA")

    # Save input to a temp PNG
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.convert("RGB").save(tmp.name, "PNG")
        in_path = tmp.name

    # 1) Preferred: /png
    try:
        out = client.predict(handle_file(in_path), api_name=HF_API_PNG)
        return _open_result_as_image(out)
    except Exception:
        pass

    # 2) Fallback: /image
    try:
        out = client.predict(Image.open(in_path).convert("RGB"), api_name=HF_API_IMAGE)
        cand = out[0] if isinstance(out, (list, tuple)) else out
        return _open_result_as_image(cand)
    except Exception:
        pass

    # Couldn’t process via Space; continue gracefully
    return img.convert("RGBA")
# ──────────────────────────────────────────────────────────────────────────────
# SegFormer helpers
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

def _as_binary(mask_like) -> Optional[np.ndarray]:
    try:
        if isinstance(mask_like, Image.Image):
            im = mask_like
        else:
            im = Image.fromarray(mask_like)
        if "A" in im.getbands():
            m = np.array(im.getchannel("A"))
        else:
            m = np.array(im.convert("L"))
            _, m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = (m > 0).astype(np.uint8) * 255
        cov = m.mean()/255.0
        # allow tiny items; reject near-empty and near-full
        if cov < 0.00001 or cov > 0.999:
            return None
        return m
    except Exception:
        return None

def _collect_masks(seg_outputs) -> Dict[str, List[np.ndarray]]:
    groups: Dict[str, List[np.ndarray]] = {}
    for out in seg_outputs:
        sec = map_label_to_section(out.get("label",""))
        if not sec: continue
        m = _as_binary(out.get("mask"))
        if m is None: continue
        groups.setdefault(sec, []).append(m)
    return groups

# ──────────────────────────────────────────────────────────────────────────────
# Section-aware component selection (prevents giant blobs)
def _select_components(mask: np.ndarray, section: str) -> np.ndarray:
    """Return a cleaned mask appropriate for the section."""
    binm = (mask > 0).astype(np.uint8)
    H, W = binm.shape
    area_img = H * W

    # connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
    comps = []
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        ar = a / area_img
        comps.append((i, a, ar, (x, y, w, h)))

    if not comps:
        return mask

    # per-section area bounds
    if section in ("footwear",):
        min_ar, max_ar, keep = 0.00003, 0.12, 2   # up to two shoes
    elif section in ("accessories",):
        min_ar, max_ar, keep = 0.00002, 0.15, 1
    else:  # top/bottom
        min_ar, max_ar, keep = 0.005, 0.75, 1

    # keep components within range (largest first). If none, fallback to largest.
    comps = [c for c in comps if min_ar <= c[2] <= max_ar]
    comps.sort(key=lambda t: t[1], reverse=True)
    if not comps:
        comps = sorted(comps, key=lambda t: t[1], reverse=True)  # fallback to biggest
        comps = comps[:1]
    else:
        comps = comps[:keep]

    out = np.zeros_like(binm, dtype=np.uint8)
    for i, *_ in comps:
        out[labels == i] = 255

    # light closing to fill pinholes on tiny items
    if section in ("footwear","accessories"):
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
    else:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), 1)

    return out

def _compose_rgba(base: Image.Image, mask_u8: np.ndarray) -> Image.Image:
    mask_full = Image.fromarray(mask_u8, "L").resize(base.size, Image.NEAREST)
    rgba = Image.new("RGBA", base.size, (0,0,0,0))
    rgba.paste(base.convert("RGB"), mask=mask_full)
    return rgba

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

# ──────────────────────────────────────────────────────────────────────────────
# Main: HF BG → SegFormer (BG + original) → merge → per-section selection
def segment_image_with_hf(seg_pipeline, img: Image.Image, hf_token: Optional[str]) -> Dict[str, Dict]:
    # 1) BG removal (RGBA, even if alpha not perfect)
    pre = remove_background_hf(img, hf_token=hf_token)
    rgb_bg = pre.convert("RGB")
    rgb_orig = img.convert("RGB")

    # 2) SegFormer on both (rescues tiny items), then merge
    outs_bg = seg_pipeline(rgb_bg)
    outs_or = seg_pipeline(rgb_orig)
    groups = _collect_masks(outs_bg)
    for k, v in _collect_masks(outs_or).items():
        groups.setdefault(k, []).extend(v)

    # 3) Section-aware component selection, compose & crop
    results: Dict[str, Dict] = {}
    for sec, masks in groups.items():
        if not masks: continue
        union = np.maximum.reduce(masks)
        sel = _select_components(union, sec)

        cov = sel.mean()/255.0
        # guard rails against junk masks
        min_cov = 0.00002 if sec in ("footwear","accessories") else 0.0005
        if cov < min_cov or cov > 0.98:
            continue

        rgba = _compose_rgba(rgb_bg, sel)
        crop, bbox = _tight_crop_rgba(rgba, pad=6)
        results[sec] = {"rgba": crop, "bbox": bbox}
    return results
