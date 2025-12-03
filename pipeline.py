# pipeline.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import os, re, json, tempfile, requests
import numpy as np
import cv2
import torch
from PIL import Image

# Hugging Face (SegFormer + BLIP + SAM2)
from transformers import (
    pipeline as hf_pipeline,
    BlipProcessor,
    BlipForConditionalGeneration,
)

# ──────────────────────────────────────────────────────────────────────────────
# Optional: Google Gemini (“nano banana”) recommender (text-only)
# Set GOOGLE_API_KEY in env to enable. (HARD-CODE REMOVED)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")   # ← use env var only
_gemini_inited = False
_gemini_model_obj = None

def _init_gemini_if_needed():
    """Lazy-init Gemini client if API key present."""
    global _gemini_inited, _gemini_model_obj
    if _gemini_inited:
        return
    if HAS_GEMINI and _GOOGLE_API_KEY:
        genai.configure(api_key=_GOOGLE_API_KEY)
        _gemini_model_obj = genai.GenerativeModel(_GEMINI_MODEL)
    _gemini_inited = True

def _json_from_text(txt: str) -> Any:
    """Best-effort JSON extractor from LLM text (handles ```json ...``` blocks)."""
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:
        pass
    m = re.search(r"```json\s*(.+?)\s*```", txt, re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m2 = re.search(r"(\[\s*{.+}\s*\])", txt, re.S)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            pass
    return None

# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace Space (BiRefNet) background removal (fallback path)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from gradio_client import Client as HFClient, handle_file
    HAS_HF_CLIENT = True
except Exception:
    HAS_HF_CLIENT = False

HF_SPACE_ID  = "not-lain/background-removal"
HF_API_PNG   = "/png"
HF_API_IMAGE = "/image"

def _open_result_as_image(x) -> Image.Image:
    """Accept a PIL.Image or (remote/local) path from the Space and return RGBA image."""
    if isinstance(x, Image.Image):
        return x.convert("RGBA")
    s = str(x)
    if s.startswith(("http://", "https://")):
        r = requests.get(s, stream=True, timeout=30)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            for ch in r.iter_content(8192):
                tmp.write(ch)
            s = tmp.name
    return Image.open(s).convert("RGBA")

def remove_background_hf(img: Image.Image, hf_token: Optional[str] = None) -> Image.Image:
    """Try BiRefNet HF Space. If anything fails, return RGBA copy of the original."""
    if not HAS_HF_CLIENT:
        return img.convert("RGBA")
    try:
        client = HFClient(HF_SPACE_ID, hf_token=hf_token)
    except Exception:
        return img.convert("RGBA")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.convert("RGB").save(tmp.name, "PNG")
        in_path = tmp.name

    # Preferred /png
    try:
        out = client.predict(handle_file(in_path), api_name=HF_API_PNG)
        return _open_result_as_image(out)
    except Exception:
        pass

    # Fallback /image
    try:
        out = client.predict(Image.open(in_path).convert("RGB"), api_name=HF_API_IMAGE)
        cand = out[0] if isinstance(out, (list, tuple)) else out
        return _open_result_as_image(cand)
    except Exception:
        return img.convert("RGBA")

# ──────────────────────────────────────────────────────────────────────────────
# SAM2 (Hugging Face) for strong foreground (full person) mask
# NOTE: HF Transformers' SAM2 API is still evolving; we wrap in try/except and
#       fall back to HF Space remover if anything fails.
# ──────────────────────────────────────────────────────────────────────────────
_SAM2_AVAILABLE = True
try:
    from transformers import SamProcessor, SamModel
except Exception:
    _SAM2_AVAILABLE = False

def _sam2_try_full_rgba(image: Image.Image, sam_processor, sam_model, device: torch.device) -> Optional[Image.Image]:
    """
    Best-effort full-foreground cutout via SAM2. If anything fails, return None.
    Strategy:
      - Run SAM2 once to get masks
      - Pick largest mask as "person/foreground"
      - Compose RGBA with that mask as alpha
    """
    try:
        import numpy as np
        img_np = np.array(image.convert("RGB"))
        inputs = sam_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = sam_model(**inputs)

        # Try to get predicted masks; the exact tensor name can vary by release.
        # Prefer "pred_masks" if present, otherwise search common attrs.
        pred = getattr(outputs, "pred_masks", None)
        if pred is None:
            # Some variants expose "masks" or "low_res_masks"
            pred = getattr(outputs, "masks", None) or getattr(outputs, "low_res_masks", None)
        if pred is None:
            return None

        # Shape [B, N, h, w] or [N, h, w]
        pm = pred
        if pm.dim() == 4:
            pm = pm[0]  # take batch 0

        # Upscale to image size if needed
        H, W = img_np.shape[:2]
        pm_up = torch.nn.functional.interpolate(
            pm.unsqueeze(0).float(), size=(H, W), mode="bilinear", align_corners=False
        )[0]

        # Choose largest mask by area after threshold
        best_mask = None
        best_area = -1
        for i in range(pm_up.shape[0]):
            m = (pm_up[i] > 0).cpu().numpy().astype("uint8") * 255
            area = int(m.sum() // 255)
            if area > best_area:
                best_area = area
                best_mask = m
        if best_mask is None:
            return None

        # Compose RGBA
        rgba = image.convert("RGBA")
        alpha = Image.fromarray(best_mask, "L")
        out = Image.new("RGBA", rgba.size, (0,0,0,0))
        out.paste(rgba.convert("RGB"), mask=alpha)
        return out
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Models: SAM2 (full foreground) + SegFormer (clothes) + BLIP (caption)
# ──────────────────────────────────────────────────────────────────────────────
def load_models(device: int = -1) -> Tuple[None, Any]:
    """
    Returns (None, seg_bundle) where seg_bundle is a dict:
      {
        "segformer": hf clothes segmentation pipeline,
        "sam2_processor": SamProcessor or None,
        "sam2_model": SamModel or None,
        "device": torch.device
      }
    device = -1 for CPU; >=0 for CUDA index.
    """
    torch_device = torch.device(f"cuda:{device}" if device >= 0 and torch.cuda.is_available() else "cpu")

    # SegFormer for clothing categories (unchanged)
    segformer = hf_pipeline(
        "image-segmentation",
        model="mattmdjaga/segformer_b2_clothes",
        device=device,
    )

    # Try to load SAM2 (HF) for foreground
    sam2_processor = None
    sam2_model = None
    if _SAM2_AVAILABLE:
        try:
            sam2_processor = SamProcessor.from_pretrained("facebook/sam2-hiera-large")
            sam2_model = SamModel.from_pretrained("facebook/sam2-hiera-large").to(torch_device)
            sam2_model.eval()
        except Exception:
            sam2_processor = None
            sam2_model = None

    return None, {"segformer": segformer, "sam2_processor": sam2_processor, "sam2_model": sam2_model, "device": torch_device}

_blip_processor, _blip_model = None, None

def load_caption_model(device: str = "cpu"):
    global _blip_processor, _blip_model
    if _blip_processor is None or _blip_model is None:
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return _blip_processor, _blip_model

def describe_image(img: Image.Image, device="cpu") -> str:
    """Generate a short caption for a cropped garment with BLIP."""
    processor, model = load_caption_model(device)
    inputs = processor(img.convert("RGB"), return_tensors="pt").to(device)
    out = model.generate(**inputs, max_length=30)
    return processor.decode(out[0], skip_special_tokens=True)

# ──────────────────────────────────────────────────────────────────────────────
# Utilities: mapping, masks, crops, resizing, attribute parsing
# ──────────────────────────────────────────────────────────────────────────────
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
    try:
        if isinstance(mask_like, Image.Image):
            im = mask_like
        elif isinstance(mask_like, np.ndarray):
            arr = mask_like
            if arr.dtype.kind == "f":
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
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

def _select_components(mask: np.ndarray, section: str) -> np.ndarray:
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

    if section == "footwear":
        min_ar, max_ar, keep, min_dim = 0.0005, 0.25, 2, 0.025
    elif section == "accessories":
        min_ar, max_ar, keep, min_dim = 0.0004, 0.25, 3, 0.02
    else:
        min_ar, max_ar, keep, min_dim = 0.01, 0.85, 1, 0.10

    filtered = [c for c in comps if (min_ar <= c[2] <= max_ar and c[3] >= min_dim)]
    filtered.sort(key=lambda t: t[1], reverse=True)
    if not filtered:
        filtered = sorted(comps, key=lambda t: t[1], reverse=True)[:keep]

    out = np.zeros_like(binm, dtype=np.uint8)
    for i, *_ in filtered[:keep]:
        out[labels == i] = 255

    # little cleanup
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    k = 3 if section in ("footwear","accessories") else 5
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((k,k), np.uint8), 1)
    return out

def _tight_crop_rgba(rgba: Image.Image, pad: int = 6) -> Tuple[Image.Image, List[int]]:
    a = np.array(rgba.split()[-1])
    ys, xs = np.where(a > 0)
    if not (ys.size and xs.size):
        return rgba, [0,0,rgba.width,rgba.height]
    y0,y1 = ys.min(), ys.max()
    x0,x1 = xs.min(), xs.max()
    x0 = max(x0 - pad, 0); y0 = max(y0 - pad, 0)
    x1 = min(x1 + pad, rgba.width); y1 = min(y1 + pad, rgba.height)
    return rgba.crop((x0,y0,x1,y1)), [int(x0),int(y0),int(x1),int(y1)]

def _compose_with_bg_rgba(pre_rgba: Image.Image, mask_u8: np.ndarray) -> Image.Image:
    m1 = Image.fromarray(mask_u8, "L").resize(pre_rgba.size, Image.NEAREST)
    m2 = pre_rgba.split()[-1]
    comb = Image.fromarray(
        np.minimum(np.array(m1, dtype=np.uint8), np.array(m2, dtype=np.uint8)),
        "L"
    )
    out = Image.new("RGBA", pre_rgba.size, (0,0,0,0))
    out.paste(pre_rgba.convert("RGB"), mask=comb)
    return out

def resize_if_too_big(img: Image.Image, max_side: int = 1280) -> Image.Image:
    """Downscale very large images to keep CPU latency reasonable."""
    w, h = img.size
    side = max(w, h)
    if side <= max_side:
        return img
    scale = max_side / float(side)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)

# ──────────────────────────────────────────────────────────────────────────────
# Attribute extraction (very light) from BLIP caption
# ──────────────────────────────────────────────────────────────────────────────
_COLOR_WORDS = [
    "black","white","grey","gray","beige","blue","navy","light blue","red",
    "green","olive","brown","yellow","purple","pink","orange","khaki","cream","off white"
]
_FIT_WORDS = ["slim", "regular", "relaxed", "oversized", "tapered", "straight"]

def _guess_color(desc: str) -> Optional[str]:
    d = (desc or "").lower()
    for w in sorted(_COLOR_WORDS, key=len, reverse=True):
        if w in d:
            return w
    return None

def _guess_fit(desc: str) -> Optional[str]:
    d = (desc or "").lower()
    for w in _FIT_WORDS:
        if w in d:
            return w
    return None

def _guess_type_from_caption(desc: str, section: str) -> Optional[str]:
    d = (desc or "").lower()
    if section == "topwear":
        for t in ["shirt","t-shirt","tee","blouse","top","jacket","hoodie","sweater","cardigan","blazer","dress","coat"]:
            if t in d:
                return t
        return "top"
    if section == "bottomwear":
        for t in ["jeans","trousers","pants","shorts","skirt","leggings"]:
            if t in d:
                return t
        return "bottom"
    if section == "footwear":
        for t in ["sneakers","boots","heels","loafers","sandals","shoes"]:
            if t in d:
                return t
        return "shoes"
    if section == "accessories":
        for t in ["bag","belt","hat","scarf","glasses","sunglasses","watch","necklace","earrings"]:
            if t in d:
                return t
        return "accessory"
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Segmentation (SAM2 foreground → SegFormer categories) → crops per section
# ──────────────────────────────────────────────────────────────────────────────
def segment_image_with_hf(seg_bundle, img: Image.Image, hf_token: Optional[str], device="cpu") -> Dict[str, Dict]:
    """
    NEW: Use SAM2 (HF) to get a robust foreground (full) mask → RGBA,
         then run SegFormer (clothes) on that RGBA to get category masks.
    Falls back to HF Space remover if SAM2 is unavailable/fails.

    Returns dict with:
      - "full": {"rgba": full cutout, "bbox": ...}
      - per section: {"rgba": crop, "bbox": ...}
    """
    img_small = resize_if_too_big(img, max_side=1280)

    # 1) Foreground via SAM2 (or fallback)
    pre_rgba = None
    try:
        sam2_proc = seg_bundle.get("sam2_processor")
        sam2_model = seg_bundle.get("sam2_model")
        torch_device = seg_bundle.get("device")
        if sam2_proc is not None and sam2_model is not None:
            pre_rgba = _sam2_try_full_rgba(img_small, sam2_proc, sam2_model, torch_device)
    except Exception:
        pre_rgba = None

    if pre_rgba is None:
        # fallback to HF Space remover
        pre_rgba = remove_background_hf(img_small, hf_token=hf_token)

    # 2) Category segmentation on the cutout (RGB)
    segformer = seg_bundle["segformer"]
    outs_bg = segformer(pre_rgba.convert("RGB"))
    groups = _collect_masks(outs_bg)

    results: Dict[str, Dict] = {"full": {"rgba": pre_rgba, "bbox": [0, 0, pre_rgba.width, pre_rgba.height]}}
    for sec, masks in groups.items():
        if not masks:
            continue
        union = np.maximum.reduce(masks)

        # clip category mask by the actual alpha of pre_rgba to avoid background noise
        alpha = np.array(pre_rgba.split()[-1])
        union = np.minimum(union, (alpha > 0).astype(np.uint8) * 255)

        sel = _select_components(union, sec)
        cov = sel.mean() / 255.0
        floor = 0.0006 if sec in ("footwear","accessories") else 0.005
        if cov < floor or cov > 0.98:
            continue
        rgba = _compose_with_bg_rgba(pre_rgba, sel)
        crop, bbox = _tight_crop_rgba(rgba, pad=6)
        results[sec] = {"rgba": crop, "bbox": bbox}
    return results

# ──────────────────────────────────────────────────────────────────────────────
# Local text-only recommender (heuristic, max 3 combos)
# ──────────────────────────────────────────────────────────────────────────────
def recommend_from_captions(results: Dict[str, Dict]) -> List[Dict[str, Any]]:
    tops = [sec for sec in results if sec == "topwear"]
    bottoms = [sec for sec in results if sec == "bottomwear"]
    shoes = [sec for sec in results if sec == "footwear"]
    accs = [sec for sec in results if sec == "accessories"]

    combos: List[Dict[str, Any]] = []
    if tops and bottoms:
        top_desc = results[tops[0]].get("description", "a top")
        bot_desc = results[bottoms[0]].get("description", "bottoms")
        combos.append({
            "combo": [tops[0], bottoms[0]],
            "description": f"{top_desc} paired with {bot_desc} for a balanced look."
        })
        if shoes:
            shoe_desc = results[shoes[0]].get("description", "shoes")
            combos.append({
                "combo": [tops[0], bottoms[0], shoes[0]],
                "description": f"{top_desc}, {bot_desc}, and {shoe_desc} for a complete outfit."
            })
        if accs:
            acc_desc = results[accs[0]].get("description", "an accessory")
            combos.append({
                "combo": [tops[0], bottoms[0], accs[0]],
                "description": f"{top_desc} with {bot_desc}, styled using {acc_desc}."
            })
    if not combos:
        combos.append({"combo": [], "description": "Not enough items to build a recommendation."})
    return combos[:3]

# ──────────────────────────────────────────────────────────────────────────────
# Gemini-based recommender (max 3 combos), JSON only, fallback to local
# ──────────────────────────────────────────────────────────────────────────────
def _items_from_results(results: Dict[str, Dict]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for sec in ("topwear","bottomwear","footwear","accessories"):
        if sec in results and "description" in results[sec]:
            desc = results[sec]["description"]
            if isinstance(desc, str) and desc.strip():
                items.append({"section": sec, "description": desc.strip()})
    return items

def gemini_recommend(results: Dict[str, Dict]) -> Optional[List[Dict[str, Any]]]:
    _init_gemini_if_needed()
    if not (_gemini_model_obj and _GOOGLE_API_KEY):
        return None

    items = _items_from_results(results)
    if not items:
        return None

    prompt = f"""
You are a fashion stylist. Given these detected items (from image analysis):
{json.dumps(items, ensure_ascii=False)}

Rules:
- Return at most 3 outfits as a JSON array.
- Each outfit object must have:
  - "combo": list of sections used, choosing from ["topwear","bottomwear","footwear","accessories"]
  - "reason": one short sentence.
- Use only sections that exist in the input list. Avoid duplicates. Keep it concise.
JSON ONLY, no prose.
"""
    try:
        resp = _gemini_model_obj.generate_content(prompt)
        txt = getattr(resp, "text", "") or ""
        data = _json_from_text(txt)
        if not isinstance(data, list):
            return None
        cleaned = []
        allowed = [x["section"] for x in items]
        for it in data[:3]:
            combo = [c for c in it.get("combo", []) if c in allowed]
            reason = (it.get("reason") or "").strip()
            if combo and reason:
                cleaned.append({"combo": combo, "description": reason})
        return cleaned or None
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Public orchestrator: Segment → Describe → (Optional) Recommend
# ──────────────────────────────────────────────────────────────────────────────
def segment_and_describe(
    seg_bundle,
    cap_proc,
    cap_model,
    img: Image.Image,
    hf_token: Optional[str],
    device="cpu",
    run_recommender: bool = False
) -> Dict[str, Dict]:
    """
    Main entry:
      1) SAM2 → robust foreground; fallback to HF Space remover
      2) SegFormer → category masks on the cutout
      3) BLIP caption per crop
      4) Extract lightweight attributes (color/type/fit)
      5) Optional up to 3 text-only combos via Gemini; fallback to local heuristic
    """
    results = segment_image_with_hf(seg_bundle, img, hf_token, device=device)

    for sec, data in list(results.items()):
        if "rgba" not in data:
            continue
        try:
            desc = describe_image(data["rgba"], device=device)
        except Exception:
            desc = "unknown item"
        # attach caption + attrs
        attrs = {
            "color": _guess_color(desc),
            "type": _guess_type_from_caption(desc, sec),
            "fit": _guess_fit(desc),
        }
        data["description"] = desc
        data["attrs"] = attrs

    if run_recommender:
        combos = gemini_recommend(results)
        if combos is None:
            combos = recommend_from_captions(results)
        results["outfit_combos"] = combos

    return results
