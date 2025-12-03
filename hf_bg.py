# hf_bg.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import tempfile
import numpy as np
from PIL import Image

# HF Space client
try:
    from gradio_client import Client as HFClient, handle_file
    HAS_HF_CLIENT = True
except Exception:
    HAS_HF_CLIENT = False

# Optional local fallback
try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except Exception:
    HAS_REMBG = False

HF_SPACE_ID  = "not-lain/background-removal"
HF_API_PNG   = "/png"
HF_API_IMAGE = "/image"


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


def _has_transparency(img: Image.Image) -> bool:
    if img.mode != "RGBA":
        return False
    a = np.asarray(img.split()[-1])
    return a.min() < 255


def _alpha_mask(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        return Image.new("L", img.size, 255)
    return img.split()[-1]


def _rembg_fallback(img: Image.Image) -> Tuple[Image.Image, str]:
    if not HAS_REMBG:
        return img.convert("RGBA"), "identity"
    try:
        out = rembg_remove(img.convert("RGBA"))
        return out, "rembg"
    except Exception:
        return img.convert("RGBA"), "identity"


def _hf_space(img: Image.Image, hf_token: Optional[str]) -> Tuple[Image.Image, str]:
    if not HAS_HF_CLIENT:
        return img.convert("RGBA"), "no-hf-client"

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.convert("RGB").save(tmp.name, "PNG")
        in_path = tmp.name

    try:
        client = HFClient(HF_SPACE_ID, hf_token=hf_token)
    except Exception:
        return img.convert("RGBA"), "hf-client-fail"

    # 1) /png (best)
    try:
        out = client.predict(handle_file(in_path), api_name=HF_API_PNG)
        res = _open_result_as_image(out)
        if _has_transparency(res):
            return res, "hf-png"
    except Exception:
        pass

    # 2) /image
    try:
        out = client.predict(Image.open(in_path).convert("RGB"), api_name=HF_API_IMAGE)
        cand = out[0] if isinstance(out, (list, tuple)) else out
        res = _open_result_as_image(cand)
        if _has_transparency(res):
            return res, "hf-image"
    except Exception:
        pass

    return img.convert("RGBA"), "hf-no-alpha"


def remove_background(
    img: Image.Image,
    hf_token: Optional[str] = None,
    prefer: str = "auto",  # "auto" | "hf" | "rembg"
) -> Dict[str, Any]:
    """
    Returns:
      {
        "image": RGBA,
        "backend": "hf-png"|"hf-image"|"rembg"|"identity"|...,
        "had_alpha": bool,
        "alpha": L-mode alpha image
      }
    """
    img = img.convert("RGB")

    if prefer == "rembg":
        rgba, backend = _rembg_fallback(img)
        return {"image": rgba, "backend": backend, "had_alpha": _has_transparency(rgba), "alpha": _alpha_mask(rgba)}

    if prefer == "hf":
        rgba, backend = _hf_space(img, hf_token)
        # ensure transparency; if not, try rembg
        if not _has_transparency(rgba):
            rb, rb_backend = _rembg_fallback(img)
            rgba, backend = rb, f"{backend}->${rb_backend}"
        return {"image": rgba, "backend": backend, "had_alpha": _has_transparency(rgba), "alpha": _alpha_mask(rgba)}

    # auto: try HF, fall back to rembg if no alpha
    rgba, backend = _hf_space(img, hf_token)
    if not _has_transparency(rgba):
        rb, rb_backend = _rembg_fallback(img)
        rgba, backend = rb, rb_backend if rb_backend != "identity" else backend
    return {"image": rgba, "backend": backend, "had_alpha": _has_transparency(rgba), "alpha": _alpha_mask(rgba)}
