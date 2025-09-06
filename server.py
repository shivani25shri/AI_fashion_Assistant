# server.py
from __future__ import annotations
import os, io, base64, uuid, requests
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np

# your local modules
from pipeline import load_models, segment_image_with_hf
from vlm_recommender_qwen import LocalVLM_Qwen


# ---------- utils ----------
def pil_to_b64_png(img: Image.Image) -> str:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def b64_to_pil(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGBA")

def fetch_to_pil(url: str) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGBA")

def make_composite(items_rgba: List[Image.Image],
                   canvas=(1024, 1024)) -> Image.Image:
    """
    Simple horizontal arrangement of transparent cutouts.
    You can replace with a smarter layout later.
    """
    W, H = canvas
    out = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    if not items_rgba:
        return out
    cols = len(items_rgba)
    cell_w = int(W / max(cols, 1))
    pad = int(0.06 * W)

    x = pad
    for im in items_rgba:
        # scale to fit cell height
        w, h = im.size
        target_h = int(H * 0.8)
        scale = target_h / h
        nw, nh = int(w * scale), int(h * scale)
        im_r = im.resize((nw, nh), Image.LANCZOS)
        y = (H - nh) // 2
        out.alpha_composite(im_r, dest=(x, y))
        x += cell_w
    return out


# ---------- load once ----------
detector, seg_pipeline = load_models(device=-1)   # detector unused here, seg used
vlm = LocalVLM_Qwen(device=("cuda" if os.environ.get("CUDA","0")=="1" else "cpu"),
                    max_new_tokens=128, temperature=0.2, max_image_side=320)

HF_TOKEN = os.getenv("HF_TOKEN")

# ---------- app ----------
app = FastAPI(title="Wardrobe AI Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/health")
def health():
    return {"ok": True}


# ---------- /segment ----------
@app.post("/segment")
async def segment(
    files: List[UploadFile] = File(...),
    return_images: str = Form("base64")  # base64 | url (url requires you to wire S3; base64 is default)
):
    results: List[Dict[str, Any]] = []
    for uf in files:
        try:
            img = Image.open(io.BytesIO(await uf.read())).convert("RGB")
            H, W = img.height, img.width

            crops = segment_image_with_hf(seg_pipeline, img, hf_token=HF_TOKEN)  # includes "full"
            entry: Dict[str, Any] = {
                "file": uf.filename,
                "status": "ok",
                "original_size": [W, H],
                "bg_removed_size": [crops["full"]["rgba"].width, crops["full"]["rgba"].height],
            }

            # full
            full_png = pil_to_b64_png(crops["full"]["rgba"])
            entry["full"] = {
                "section": "full",
                "bbox": crops["full"]["bbox"],
                "png_b64": full_png
            }

            # items
            items: List[Dict[str, Any]] = []
            for sec in ("topwear", "bottomwear", "footwear", "accessories"):
                if sec not in crops:
                    continue
                it = crops[sec]
                items.append({
                    "id": f"{uf.filename}_{sec}_{uuid.uuid4().hex[:6]}",
                    "section": sec,
                    "bbox": it["bbox"],
                    "png_b64": pil_to_b64_png(it["rgba"])
                })
            entry["items"] = items
            results.append(entry)
        except Exception as e:
            results.append({"file": uf.filename, "status": f"error: {e}"})

    return {"results": results}


# ---------- /recommend ----------
class RecommendItem(BaseModel):
    id: str
    section: str
    png_b64: Optional[str] = None
    png_url: Optional[str] = None

class RecommendReq(BaseModel):
    items: List[RecommendItem]
    prompt: Optional[str] = ""

@app.post("/recommend")
def recommend(req: RecommendReq):
    # Build items for the VLM
    items_vlm: List[Dict[str, Any]] = []
    index_map: List[Dict[str, Any]] = []

    for i, it in enumerate(req.items, start=1):
        if it.png_b64:
            img = b64_to_pil(it.png_b64)
        elif it.png_url:
            img = fetch_to_pil(it.png_url)
        else:
            continue
        items_vlm.append({"section": it.section, "rgba": img})
        index_map.append({"index": i, "id": it.id})

    recs = vlm.recommend(items_vlm)  # [{"combo":[...],"description":"..."}]

    # Attach the user's free-text intent (not used by model above, but useful)
    return {"index_map": index_map, "recommendations": recs, "prompt": req.prompt or ""}


# ---------- /outfits ----------
class SaveOutfitReq(BaseModel):
    name: Optional[str] = "Outfit"
    items: List[RecommendItem]
    generate_composite: bool = True

@app.post("/outfits")
def save_outfit(req: SaveOutfitReq):
    imgs: List[Image.Image] = []
    for it in req.items:
        if it.png_b64:
            imgs.append(b64_to_pil(it.png_b64))
        elif it.png_url:
            imgs.append(fetch_to_pil(it.png_url))

    composite_b64 = None
    if req.generate_composite and imgs:
        comp = make_composite(imgs)
        composite_b64 = pil_to_b64_png(comp)

    return {
        "outfit_id": f"of_{uuid.uuid4().hex[:10]}",
        "name": req.name,
        "composite_png_b64": composite_b64
    }
