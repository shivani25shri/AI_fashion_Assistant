# app.py
from __future__ import annotations

import io
import os
import base64
import zipfile
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
    PlainTextResponse,
    HTMLResponse,
)
from PIL import Image

from pipeline import load_models, segment_image_with_hf
from vlm_recommender_qwen import LocalVLM_Qwen


app = FastAPI(title="Wardrobe Segmentation API", version="1.1.0")

SEG_PIPELINE = None
QWEN: LocalVLM_Qwen | None = None
HF_TOKEN = os.getenv("HF_TOKEN")


@app.on_event("startup")
def _load_models():
    global SEG_PIPELINE
    _, SEG_PIPELINE_LOCAL = load_models(device=-1)  # CPU default
    SEG_PIPELINE = SEG_PIPELINE_LOCAL
    print("Device set to use cpu")


# ---------------- helpers ----------------
def _read_upload_to_pil(f: UploadFile) -> Image.Image:
    raw = f.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail=f"Empty file: {f.filename}")
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return buf.getvalue()


def _to_data_url(img: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes(img)).decode("ascii")


def _collect_items_for_vlm(seg: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for sec, data in seg.items():
        if sec == "full":
            continue
        items.append({"section": sec, "rgba": data["rgba"]})
    return items


def _render_gallery_html(per_file_results: List[Dict[str, Any]]) -> str:
    # Inline CSS; simple grid.
    css = """
    <style>
      body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial,sans-serif;
           background:#0b0e12;color:#eef; margin:20px}
      h1{font-size:22px;margin:0 0 12px}
      .file{margin:18px 0;padding:16px;border:1px solid #2a2f3a;border-radius:12px;background:#11151b}
      .name{opacity:.8;margin-bottom:10px}
      .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px}
      .card{background:#0f1320;border:1px solid #252a36;border-radius:12px;padding:10px}
      .cap{font-size:12px;opacity:.7;margin:6px 0 0}
      img{width:100%;height:auto;border-radius:8px;background:#222}
      .badge{display:inline-block;background:#6a5acd33;border:1px solid #6a5acd55;color:#cfd;
             border-radius:6px;padding:2px 8px;font-size:12px;margin-left:8px}
    </style>
    """
    # Build cards
    parts = ["<html><head><meta charset='utf-8'><title>Segment Gallery</title>", css, "</head><body>"]
    parts.append("<h1>Segment Gallery</h1>")
    for entry in per_file_results:
        parts.append(f"<div class='file'><div class='name'>{entry['file']}</div>")
        parts.append("<div class='grid'>")
        for sec in ["full", "topwear", "bottomwear", "footwear", "accessories"]:
            obj = entry.get(sec)
            if not obj:
                continue
            parts.append("<div class='card'>")
            parts.append(f"<img src='{obj['png']}' alt='{sec}'>")
            parts.append(f"<div class='cap'>{sec.title()}<span class='badge'>{obj['width']}×{obj['height']}</span></div>")
            parts.append("</div>")
        parts.append("</div></div>")
    parts.append("</body></html>")
    return "".join(parts)


# ---------------- routes ----------------
@app.get("/", response_class=PlainTextResponse)
def root():
    return "Wardrobe API up. See /docs, or /segment_gallery for an HTML demo."


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/segment")
def segment(
    files: List[UploadFile] = File(..., description="One or more image files"),
    embed_png: bool = Query(False, description="If true, include base64 PNGs in JSON"),
):
    if SEG_PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    results = []
    for f in files:
        img = _read_upload_to_pil(f)
        W, H = img.size
        crops = segment_image_with_hf(SEG_PIPELINE, img, hf_token=HF_TOKEN)

        entry: Dict[str, Any] = {"file": f.filename, "status": "ok", "original_size": [W, H]}
        for sec, data in crops.items():
            rgba: Image.Image = data["rgba"]
            sec_obj: Dict[str, Any] = {
                "section": sec,
                "bbox": data.get("bbox", [0, 0, rgba.width, rgba.height]),
                "width": rgba.width,
                "height": rgba.height,
            }
            if embed_png:
                sec_obj["png"] = _to_data_url(rgba)
            entry[sec] = sec_obj
        results.append(entry)

    return {"results": results}


@app.get("/segment_gallery", response_class=HTMLResponse)
def segment_gallery_form():
    # tiny upload form to use from a browser
    return HTMLResponse("""
      <html><head><meta charset="utf-8"><title>Segment Gallery</title></head>
      <body style="font-family:system-ui;margin:30px">
        <h2>Upload images to see background-removed + categories</h2>
        <form action="/segment_gallery" method="post" enctype="multipart/form-data">
          <input type="file" name="files" multiple accept="image/*">
          <button type="submit">Segment</button>
        </form>
        <p>Tip: You can also POST to <code>/segment?embed_png=true</code> for JSON with inline images.</p>
      </body></html>
    """)


@app.post("/segment_gallery", response_class=HTMLResponse)
def segment_gallery(files: List[UploadFile] = File(...)):
    if SEG_PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    per_file_results: List[Dict[str, Any]] = []
    for f in files:
        img = _read_upload_to_pil(f)
        crops = segment_image_with_hf(SEG_PIPELINE, img, hf_token=HF_TOKEN)

        entry: Dict[str, Any] = {"file": f.filename}
        for sec, data in crops.items():
            rgba: Image.Image = data["rgba"]
            entry[sec] = {
                "png": _to_data_url(rgba),
                "width": rgba.width,
                "height": rgba.height,
            }
        per_file_results.append(entry)

    return HTMLResponse(_render_gallery_html(per_file_results))


@app.post("/segment_zip")
def segment_zip(files: List[UploadFile] = File(...)):
    if SEG_PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        manifest: Dict[str, Any] = {"results": []}
        for f in files:
            img = _read_upload_to_pil(f)
            crops = segment_image_with_hf(SEG_PIPELINE, img, hf_token=HF_TOKEN)
            stem = Path(f.filename).stem
            rec: Dict[str, Any] = {"file": f.filename, "sections": {}}
            for sec, data in crops.items():
                path = f"{stem}/{sec}.png"
                z.writestr(path, _png_bytes(data["rgba"]))
                rec["sections"][sec] = {"path": path, "bbox": data.get("bbox")}
            manifest["results"].append(rec)
        z.writestr("manifest.json", _png_bytes(Image.new("RGBA",(1,1))))  # keep zip openable
        # better: write real JSON
        z.writestr("manifest.json", __import__("json").dumps(manifest, indent=2).encode())

    mem.seek(0)
    return StreamingResponse(
        mem, media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=segments.zip"},
    )


@app.post("/recommend")
def recommend(files: List[UploadFile] = File(...)):
    if SEG_PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    global QWEN
    if QWEN is None:
        QWEN = LocalVLM_Qwen(device="cpu", max_new_tokens=128, temperature=0.2, max_image_side=320, seed=42, cpu_threads=4)

    all_items: List[Dict[str, Any]] = []
    per_file: List[Dict[str, Any]] = []

    for f in files:
        img = _read_upload_to_pil(f)
        crops = segment_image_with_hf(SEG_PIPELINE, img, hf_token=HF_TOKEN)
        items = _collect_items_for_vlm(crops)
        per_file.append({"file": f.filename, "found": [it["section"] for it in items]})
        all_items.extend(items)

    if not all_items:
        return {"files": per_file, "recommendations": [], "message": "No garments detected."}

    recs = QWEN.recommend(all_items)
    return {"files": per_file, "recommendations": recs}