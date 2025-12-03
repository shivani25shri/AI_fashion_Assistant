# server.py
from __future__ import annotations

import os, io, base64, uuid, time, json, hashlib, requests
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from functools import lru_cache

# ── Env ───────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ── FastAPI / Pydantic ────────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

# ── Imaging ───────────────────────────────────────────────────────────────────
from PIL import Image, ImageDraw

# ── Local modules ─────────────────────────────────────────────────────────────
from pipeline import load_models, segment_image_with_hf
from supabase_utils import (
    upload_json_to_storage,
    persist_recommendation as persist_rec_utils,
    add_catalog_items_bulk,
    fetch_catalog as sb_fetch_catalog,
    recommend_from_supabase as sb_recommend_from_catalog,
    get_client,
)

# ── OpenAI (optional at startup; required by /recommend) ─────────────────────
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_IMAGE_MAX_SIDE = int(os.getenv("OPENAI_IMAGE_MAX_SIDE", "384"))
OPENAI_MIN_INTERVAL_SEC = float(os.getenv("OPENAI_MIN_INTERVAL_SEC", "6"))

ocli: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
_last_openai_ts = 0.0
def _respect_rate_limit():
    global _last_openai_ts
    now = time.time()
    wait = (_last_openai_ts + OPENAI_MIN_INTERVAL_SEC) - now
    if wait > 0:
        time.sleep(wait)
    _last_openai_ts = time.time()

# ── Supabase storage client (images + previews) ───────────────────────────────
from supabase import create_client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE") or os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "wadrobe_items")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE (or SUPABASE_KEY) in the environment")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Report backends (helps you verify we’re using the HF Space + SegFormer) ──
BACKENDS_META = {
    "bg": "hf_birefnet",  # pipeline.remove_background_hf()
    "seg": "mattmdjaga/segformer_b2_clothes"
}

# ── Storage helpers ───────────────────────────────────────────────────────────
def _img_to_png_bytes(img: Image.Image) -> bytes:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def upload_png_to_supabase(img: Image.Image, key_prefix: str = "") -> str:
    key = f"{key_prefix}{uuid.uuid4().hex}.png"
    body = _img_to_png_bytes(img)
    sb.storage.from_(SUPABASE_BUCKET).upload(
        path=key,
        file=body,
        file_options={"content-type": "image/png", "cache-control": "31536000, immutable", "upsert": "true"},
    )
    url = sb.storage.from_(SUPABASE_BUCKET).get_public_url(key)
    return url if isinstance(url, str) else url.get("publicUrl")  # type: ignore

def _infer_image_ct(filename: str) -> str:
    name = (filename or "").lower()
    if name.endswith(".jpg") or name.endswith(".jpeg"):
        return "image/jpeg"
    if name.endswith(".webp"):
        return "image/webp"
    if name.endswith(".png"):
        return "image/png"
    return "application/octet-stream"

def upload_bytes_to_supabase(data: bytes, key_prefix: str, content_type: str) -> str:
    key = f"{key_prefix}{uuid.uuid4().hex}"
    ext_map = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
    key = key + ext_map.get(content_type, "")
    sb.storage.from_(SUPABASE_BUCKET).upload(
        path=key,
        file=data,
        file_options={"content-type": content_type, "cache-control": "31536000, immutable", "upsert": "true"},
    )
    url = sb.storage.from_(SUPABASE_BUCKET).get_public_url(key)
    return url if isinstance(url, str) else url.get("publicUrl")  # type: ignore

# ── Image utils ───────────────────────────────────────────────────────────────
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

def make_composite(items_rgba: List[Image.Image], canvas=(1024, 768)) -> Image.Image:
    W, H = canvas
    out = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    x = 20
    for im in items_rgba or []:
        w, h = im.size
        scale = min(1.0, (H * 0.8) / max(1, h))
        nw, nh = int(w * scale), int(h * scale)
        im_r = im.resize((nw, nh), Image.LANCZOS)
        y = (H - nh) // 2
        out.alpha_composite(im_r, dest=(x, y))
        x += nw + 20
    return out

def build_recommend_preview(items_rgba: List[Image.Image], canvas=(1024, 512)) -> Image.Image:
    W, H = canvas
    out = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    if not items_rgba:
        return out
    gap = 20
    x = gap
    max_h = int(H * 0.85)
    for im in items_rgba:
        w, h = im.size
        scale = min(1.0, max_h / max(1, h))
        nw, nh = int(w * scale), int(h * scale)
        im_r = im.resize((max(1, nw), max(1, nh)), Image.LANCZOS)
        y = (H - nh) // 2
        out.alpha_composite(im_r, dest=(x, y))
        x += nw + gap
        if x > W - gap:
            break
    try:
        d = ImageDraw.Draw(out)
        d.text((12, 12), "recommend preview", fill=(0, 0, 0, 128))
    except Exception:
        pass
    return out

def _inline_png_for_openai(img: Image.Image) -> Dict[str, Any]:
    w, h = img.size
    s = max(w, h)
    if s > OPENAI_IMAGE_MAX_SIDE and s > 0:
        scale = OPENAI_IMAGE_MAX_SIDE / float(s)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

# ── Extra helpers for URL-based segment ───────────────────────────────────────
def _guess_ct_from_url(url: str) -> str:
    path = urlparse(url).path.lower()
    if path.endswith((".jpg", ".jpeg")): return "image/jpeg"
    if path.endswith(".png"): return "image/png"
    if path.endswith(".webp"): return "image/webp"
    return "application/octet-stream"

def fetch_image_bytes(url: str, timeout: int = 30) -> tuple[bytes, str, str]:
    """Return (bytes, content_type, filename_guess) for a remote image URL."""
    r = requests.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    ct = (r.headers.get("Content-Type") or _guess_ct_from_url(url)).split(";")[0].strip()
    name = os.path.basename(urlparse(url).path) or f"{uuid.uuid4().hex}"
    if "." not in name:
        ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(ct, "")
        name = name + ext
    return r.content, ct, name

# ── Models ────────────────────────────────────────────────────────────────────
detector, seg_pipeline = load_models(device=-1)  # detector unused
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI(title="Wardrobe AI Backend (Supabase Catalog + Segment + Recommend)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h3>Wardrobe AI Backend</h3>
    <ul>
      <li>POST <code>/segment</code> → stores original + crops as CDN images (multipart)</li>
      <li>POST <code>/segment/url</code> → same as above but JSON with CDN URLs</li>
      <li>POST <code>/segment/url/ingest</code> → segment from CDN URL(s) and insert into catalog</li>
      <li>POST <code>/catalog/ingest_segment</code> → add segmented items to catalog</li>
      <li>POST <code>/catalog/add</code> → add any items (section, image_url, description)</li>
      <li>GET  <code>/catalog</code> → list catalog items</li>
      <li>POST <code>/recommend</code> → OpenAI suggestions + preview + persisted JSON</li>
      <li>POST <code>/recommend/from_catalog</code> → catalog-only recommender</li>
      <li>POST <code>/outfits</code> → build composite image</li>
      <li>GET  <code>/health</code></li>
      <li>Docs: <a href="/docs">/docs</a></li>
    </ul>
    """

@app.get("/health")
def health():
    return {"ok": True}

# ── /segment (multipart) ──────────────────────────────────────────────────────
@app.post("/segment")
async def segment(
    files: List[UploadFile] = File(...),
    user_id: Optional[str] = Query(None, description="Optional storage namespace, e.g. 'shivani'"),
    also_embed_png: bool = Query(False, description="Include base64 alongside URLs (debug)")
):
    results: List[Dict[str, Any]] = []
    ns = f"{user_id.strip()}/" if user_id else ""

    for uf in files:
        try:
            raw = await uf.read()
            orig_ct = _infer_image_ct(uf.filename)
            source_url = upload_bytes_to_supabase(raw, key_prefix=f"{ns}source/", content_type=orig_ct)

            img = Image.open(io.BytesIO(raw)).convert("RGB")
            crops = segment_image_with_hf(seg_pipeline, img, hf_token=HF_TOKEN)

            entry: Dict[str, Any] = {
                "file": uf.filename,
                "status": "ok",
                "source": {"content_type": orig_ct, "url": source_url},
                "items": [],
                "backend": BACKENDS_META,   # ← meta
            }

            # full
            full_im = crops["full"]["rgba"]
            full_url = upload_png_to_supabase(full_im, key_prefix=f"{ns}full/")
            full = {"section": "full", "bbox": crops["full"]["bbox"], "url": full_url}
            if also_embed_png:
                full["png_b64"] = pil_to_b64_png(full_im)
            entry["full"] = full

            # categories
            for sec in ("topwear", "bottomwear", "footwear", "accessories"):
                if sec not in crops:
                    continue
                im = crops[sec]["rgba"]
                url = upload_png_to_supabase(im, key_prefix=f"{ns}{sec}/")
                item = {
                    "id": f"{uf.filename}_{sec}_{uuid.uuid4().hex[:6]}",
                    "section": sec,
                    "bbox": crops[sec]["bbox"],
                    "url": url
                }
                if also_embed_png:
                    item["png_b64"] = pil_to_b64_png(im)
                entry["items"].append(item)

            results.append(entry)
        except Exception as e:
            results.append({"file": uf.filename, "status": f"error: {e}"})

    return {"results": results}

# ── /segment/url (JSON with CDN image URLs) ───────────────────────────────────
class SegmentURLReq(BaseModel):
    image_url: Union[str, List[str]]
    user_id: Optional[str] = None
    also_embed_png: bool = False

@app.post("/segment/url")
def segment_from_url(req: SegmentURLReq):
    urls: List[str] = [req.image_url] if isinstance(req.image_url, str) else list(req.image_url)
    ns = f"{req.user_id.strip()}/" if req.user_id else ""
    results: List[Dict[str, Any]] = []

    for u in urls:
        try:
            raw, ct, filename = fetch_image_bytes(u)
            # 1) save ORIGINAL to /source/
            source_url = upload_bytes_to_supabase(raw, key_prefix=f"{ns}source/", content_type=ct)
            # 2) segment
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            crops = segment_image_with_hf(seg_pipeline, img, hf_token=HF_TOKEN)

            entry: Dict[str, Any] = {
                "file": filename,
                "status": "ok",
                "source": {"content_type": ct, "url": source_url},
                "items": [],
                "backend": BACKENDS_META,   # ← meta
            }

            # full
            if "full" in crops:
                full_im = crops["full"]["rgba"]
                full_url = upload_png_to_supabase(full_im, key_prefix=f"{ns}full/")
                full = {"section": "full", "bbox": crops["full"]["bbox"], "url": full_url}
                if req.also_embed_png:
                    full["png_b64"] = pil_to_b64_png(full_im)
                entry["full"] = full

            # categories
            for sec in ("topwear", "bottomwear", "footwear", "accessories"):
                if sec not in crops:
                    continue
                im = crops[sec]["rgba"]
                url_cdn = upload_png_to_supabase(im, key_prefix=f"{ns}{sec}/")
                item = {
                    "id": f"{filename}_{sec}_{uuid.uuid4().hex[:6]}",
                    "section": sec,
                    "bbox": crops[sec]["bbox"],
                    "url": url_cdn,
                }
                if req.also_embed_png:
                    item["png_b64"] = pil_to_b64_png(im)
                entry["items"].append(item)

            results.append(entry)
        except Exception as e:
            results.append({"file": os.path.basename(urlparse(u).path) or u, "status": f"error: {e}"})

    return {"results": results}

# ── /segment/url/ingest → segment and auto-insert into catalog ───────────────
class SegmentURLIngestReq(BaseModel):
    image_url: Union[str, List[str]]
    user_id: str
    # Optional static descriptions per section (fallback to "topwear item" etc.)
    default_descriptions: Optional[Dict[str, str]] = None

@app.post("/segment/url/ingest")
def segment_url_and_ingest(req: SegmentURLIngestReq):
    """
    Segments CDN image(s), uploads crops to Supabase, and immediately inserts
    topwear/bottomwear/footwear/accessories into the `catalog` table.
    Returns both the segmentation results and the inserted catalog rows.
    """
    seg = segment_from_url(SegmentURLReq(image_url=req.image_url, user_id=req.user_id, also_embed_png=False))
    all_inserted: List[Dict[str, Any]] = []

    # Flatten items -> build rows for catalog
    for entry in seg["results"]:
        if entry.get("status") != "ok":
            continue
        rows = []
        for it in entry.get("items", []):
            sec = it["section"]
            desc = (req.default_descriptions or {}).get(sec) or f"{sec} item"
            rows.append({
                "section": sec,
                "description": desc,
                "image_url": it["url"],
                "attrs": {},
                "extra": {"source_file": entry.get("file")},
            })
        if rows:
            inserted = add_catalog_items_bulk(req.user_id, rows)
            all_inserted.extend(inserted)

    return {"segmentation": seg, "inserted": all_inserted, "count": len(all_inserted)}

# ── Catalog models & endpoints ────────────────────────────────────────────────
class CatalogItemIn(BaseModel):
    section: str = Field(pattern="^(topwear|bottomwear|footwear|accessories)$")
    image_url: str
    description: Optional[str] = ""
    attrs: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = None

class CatalogIngestFromSegmentReq(BaseModel):
    user_id: str
    items: List[CatalogItemIn]

@app.post("/catalog/ingest_segment")
def catalog_ingest_segment(req: CatalogIngestFromSegmentReq = Body(...)):
    rows = [{
        "section": it.section,
        "description": it.description or f"{it.section} item",
        "image_url": it.image_url,
        "attrs": it.attrs or {},
        "extra": it.extra or {},
    } for it in req.items]
    inserted = add_catalog_items_bulk(req.user_id, rows)
    return {"inserted": inserted, "count": len(inserted)}

class CatalogAddReq(BaseModel):
    user_id: str
    items: List[CatalogItemIn]

@app.post("/catalog/add")
def catalog_add(req: CatalogAddReq = Body(...)):
    rows = [{
        "section": it.section,
        "description": it.description or f"{it.section} item",
        "image_url": it.image_url,
        "attrs": it.attrs or {},
        "extra": it.extra or {},
    } for it in req.items]
    inserted = add_catalog_items_bulk(req.user_id, rows)
    return {"inserted": inserted, "count": len(inserted)}

@app.get("/catalog")
def catalog_list(
    user_id: str = Query(...),
    section: Optional[str] = Query(None, pattern="^(topwear|bottomwear|footwear|accessories)$"),
    limit: int = Query(100, ge=1, le=1000),
):
    rows = sb_fetch_catalog(user_id=user_id, section=section, limit=limit)
    return {"items": rows, "count": len(rows)}

# ── OpenAI recommend (vision) ─────────────────────────────────────────────────
class RecommendItem(BaseModel):
    id: str
    section: str
    url: Optional[str] = None
    png_url: Optional[str] = None
    png_b64: Optional[str] = None

class RecommendReq(BaseModel):
    items: List[RecommendItem]
    prompt: Optional[str] = ""

def _fallback_from_sections(items_vlm: List[Dict[str, Any]]) -> Dict[str, Any]:
    combos = []
    for i, it in enumerate(items_vlm, start=1):
        sec = it["section"]
        if   sec == "topwear":     suggest = ["bottomwear", "footwear", "accessories"]
        elif sec == "bottomwear":  suggest = ["topwear", "footwear", "accessories"]
        elif sec == "footwear":    suggest = ["topwear", "bottomwear", "accessories"]
        elif sec == "accessories": suggest = ["topwear", "bottomwear", "footwear"]
        else:                      suggest = ["topwear","bottomwear","footwear","accessories"]
        combos.append({
            "source_index": i,
            "source_section": sec,
            "suggest_with": suggest,
            "style_note": "A balanced mix based on common wardrobe pairing heuristics."
        })
    return {"combos": combos}

@lru_cache(maxsize=256)
def _memo_gate(sig: str) -> bool:
    return True

COMBO_SCHEMA = {
    "name": "wardrobe_combos",
    "schema": {
        "type": "object",
        "properties": {
            "combos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_index": {"type": "integer"},
                        "source_section": {"type": "string", "enum": ["topwear","bottomwear","footwear","accessories","full"]},
                        "suggest_with": {"type": "array", "items": {"type": "string","enum": ["topwear","bottomwear","footwear","accessories"]}, "minItems": 1},
                        "style_note": {"type": "string"}
                    },
                    "required": ["source_index","source_section","suggest_with","style_note"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["combos"],
        "additionalProperties": False
    },
    "strict": True
}

def _openai_build_messages(items_vlm: List[Dict[str, Any]], user_prompt: str) -> List[Dict[str, Any]]:
    system = (
        "You are a fashion stylist AI. You receive cutout clothing items with transparent backgrounds. "
        "Analyze silhouettes, textures, patterns, and colors. "
        "Return ONLY JSON that adheres to the provided schema. "
        "Never hallucinate items; only suggest categories to pair with each provided item."
    )
    content = [{"type": "text", "text": "Inputs follow:"}]
    for i, it in enumerate(items_vlm, start=1):
        content.append({"type": "text", "text": f"Item #{i} section={it['section']}."})
        content.append(_inline_png_for_openai(it["rgba"]))
    if user_prompt:
        content.append({"type": "text", "text": f"User preferences: {user_prompt.strip()}."})
    content.append({"type": "text", "text": "Respond with JSON only, following the schema."})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]

def _openai_call(parts_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    if ocli is None:
        raise RuntimeError("OPENAI_API_KEY not set")
    _respect_rate_limit()
    resp = ocli.chat.completions.create(
        model=OPENAI_MODEL,
        messages=parts_messages,
        response_format={"type": "json_schema", "json_schema": COMBO_SCHEMA},
        temperature=0.1,
        max_tokens=384,
    )
    txt = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(txt)
    except Exception:
        return {"combos": []}

def _sig_for_items(items_vlm: List[Dict[str, Any]], user_prompt: str) -> str:
    h = hashlib.sha256()
    h.update((user_prompt or "").encode("utf-8"))
    for it in items_vlm:
        h.update(it["section"].encode("utf-8"))
        thumb = it["rgba"].copy(); thumb.thumbnail((96, 96))
        b = io.BytesIO(); thumb.save(b, format="PNG")
        h.update(hashlib.md5(b.getvalue()).digest())
    return h.hexdigest()

def _persist_and_preview(payload: Dict[str, Any], items_vlm: List[Dict[str, Any]], user_id: Optional[str], session_id: Optional[str], persist_table: bool, persist_storage: bool) -> Dict[str, Any]:
    try:
        preview_img = build_recommend_preview([itm["rgba"] for itm in items_vlm])
        ns = f"{(user_id or 'anon').strip()}/results/"
        payload["result_preview_url"] = upload_png_to_supabase(preview_img, key_prefix=ns)
    except Exception:
        payload["result_preview_url"] = None

    persist_info = persist_rec_utils(
        payload, user_id=user_id or "anon", session_id=session_id,
        store_in_table=persist_table, store_in_storage=persist_storage
    )
    payload.update({
        "result_cdn_url": persist_info.get("cdn_url"),
        "db_inserted": persist_info.get("db_inserted", False),
        "db_info": persist_info.get("db_row")
    })
    if "db_error" in persist_info: payload["db_error"] = persist_info["db_error"]
    if "storage_error" in persist_info: payload["storage_error"] = persist_info["storage_error"]
    return payload

@app.post("/recommend")
def recommend(
    req: RecommendReq,
    fast: bool = False,
    timeout_sec: int = 35,
    user_id: Optional[str] = Query(None, description="Namespace for persistence (e.g. 'shivani')"),
    session_id: Optional[str] = Query(None, description="Optional session identifier"),
    persist_storage: bool = Query(True, description="Store JSON in Supabase Storage for CDN URL"),
    persist_table: bool = Query(True, description="Insert a row into 'recommendations' table"),
):
    t0 = time.perf_counter()
    items_vlm: List[Dict[str, Any]] = []
    index_map: List[Dict[str, Any]] = []

    for i, it in enumerate(req.items, start=1):
        if it.png_b64:
            img = b64_to_pil(it.png_b64)
        elif it.url:
            img = fetch_to_pil(it.url)
        elif it.png_url:
            img = fetch_to_pil(it.png_url)
        else:
            continue
        items_vlm.append({"section": it.section, "rgba": img})
        index_map.append({"index": i, "id": it.id})

    if fast or not items_vlm:
        recs = _fallback_from_sections(items_vlm)
        payload = {
            "index_map": index_map,
            "recommendations": recs,
            "prompt": req.prompt or "",
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "mode": "fallback"
        }
        return _persist_and_preview(payload, items_vlm, user_id, session_id, persist_table, persist_storage)

    if ocli is None:
        return JSONResponse({"error": "OPENAI_API_KEY not set"}, status_code=400)

    messages = _openai_build_messages(items_vlm, req.prompt or "")
    sig = _sig_for_items(items_vlm, req.prompt or "")

    def _run():
        _ = _memo_gate(sig)
        try:
            return {"_mode": "openai", **_openai_call(messages)}
        except Exception as e:
            return {"_mode": "quota_fallback", **_fallback_from_sections(items_vlm), "error": f"openai_error: {e}"}

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run)
        try:
            recs = fut.result(timeout=timeout_sec)
            mode = recs.pop("_mode", "openai")
        except FuturesTimeout:
            recs = _fallback_from_sections(items_vlm)
            mode = "timeout_fallback"

    payload = {
        "index_map": index_map,
        "recommendations": recs,
        "prompt": req.prompt or "",
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "mode": mode
    }
    return _persist_and_preview(payload, items_vlm, user_id, session_id, persist_table, persist_storage)

# ── Catalog-only recommender ──────────────────────────────────────────────────
class CatalogRecommendReq(BaseModel):
    user_id: str
    target_section: Optional[str] = Field(None, pattern="^(topwear|bottomwear|footwear|accessories)$")
    target_description: Optional[str] = None
    target_catalog_id: Optional[str] = None
    topk: int = 3

@app.post("/recommend/from_catalog")
def recommend_from_catalog(req: CatalogRecommendReq):
    # Resolve the target piece
    if req.target_catalog_id:
        row_res = get_client().table("catalog").select("*").eq("id", req.target_catalog_id).limit(1).execute()
        rows = row_res.data or []
        if not rows:
            return JSONResponse({"error": "target_catalog_id not found"}, status_code=404)
        row = rows[0]
        target_piece = {"section": row.get("section"), "description": row.get("description") or ""}
    else:
        if not req.target_section:
            return JSONResponse({"error": "provide target_section or target_catalog_id"}, status_code=400)
        target_piece = {"section": req.target_section, "description": req.target_description or ""}

    # Compute catalog recommendations
    recs = sb_recommend_from_catalog(req.user_id, target_piece, topk=max(1, min(req.topk, 6)))

    # Build a small preview from first item per returned section
    preview_imgs: List[Image.Image] = []
    for sec, items in recs.items():
        if items:
            url = items[0].get("image_url")
            if url:
                try:
                    preview_imgs.append(fetch_to_pil(url))
                except Exception:
                    pass

    preview_url = None
    if preview_imgs:
        try:
            preview_img = build_recommend_preview(preview_imgs)
            ns = f"{req.user_id.strip()}/results/"
            preview_url = upload_png_to_supabase(preview_img, key_prefix=ns)
        except Exception:
            preview_url = None

    return {
        "target": target_piece,
        "recommendations": recs,
        "result_preview_url": preview_url,
        "mode": "catalog"
    }

# ── /outfits (compose multiple cutouts into one PNG) ──────────────────────────
class SaveOutfitReq(BaseModel):
    name: Optional[str] = "Outfit"
    items: List[RecommendItem]
    upload_composite: bool = True
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@app.post("/outfits")
def save_outfit(req: SaveOutfitReq):
    imgs: List[Image.Image] = []
    for it in req.items:
        if it.png_b64:
            imgs.append(b64_to_pil(it.png_b64))
        elif it.url or it.png_url:
            imgs.append(fetch_to_pil(it.url or it.png_url))

    if not imgs:
        return JSONResponse({"error": "no images"}, status_code=400)

    canvas = make_composite(imgs, canvas=(1024, 768))

    if req.upload_composite:
        ns = f"{req.user_id.strip()}/" if req.user_id else ""
        url = upload_png_to_supabase(canvas, key_prefix=f"{ns}outfits/")
        meta = {"name": req.name, "composite_url": url, "user_id": req.user_id, "session_id": req.session_id, "created_at": int(time.time())}
        try:
            meta_cdn = upload_json_to_storage(meta, user_id=req.user_id or "anon")
        except Exception:
            meta_cdn = None
        return {"name": req.name, "composite_url": url, "meta_cdn_url": meta_cdn}
    else:
        return {"name": req.name, "composite_png_b64": pil_to_b64_png(canvas)}
