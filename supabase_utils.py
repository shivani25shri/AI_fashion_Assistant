# supabase_utils.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os, io, uuid, json, time
from PIL import Image

# ── Lazy import so unit tests don't fail without creds ─────────────────────────
try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except Exception:
    HAS_SUPABASE = False
    Client = Any  # type: ignore

# ── Config ────────────────────────────────────────────────────────────────────
_SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
# Prefer SERVICE_ROLE/KEY for backend writes; ANON acceptable for read-only/dev.
_SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE")
    or os.getenv("SUPABASE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or ""
).strip()

# Match your server’s bucket default
_SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "wadrobe_items").strip()

# Folder (prefix) inside the bucket where we store JSON results
_SUPABASE_RESULTS_PREFIX = os.getenv("SUPABASE_RESULTS_PREFIX", "results/").strip("/")

_sb_client: Optional[Client] = None


def _require_config():
    if not _SUPABASE_URL or not _SUPABASE_KEY:
        raise RuntimeError("Supabase not configured. Set SUPABASE_URL and SUPABASE_(SERVICE_ROLE or KEY or ANON_KEY).")


def get_client() -> Client:
    """
    Returns a cached supabase client or raises RuntimeError if not configured.
    """
    global _sb_client
    _require_config()
    if not HAS_SUPABASE:
        raise RuntimeError("supabase-py not installed. `pip install supabase`")
    if _sb_client is None:
        _sb_client = create_client(_SUPABASE_URL, _SUPABASE_KEY)
    return _sb_client


# ── Storage helpers ───────────────────────────────────────────────────────────
def _normalize_public_url(url_obj: Any) -> str:
    # Some SDK versions return dict {"publicUrl": "..."}
    return url_obj if isinstance(url_obj, str) else (url_obj or {}).get("publicUrl", "")

def upload_image_to_storage(
    section: str,
    img: Image.Image,
    user_id: str = "anon",
    bucket: Optional[str] = None,
    content_type: str = "image/png",
    upsert: bool = True,
    cache_control: str = "31536000, immutable",
) -> str:
    """
    Uploads a PIL image to Supabase Storage and returns a public URL.
    Path: {user_id}/{section}/{uuid}.png
    """
    sb = get_client()
    bucket = bucket or _SUPABASE_BUCKET
    path = f"{user_id}/{section}/{uuid.uuid4().hex}.png"
    buf = io.BytesIO()
    img.convert("RGBA").save(buf, format="PNG", optimize=True)
    buf.seek(0)
    sb.storage.from_(bucket).upload(
        path=path,
        file=buf.getvalue(),
        file_options={
            "content-type": content_type,
            "upsert": "true" if upsert else "false",   # ← MUST be string
            "cache-control": str(cache_control),       # ← ensure string
        },
    )
    return _normalize_public_url(sb.storage.from_(bucket).get_public_url(path))

def upload_json_to_storage(
    data: Dict[str, Any],
    user_id: str = "anon",
    key_prefix: Optional[str] = None,
    bucket: Optional[str] = None,
    filename: Optional[str] = None,
    upsert: bool = True,
    cache_control: str = "31536000, immutable",
) -> str:
    """
    Upload a JSON-serializable dict to Supabase Storage and return a public URL.
    Tries application/json first; if the bucket disallows it, falls back to text/plain.
    Path: {user_id}/{key_prefix or results/}/{uuid}.json
    """
    sb = get_client()
    bucket = bucket or _SUPABASE_BUCKET
    key_prefix = (key_prefix or _SUPABASE_RESULTS_PREFIX).strip("/")
    fname = filename or f"{uuid.uuid4().hex}.json"
    path = f"{user_id}/{key_prefix}/{fname}"

    body = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def _upload(ct: str):
        return sb.storage.from_(bucket).upload(
            path=path,
            file=body,
            file_options={
                "content-type": ct,
                "upsert": "true" if upsert else "false",  # ← MUST be string
                "cache-control": str(cache_control),      # ← ensure string
            },
        )

    try:
        _upload("application/json")
    except Exception:
        # bucket likely blocks JSON; retry as text/plain
        _upload("text/plain")

    return _normalize_public_url(sb.storage.from_(bucket).get_public_url(path))


# ── Recommendation persistence (Storage JSON + optional DB) ───────────────────
def persist_recommendation(
    result_payload: Dict[str, Any],
    user_id: str = "anon",
    session_id: Optional[str] = None,
    store_in_table: bool = True,
    store_in_storage: bool = True,
    bucket: Optional[str] = None,
    key_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stores the recommendation payload to Storage (returns public CDN URL) and
    optionally inserts a row into `public.recommendations`.

    Returns:
        {
          "cdn_url": Optional[str],
          "db_inserted": bool,
          "db_row": Optional[dict],
          "storage_error": Optional[str],
          "db_error": Optional[str],
        }
    """
    out: Dict[str, Any] = {"cdn_url": None, "db_inserted": False, "db_row": None}

    # 1) Storage
    if store_in_storage:
        try:
            out["cdn_url"] = upload_json_to_storage(
                data=result_payload,
                user_id=user_id,
                key_prefix=key_prefix or _SUPABASE_RESULTS_PREFIX,
                bucket=bucket or _SUPABASE_BUCKET,
            )
        except Exception as e:
            out["storage_error"] = str(e)

    # 2) Table
    if store_in_table:
        try:
            sb = get_client()
            row = {
                "user_id": user_id,
                "session_id": session_id,
                "result_json": result_payload,
                "result_cdn_url": out.get("cdn_url"),
            }
            resp = sb.table("recommendations").insert(row).execute()
            out["db_inserted"] = True
            out["db_row"] = getattr(resp, "data", None)
        except Exception as e:
            out["db_error"] = str(e)

    return out

def list_recommendations(
    user_id: str,
    limit: int = 20,
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch latest rows from public.recommendations for a user (optionally by session_id).
    """
    sb = get_client()
    q = (
        sb.table("recommendations")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(max(1, min(limit, 200)))
    )
    if session_id:
        q = q.eq("session_id", session_id)
    res = q.execute()
    return res.data or []  # type: ignore


# ── Catalog helpers (DB) ──────────────────────────────────────────────────────
# Expected table: catalog
# Columns (suggested):
#   id uuid pk, user_id text, section text, description text,
#   image_url text, attrs jsonb, extra jsonb, created_at timestamptz default now()

def add_catalog_item(
    user_id: str,
    section: str,
    description: str,
    image_url: str,
    attrs: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Insert one catalog row.
    """
    sb = get_client()
    payload = {
        "user_id": user_id,
        "section": section,
        "description": description,
        "image_url": image_url,
        "attrs": attrs or {},
        "extra": extra or {},
    }
    res = sb.table("catalog").insert(payload).execute()
    return (res.data or [{}])[0]  # type: ignore

def add_catalog_items_bulk(
    user_id: str,
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Bulk insert items: each must include at least section, description, image_url.
    Will inject user_id if missing.
    """
    sb = get_client()
    rows = []
    for it in items:
        rows.append({
            "user_id": user_id,
            "section": it.get("section"),
            "description": it.get("description"),
            "image_url": it.get("image_url"),
            "attrs": it.get("attrs", {}),
            "extra": it.get("extra", {}),
        })
    res = sb.table("catalog").insert(rows).execute()
    return res.data or []  # type: ignore

def upsert_catalog_items(
    user_id: str,
    items: List[Dict[str, Any]],
    on_conflict: str = "image_url",  # change to composite index if you have one
) -> List[Dict[str, Any]]:
    """
    Upsert by `on_conflict` column (default image_url). Ensure a unique index.
    """
    sb = get_client()
    rows = []
    for it in items:
        rows.append({
            "user_id": user_id,
            "section": it.get("section"),
            "description": it.get("description"),
            "image_url": it.get("image_url"),
            "attrs": it.get("attrs", {}),
            "extra": it.get("extra", {}),
        })
    res = sb.table("catalog").upsert(rows, on_conflict=on_conflict).execute()
    return res.data or []  # type: ignore

def fetch_catalog(
    user_id: str,
    section: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Fetch catalog items for a user; optionally filter by section.
    """
    sb = get_client()
    q = sb.table("catalog").select("*").eq("user_id", user_id)
    if section:
        q = q.eq("section", section)
    q = q.limit(max(1, min(limit, 1000)))
    res = q.execute()
    return res.data or []  # type: ignore

def search_catalog_by_text(
    user_id: str,
    query: str,
    section: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Case-insensitive LIKE search on description. For better results, add pgvector later.
    """
    sb = get_client()
    q = sb.table("catalog").select("*").eq("user_id", user_id).ilike("description", f"%{query}%")
    if section:
        q = q.eq("section", section)
    q = q.limit(max(1, min(limit, 200)))
    res = q.execute()
    return res.data or []  # type: ignore


# ── Simple text-based cross-category recommender ──────────────────────────────
_COLOR_WORDS = [
    "black","white","grey","gray","beige","blue","navy","light blue","red",
    "green","olive","brown","yellow","purple","pink","orange","khaki","cream","off white"
]
_FIT_WORDS = ["slim","regular","relaxed","oversized","tapered","straight"]

def _keyword_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    import re as _re
    a_tokens = set(_re.findall(r"[a-z]+", a.lower()))
    b_tokens = set(_re.findall(r"[a-z]+", b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))

def recommend_from_supabase(
    user_id: str,
    target_piece: Dict[str, Any],
    topk: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Given a target piece (e.g., {"section":"topwear","description":"blue denim jacket"}),
    fetch up to `topk` complementary items per other section from the user's catalog.
    Very light heuristic: keyword overlap + small bonuses for same color/fit words.
    """
    sb = get_client()
    _ = sb  # assert configured

    qdesc = (target_piece.get("description") or "").lower()
    section = (target_piece.get("section") or "").lower()
    all_sections = ["topwear","bottomwear","footwear","accessories"]
    search_sections = [s for s in all_sections if s != section]

    out: Dict[str, List[Dict[str, Any]]] = {}
    for cat in search_sections:
        res = get_client().table("catalog").select("*").eq("user_id", user_id).eq("section", cat).limit(100).execute()
        items = res.data or []
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for it in items:
            d = (it.get("description") or "").lower()
            overlap = _keyword_overlap(qdesc, d)

            color_bonus = 0.0
            for c in _COLOR_WORDS:
                if c in qdesc and c in d:
                    color_bonus = 0.2
                    break

            fit_bonus = 0.0
            for f in _FIT_WORDS:
                if f in qdesc and f in d:
                    fit_bonus = 0.1
                    break

            scored.append((overlap + color_bonus + fit_bonus, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        out[cat] = [it for _, it in scored[:topk]]

    return out


__all__ = [
    "get_client",
    "upload_image_to_storage",
    "upload_json_to_storage",
    "persist_recommendation",
    "list_recommendations",
    "add_catalog_item",
    "add_catalog_items_bulk",
    "upsert_catalog_items",
    "fetch_catalog",
    "search_catalog_by_text",
    "recommend_from_supabase",
]
