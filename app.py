# app.py
import os, io, json, traceback
import streamlit as st
from PIL import Image

# ---- local modules ----
from pipeline import load_models, load_caption_model, segment_and_describe
from supabase_utils import (
    get_client,
    upload_image_to_storage,
    add_catalog_items_bulk,
    recommend_from_supabase,
)

st.set_page_config(page_title="👗 Complete the Look – Catalog + Cross Recommendations",
                   page_icon="🛍️", layout="wide")

st.title("👗 Complete the Look – Catalog Indexing + Cross-Category Recommender")

# -----------------------------------------------------------------------------------
# ENV CHECK (Supabase)
# -----------------------------------------------------------------------------------
def supabase_ok() -> bool:
    try:
        _ = get_client()
        return True
    except Exception as e:
        st.error(f"Supabase not ready: {e}")
        return False

# -----------------------------------------------------------------------------------
# MODELS (cached)
# -----------------------------------------------------------------------------------
@st.cache_resource
def init_models():
    _, seg_pipeline = load_models(device=-1)          # CPU segformer
    cap_proc, cap_model = load_caption_model("cpu")   # BLIP on CPU
    return seg_pipeline, cap_proc, cap_model

seg_pipeline, cap_proc, cap_model = init_models()

# -----------------------------------------------------------------------------------
# SIDEBAR: USER PREFS
# -----------------------------------------------------------------------------------
st.sidebar.header("Your Preferences")
user_id = st.sidebar.text_input("User ID (required)", value="demo-user")
preferred_colors = st.sidebar.multiselect(
    "Preferred colors (optional)",
    ["black","white","grey","gray","beige","cream","off white",
     "blue","navy","light blue","red","green","olive","brown",
     "yellow","purple","pink","orange","khaki"],
    default=[]
)
preferred_fit = st.sidebar.multiselect(
    "Preferred fit (optional)",
    ["slim","regular","relaxed","oversized","tapered","straight"],
    default=[]
)
st.sidebar.caption("These are used as soft filters when picking from your catalog.")

if not user_id:
    st.warning("Enter a User ID in the sidebar to proceed.")

# -----------------------------------------------------------------------------------
# 1) CATALOG INDEXING
# -----------------------------------------------------------------------------------
st.header("1) Index your **catalog images** to Supabase")
st.write("Upload 1+ images that represent your personal catalog. "
         "We’ll segment garments → caption each crop with BLIP → upload each crop to Supabase Storage → insert rows to the `catalog` table.")

catalog_files = st.file_uploader(
    "Upload catalog images",
    type=["jpg","jpeg","png","webp"],
    accept_multiple_files=True
)

index_btn = st.button("📤 Index Catalog to Supabase", use_container_width=True)

def _short_attrs_from_caption(desc: str):
    """
    Tiny heuristic to extract attrs from a BLIP caption.
    Returns {'color': '...', 'type': '...'} best-effort.
    """
    d = (desc or "").lower()
    # color guess
    colors = ["black","white","grey","gray","beige","cream","off white",
              "blue","navy","light blue","red","green","olive","brown",
              "yellow","purple","pink","orange","khaki"]
    color = next((c for c in colors if c in d), None)

    # type guess
    keywords = [
        ("topwear", ["shirt","t-shirt","tee","blouse","top","jacket","coat","hoodie","sweater","cardigan","blazer","dress"]),
        ("bottomwear", ["jeans","trousers","pants","shorts","skirt","leggings","cargo"]),
        ("footwear", ["sneaker","shoe","boot","heel","loafer","sandal"]),
        ("accessories", ["belt","bag","handbag","backpack","hat","cap","scarf","glasses","sunglasses","watch","wallet","necklace","earring"]),
    ]
    typ = None
    for _, words in keywords:
        for w in words:
            if w in d:
                typ = w
                break
        if typ: break
    return {"color": color, "type": typ}

def _pick_section(section_key: str, fallback: str, desc: str):
    """Prefer the detected section; otherwise guess from description."""
    if section_key in ("topwear","bottomwear","footwear","accessories"):
        return section_key
    # Guess from desc
    d = desc.lower()
    if any(w in d for w in ["shirt","t-shirt","tee","blouse","top","jacket","coat","hoodie","sweater","cardigan","blazer","dress"]):
        return "topwear"
    if any(w in d for w in ["jeans","trousers","pants","shorts","skirt","leggings","cargo"]):
        return "bottomwear"
    if any(w in d for w in ["sneaker","shoe","boot","heel","loafer","sandal"]):
        return "footwear"
    if any(w in d for w in ["belt","bag","handbag","backpack","hat","cap","scarf","glasses","sunglasses","watch","wallet","necklace","earring"]):
        return "accessories"
    return fallback

if index_btn:
    if not user_id:
        st.error("Please set a User ID in the sidebar first.")
    elif not supabase_ok():
        st.stop()
    elif not catalog_files:
        st.warning("Upload at least one catalog image.")
    else:
        total_crops = 0
        all_rows = []
        with st.spinner("Indexing catalog…"):
            try:
                for idx, f in enumerate(catalog_files, 1):
                    img = Image.open(f).convert("RGB")
                    st.write(f"Processing catalog image {idx}: **{f.name}**")
                    results = segment_and_describe(
                        seg_pipeline, cap_proc, cap_model,
                        img, hf_token=None, device="cpu",
                        run_recommender=False
                    )

                    # For each section crop: upload → add to rows
                    for sec, data in results.items():
                        if "rgba" not in data:
                            continue
                        desc = data.get("description", "garment")
                        # Upload to storage
                        try:
                            url = upload_image_to_storage(sec, data["rgba"], user_id=user_id)
                        except Exception as e:
                            st.error(f"Storage upload failed for {sec} in {f.name}: {e}")
                            continue

                        attrs = _short_attrs_from_caption(desc)
                        section = _pick_section(sec, fallback="topwear", desc=desc)
                        row = {
                            "section": section,
                            "description": desc,
                            "image_url": url,
                            "attrs": attrs,
                            "extra": {"source_file": f.name},
                        }
                        all_rows.append(row)
                        total_crops += 1

                # Bulk insert rows
                if all_rows:
                    _ = add_catalog_items_bulk(user_id, all_rows)
                    st.success(f"Indexed {len(all_rows)} catalog items for user `{user_id}` ✅")
                else:
                    st.warning("No garments were detected to index.")
            except Exception as e:
                st.error("Catalog indexing failed.")
                st.code("".join(traceback.format_exc()))
        if total_crops:
            st.toast(f"Uploaded {total_crops} segmented pieces to Supabase", icon="✅")

# -----------------------------------------------------------------------------------
# 2) QUERY IMAGE → CROSS-CATEGORY RECOMMENDATIONS
# -----------------------------------------------------------------------------------
st.header("2) Upload ONE input → get cross-category outfits")
st.write("We’ll segment your input and make cross-category matches against your **stored catalog**.")

query_file = st.file_uploader("Upload an input image", type=["jpg","jpeg","png","webp"], accept_multiple_files=False)
run_recs = st.button("✨ Recommend Outfits", use_container_width=True)

def _bias_by_prefs(items, colors, fits):
    """Soft-bias a list of catalog rows by user-preferred colors/fits."""
    if not colors and not fits:
        return items
    def score(row):
        s = 0
        d = (row.get("description") or "").lower()
        # color bias
        for c in colors:
            if c.lower() in d:
                s += 1
        # fit bias (looks in text)
        for f in fits:
            if f.lower() in d:
                s += 0.5
        return s
    return sorted(items, key=score, reverse=True)

def _choose_target_piece(results: dict):
    """
    Choose the primary piece from the query image:
    prefer topwear or bottomwear (then footwear, accessories).
    """
    for sec in ("topwear","bottomwear","footwear","accessories"):
        if sec in results and "description" in results[sec]:
            return {"section": sec, "description": results[sec]["description"], "rgba": results[sec].get("rgba")}
    return None

def _build_three_combos(target, complements):
    """
    Build up to 3 combos: (top+bottom), (+shoes), (+accessory)
    or if target is bottomwear, reverse (top first).
    """
    combos = []
    tsec = target["section"]

    # complements is dict: {"topwear":[...], "bottomwear":[...], "footwear":[...], "accessories":[...]}
    # Ensure order for top/bottom depending on target
    if tsec == "topwear":
        bot = complements.get("bottomwear", [])
        shoe = complements.get("footwear", [])
        acc = complements.get("accessories", [])
        if bot:
            combos.append({
                "combo": ["topwear","bottomwear"],
                "desc": f"{target['description']} + {bot[0]['description']}",
                "refs": {"bottomwear": bot[0]}
            })
            if shoe:
                combos.append({
                    "combo": ["topwear","bottomwear","footwear"],
                    "desc": f"{target['description']} + {bot[0]['description']} + {shoe[0]['description']}",
                    "refs": {"bottomwear": bot[0], "footwear": shoe[0]}
                })
            if acc:
                combos.append({
                    "combo": ["topwear","bottomwear","accessories"],
                    "desc": f"{target['description']} + {bot[0]['description']} + {acc[0]['description']}",
                    "refs": {"bottomwear": bot[0], "accessories": acc[0]}
                })
    elif tsec == "bottomwear":
        top = complements.get("topwear", [])
        shoe = complements.get("footwear", [])
        acc = complements.get("accessories", [])
        if top:
            combos.append({
                "combo": ["topwear","bottomwear"],
                "desc": f"{top[0]['description']} + {target['description']}",
                "refs": {"topwear": top[0]}
            })
            if shoe:
                combos.append({
                    "combo": ["topwear","bottomwear","footwear"],
                    "desc": f"{top[0]['description']} + {target['description']} + {shoe[0]['description']}",
                    "refs": {"topwear": top[0], "footwear": shoe[0]}
                })
            if acc:
                combos.append({
                    "combo": ["topwear","bottomwear","accessories"],
                    "desc": f"{top[0]['description']} + {target['description']} + {acc[0]['description']}",
                    "refs": {"topwear": top[0], "accessories": acc[0]}
                })
    else:
        # If target is footwear or accessories, try to pair with top+bottom if available
        top = complements.get("topwear", [])
        bot = complements.get("bottomwear", [])
        if top and bot:
            combos.append({
                "combo": ["topwear","bottomwear", tsec],
                "desc": f"{top[0]['description']} + {bot[0]['description']} + {target['description']}",
                "refs": {"topwear": top[0], "bottomwear": bot[0]}
            })

    return combos[:3]

if run_recs:
    if not user_id:
        st.error("Please set a User ID in the sidebar first.")
        st.stop()
    if not supabase_ok():
        st.stop()
    if not query_file:
        st.warning("Upload one input image first.")
        st.stop()

    img = Image.open(query_file).convert("RGB")
    st.subheader("📸 Your Input")
    st.image(img, width=420)

    with st.spinner("Analyzing input…"):
        try:
            qres = segment_and_describe(
                seg_pipeline, cap_proc, cap_model,
                img, hf_token=None, device="cpu",
                run_recommender=False
            )
        except Exception as e:
            st.error(f"Segmentation failed: {e}")
            st.stop()

    # show detected pieces
    st.subheader("Detected in Input")
    for sec, data in qres.items():
        if "rgba" in data:
            st.image(data["rgba"], caption=f"{sec} – {data.get('description','N/A')}", width=180)

    target = _choose_target_piece(qres)
    if not target:
        st.warning("No primary garment detected to base recommendations on.")
        st.stop()

    # Query Supabase for complementary pieces
    with st.spinner("Fetching matches from your catalog…"):
        comp_raw = recommend_from_supabase(
            user_id=user_id,
            target_piece={"section": target["section"], "description": target["description"]},
            topk=10,  # fetch a few to bias by prefs
        )

    # Soft-bias by prefs
    comp_biased = {}
    for sec, rows in comp_raw.items():
        comp_biased[sec] = _bias_by_prefs(rows, preferred_colors, preferred_fit)

    combos = _build_three_combos(target, comp_biased)

    st.subheader("✨ Cross-Category Recommendations")
    if not combos:
        st.info("Not enough items in your catalog to build outfits. Try indexing more pieces.")
    else:
        for i, c in enumerate(combos, 1):
            st.markdown(f"**Combo {i}:** {c['desc']}")
            # build visual strip
            cols = st.columns(len(c["combo"]))
            # first: show the target piece for top/bottom appropriately
            for j, sec in enumerate(c["combo"]):
                with cols[j]:
                    if sec == target["section"]:
                        # show the crop from query
                        if target.get("rgba"):
                            st.image(target["rgba"], caption=f"query {sec}", width=160)
                        else:
                            st.caption(f"query {sec}")
                    else:
                        ref = c["refs"].get(sec) if "refs" in c else None
                        if ref and "image_url" in ref:
                            st.image(ref["image_url"], caption=f"{sec}", width=160)
                        else:
                            st.caption(sec)

    # Export JSON summary
    export = {
        "user_id": user_id,
        "target_piece": {"section": target["section"], "description": target["description"]},
        "combos": combos,
    }
    st.subheader("📦 JSON")
    st.code(json.dumps(export, indent=2), language="json")
    st.download_button(
        "Download JSON",
        data=io.BytesIO(json.dumps(export, indent=2).encode("utf-8")),
        file_name="recommendations.json",
        mime="application/json"
    )
c