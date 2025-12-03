# app_streamlit.py
from __future__ import annotations
import os, io, json
from typing import Dict, Any
import streamlit as st
from PIL import Image

from pipeline import (
    load_models,
    load_caption_model,
    segment_and_describe,
)
from supabase_utils import (
    add_catalog_item,
    add_catalog_items_bulk,
    recommend_from_supabase,
)

st.set_page_config(page_title="Complete the Look – Catalog + Cross-fit", page_icon="👗", layout="wide")
st.title("👗 Complete the Look – Segment • Caption • Recommend (Supabase Catalog)")

# --------- Config / Inputs ----------
with st.sidebar:
    st.subheader("🔐 Configuration")
    user_id = st.text_input("User ID", value=os.getenv("DEMO_USER_ID", "demo-user-001"))
    st.caption("All catalog operations are scoped to this user_id.")
    save_to_catalog = st.checkbox("Save detected pieces into catalog", value=False)
    run_gemini = st.checkbox("Use Gemini for 3-outfit text combos (optional)", value=False)
    st.markdown("---")
    st.caption("Tip: Ensure SUPABASE_URL and key are set in env for this app.")

# --------- Load models once ----------
@st.cache_resource
def init_models():
    _, seg_pipeline = load_models(device=-1)   # CPU
    cap_proc, cap_model = load_caption_model(device="cpu")
    return seg_pipeline, cap_proc, cap_model

seg_pipeline, cap_proc, cap_model = init_models()

# --------- File upload (multi) ----------
uploaded_files = st.file_uploader(
    "Upload one or more outfit images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

# Utility: small descriptor string for UI
def short_desc(data: Dict[str, Any]) -> str:
    if not data:
        return "item"
    # If attrs existed you'd use them; here we rely on description
    return data.get("description") or "item"

# --------- Process images ----------
all_results: Dict[str, Dict[str, Any]] = {}
if uploaded_files:
    for idx, f in enumerate(uploaded_files, 1):
        img_id = f.name or f"image_{idx}"
        st.header(f"📸 Image {idx}: {img_id}")

        img = Image.open(f).convert("RGB")
        st.image(img, caption=f"Uploaded {img_id}", width=420)

        with st.spinner("Segmenting and describing…"):
            res = segment_and_describe(
                seg_pipeline,
                cap_proc,
                cap_model,
                img,
                hf_token=None,
                device="cpu",
                run_recommender=run_gemini  # if True, pipeline will attach 'outfit_combos'
            )

        # Show crops
        st.subheader("👕 Detected Pieces")
        row = st.columns(4)
        shown = 0
        for sec in ("topwear","bottomwear","footwear","accessories"):
            if sec in res and "rgba" in res[sec]:
                with row[shown % 4]:
                    st.image(res[sec]["rgba"], caption=f"{sec} – {short_desc(res[sec])}", width=180)
                shown += 1
        if shown == 0:
            st.info("No wearable pieces detected in this image.")

        # Optional: save to catalog (each detected piece as a catalog item)
        if save_to_catalog:
            items_to_save = []
            for sec in ("topwear","bottomwear","footwear","accessories"):
                if sec in res and "rgba" in res[sec]:
                    # Use description as name if nothing else
                    desc = res[sec].get("description") or f"{sec}"
                    items_to_save.append({
                        "name": desc[:60],
                        "category": sec,
                        "description": desc,
                        "color": None,
                        "fit": None,
                        "image_url": None,  # If you have a CDN uploader, put the URL here
                    })
            if items_to_save:
                with st.spinner("Saving to catalog…"):
                    out = add_catalog_items_bulk(user_id, items_to_save)
                st.success(f"Saved {len(items_to_save)} item(s) to catalog for user {user_id}")

        # Keep a JSON-safe copy (drop PIL images)
        safe = {}
        for k,v in res.items():
            if isinstance(v, dict):
                safe[k] = {kk: vv for kk, vv in v.items() if kk != "rgba"}
            else:
                safe[k] = v
        all_results[img_id] = safe

        # Optional: Show text-only Gemini 3-outfit combos for this single image
        if run_gemini and "outfit_combos" in res:
            st.subheader("✨ Gemini Combos (Text-only)")
            for i, combo in enumerate(res["outfit_combos"], 1):
                st.markdown(f"**Combo {i}:** `{combo['combo']}` — {combo['description']}")

    # --------- Cross-fit recommendation using your catalog ----------
    st.header("🧩 Cross-fit from Catalog (based on one detected piece)")
    # We’ll pick the first image that has a topwear/bottomwear/etc as the “anchor”
    anchor = None
    for img_id, result in all_results.items():
        for sec in ("topwear","bottomwear","footwear","accessories"):
            if sec in result:
                anchor = {"image": img_id, "section": sec, "description": result[sec].get("description","")}
                break
        if anchor:
            break

    if not anchor:
        st.info("No detected pieces to anchor cross-fit recommendations.")
    else:
        st.write(f"Using **{anchor['section']}** from **{anchor['image']}** as the anchor.")
        with st.spinner("Searching your Supabase catalog…"):
            matches = recommend_from_supabase(
                user_id=user_id,
                target_piece={"section": anchor["section"], "description": anchor["description"]},
                topk=3
            )

        # Show up to 3 concise outfits composed from catalog matches
        st.subheader("✨ 3 Cross-fit Outfits (from your catalog)")
        outfits = []

        # 1) Anchor + best bottomwear
        outfit1 = [("anchor", anchor["section"], anchor["description"])]
        bw = matches.get("bottomwear", [])
        if anchor["section"] != "bottomwear" and bw:
            outfit1.append(("catalog", bw[0]["category"], bw[0]["description"], bw[0].get("image_url")))
        outfits.append(outfit1)

        # 2) Anchor + best topwear/bottomwear pair
        outfit2 = [("anchor", anchor["section"], anchor["description"])]
        if anchor["section"] != "topwear" and matches.get("topwear"):
            outfit2.append(("catalog", "topwear", matches["topwear"][0]["description"], matches["topwear"][0].get("image_url")))
        if anchor["section"] != "bottomwear" and matches.get("bottomwear"):
            outfit2.append(("catalog", "bottomwear", matches["bottomwear"][0]["description"], matches["bottomwear"][0].get("image_url")))
        outfits.append(outfit2)

        # 3) Anchor + best footwear + accessory
        outfit3 = [("anchor", anchor["section"], anchor["description"])]
        if matches.get("footwear"):
            f0 = matches["footwear"][0]
            outfit3.append(("catalog", "footwear", f0["description"], f0.get("image_url")))
        if matches.get("accessories"):
            a0 = matches["accessories"][0]
            outfit3.append(("catalog", "accessories", a0["description"], a0.get("image_url")))
        outfits.append(outfit3)

        # Render 3 outfits
        for i, outfit in enumerate(outfits, 1):
            st.markdown(f"**Outfit {i}**")
            cols = st.columns(len(outfit))
            for j, piece in enumerate(outfit):
                kind = piece[0]
                if kind == "anchor":
                    _, sec, desc = piece
                    with cols[j]:
                        # show the original uploaded image for anchor (small)
                        # find the image file again
                        up = next((u for u in uploaded_files if (u.name == anchor["image"])), None)
                        if up:
                            st.image(up, caption=f"{sec} (anchor)", width=150)
                        st.caption(desc)
                else:
                    _, sec, desc, url = piece
                    with cols[j]:
                        if url:
                            st.image(url, caption=f"{sec} (catalog)", width=150)
                        else:
                            st.write(f"*{sec}* (no image)")
                        st.caption(desc)
            st.write("---")

    # --------- Export everything we computed ----------
    st.subheader("📦 Export (JSON)")
    export_payload = {"results": all_results}
    json_str = json.dumps(export_payload, indent=2)
    st.code(json_str, language="json")
    st.download_button(
        "Download JSON",
        io.BytesIO(json_str.encode("utf-8")),
        file_name="fashion_results.json",
        mime="application/json",
        type="primary"
    )
else:
    st.info("Upload one or more images to begin.")
