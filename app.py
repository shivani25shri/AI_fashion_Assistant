import streamlit as st
from PIL import Image
from pipeline import load_models, segment_clothes, segment_flatlay 
from google import genai
from google.genai import types
import re
import json

# Gemini SDK setup
client = genai.Client(api_key="AIzaSyDaGNVY8FBCDhzq0qWq4kt1QCZecS40boA")

st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.title("👗 Fashion Segmenter & Recommender")

@st.cache_resource
def init_models():
    return load_models(device=0)

detector, seg = init_models()

#  Mode selector
mode = st.radio("Choose Input Mode:", ["Person Images", "Flat-lay Outfit"], horizontal=True)

#  Upload 
files = st.file_uploader(
    "Upload up to 5 images…", type=["jpg","jpeg","png"], accept_multiple_files=True
)[:5]

if files:
    st.write(f"Processing {len(files)} file{'s' if len(files)>1 else ''}")
    if st.button("Segment & Recommend"):
        all_items = []
        for f in files:
            st.markdown("---")
            st.subheader(f"Image: {f.name}")
            img = Image.open(f).convert("RGB")
            img.thumbnail((600,600), Image.LANCZOS)
            st.image(img, width=img.width)

            # segmentation depending on mode 
            if mode == "Person Images":
                crops = segment_clothes(detector, seg, img)
            else:
                crops = segment_flatlay(seg, img)

            if not crops:
                st.warning("No garments found.")
                continue

            cols = st.columns(len(crops))
            for (sec, data), col in zip(crops.items(), cols):
                with col:
                    st.subheader(sec.title())
                    st.image(data["rgba"], width=140)
                    all_items.append({"section": sec, "rgba": data["rgba"]})

        if not all_items:
            st.warning("No segmented items to recommend outfits from.")
            st.stop()

        #Build prompt 
        descs = "\n".join(f"{i+1}. {item['section']}" for i, item in enumerate(all_items))
        system = "You are a fashion stylist assistant."
        user = (
            f"I have these {len(all_items)} garment items:\n"
            f"{descs}\n\n"
            f"Only use indices between 1 and {len(all_items)}. "
            "Please suggest 3 cohesive outfit combinations. "
            "Each combo should include at least one topwear and one bottomwear, "
            "and optionally footwear or accessories. "
            "Return valid JSON: "
            "[{\"combo\": [indices], \"description\": \"…\"}, …]."
        )

        #Call Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.7
            )
        )
        raw = response.text

        # strip code fences
        m = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", raw, re.S)
        json_str = m.group(1) if m else raw

        try:
            recommendations = json.loads(json_str)
        except json.JSONDecodeError:
            st.error("Failed to parse recommendations JSON")
            st.text(raw)
            st.stop()

        # Display
        st.markdown("### Outfit Recommendations")
        for idx, rec in enumerate(recommendations, start=1):
            combo = rec.get("combo", [])
            desc  = rec.get("description", "")
            st.subheader(f"Option {idx}")
            st.write(desc)

            valid = [i for i in combo if 1 <= i <= len(all_items)]
            cols = st.columns(len(valid))
            for col, item_idx in zip(cols, valid):
                item = all_items[item_idx-1]
                with col:
                    st.image(item["rgba"], caption=item["section"].title(), width=180)
