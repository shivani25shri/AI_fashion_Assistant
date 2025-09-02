import os
import streamlit as st
from PIL import Image
from pipeline import load_models, segment_image_with_hf

st.set_page_config(page_title="HF BG Removal + Clothing Segmentation", layout="wide")
st.title("🧵 HF Background Removal → Clothing Segmentation")

@st.cache_resource(show_spinner=False)
def init_models():
    # -1 CPU; set 0 to use CUDA:0 if available
    return load_models(device=-1)

_, seg = init_models()

files = st.file_uploader(
    "Upload images…", type=["jpg","jpeg","png"], accept_multiple_files=True
)

if files:
    for f in files:
        st.markdown("---")
        st.subheader(f"Image: {f.name}")
        img = Image.open(f).convert("RGB")

        # run: HF BG remover → HF SegFormer → buckets
        try:
            crops = segment_image_with_hf(seg, img, hf_token=os.getenv("HF_TOKEN"))
        except Exception as e:
            st.error(f"Failed: {e}")
            continue

        # show original + buckets
        st.image(img, caption="Original", use_container_width=True)

        order  = ["topwear", "bottomwear", "footwear", "accessories"]
        labels = ["Topwear", "Bottomwear", "Footwear", "Accessories"]
        cols = st.columns(4)

        # fixed widths so tiny items (shoes/bags) aren’t upscaled to huge pixelated blobs
        size_by_section = {"topwear": 260, "bottomwear": 260, "footwear": 180, "accessories": 160}

        for col, key, label in zip(cols, order, labels):
            with col:
                st.markdown(f"**{label}**")
                if key in crops:
                    col.image(crops[key]["rgba"], width=size_by_section[key])
                else:
                    col.write("—")
else:
    st.info("Upload 1+ images to begin.")
