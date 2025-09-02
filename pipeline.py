    # pipeline.py

import os
os.environ["CURL_CA_BUNDLE"] = ""
import requests
from huggingface_hub import configure_http_backend
from ultralytics import YOLO
from transformers import pipeline as hf_pipeline
from PIL import Image
import numpy as np


def _no_verify_session():
    s = requests.Session(); s.verify = False; return s
configure_http_backend(backend_factory=_no_verify_session)


# -------------------- Model loader --------------------
def load_models(device: int = 0):
    detector = YOLO("yolov8n.pt")   # YOLO only used for person mode
    seg = hf_pipeline(
        "image-segmentation",
        model="mattmdjaga/segformer_b2_clothes",
        device=device
    )
    return detector, seg


# -------------------- Label mapping --------------------
def map_label_to_section(label: str) -> str | None:
    l = label.lower()
    if l in {"upper-clothes","t-shirt","shirt","coat","jacket","outerwear","blazer","dress","top"}:
        return "topwear"
    if l in {"pants","trousers","jeans","skirt","shorts"}:
        return "bottomwear"
    if any(k in l for k in ("shoe","sneaker","boot","heel")):
        return "footwear"
    if any(k in l for k in ("bag","handbag","backpack","hat","belt","scarf","glasses","sunglasses","watch","wallet")):
        return "accessories"
    return None


# -------------------- Internal helpers --------------------
def _collect_masks(seg_outputs):
    groups = {}
    for out in seg_outputs:
        sec = map_label_to_section(out["label"])
        if not sec: continue
        arr = np.array(out["mask"]) if isinstance(out["mask"], Image.Image) else out["mask"]
        if arr.dtype in (np.float32,np.float64):
            arr = (arr*255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        bin_arr = (arr>127).astype(np.uint8)*255
        groups.setdefault(sec, []).append(bin_arr)
    return groups

def _union_and_crop(base_img: Image.Image, groups: dict[str,list[np.ndarray]], pad:int=6):
    results = {}
    for sec, masks in groups.items():
        if not masks: continue
        union = np.maximum.reduce(masks)
        mask_full = Image.fromarray(union,"L").resize(base_img.size, Image.NEAREST)
        rgba_full = Image.new("RGBA", base_img.size)
        rgba_full.paste(base_img, mask=mask_full)

        alpha = np.array(rgba_full.split()[-1])
        ys,xs = np.where(alpha>0)
        if ys.size and xs.size:
            y0,y1=ys.min(),ys.max()
            x0,x1=xs.min(),xs.max()
        else:
            y0,y1=0, base_img.height
            x0,x1=0, base_img.width

        x0,y0=max(x0-pad,0), max(y0-pad,0)
        x1,y1=min(x1+pad, base_img.width), min(y1+pad, base_img.height)
        bbox=[int(x0),int(y0),int(x1),int(y1)]
        rgba_crop = rgba_full.crop((x0,y0,x1,y1))
        results[sec]={"rgba": rgba_crop,"bbox": bbox}
    return results


# -------------------- Public API --------------------
def segment_clothes(detector, seg, img: Image.Image):
    """
    Person mode: detect person box with YOLO, crop, then segment.
    """
    res = detector(img, imgsz=640)[0]
    person_box = next((tuple(b.xyxy.cpu().numpy().astype(int)[0]) 
                       for b in res.boxes if int(b.cls)==0), None)
    if person_box:
        x1,y1,x2,y2 = person_box
        crop = img.crop((x1,y1,x2,y2))
    else:
        crop = img.copy()
    outputs = seg(crop)
    groups = _collect_masks(outputs)
    return _union_and_crop(crop, groups, pad=6)

def segment_flatlay(seg, img: Image.Image):
    """
    Flat-lay mode: run segmentation on the full image (no YOLO).
    """
    outputs = seg(img)
    groups = _collect_masks(outputs)
    return _union_and_crop(img, groups, pad=6)
