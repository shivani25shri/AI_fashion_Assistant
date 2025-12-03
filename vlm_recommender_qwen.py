# vlm_recommender_qwen.py
from __future__ import annotations
import json, re, torch
from typing import List, Dict, Any, Optional
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

class LocalVLM_Qwen:
    def __init__(self, model_id: str = MODEL_ID, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load processor + model
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device != "cpu" else None,
            trust_remote_code=True
        ).to(self.device)

    def recommend(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        items: [{"section": "topwear|bottomwear|footwear|accessories", "rgba": PIL.Image}, ...]
        returns: [{"combo":[indices], "description": "..."}]
        """
        if not items:
            return []

        # Build prompt
        listing = "\n".join(f"{i+1}. {it['section']}" for i, it in enumerate(items))
        prompt = (
            "You are a fashion stylist. Respond ONLY with JSON.\n"
            "Constraints:\n"
            "1) Create exactly 3 cohesive outfit combinations using the provided numbered items.\n"
            "2) Each combo MUST include at least one topwear and one bottomwear.\n"
            "3) You MAY add footwear and/or accessories.\n"
            f"4) Only use indices 1..{len(items)}.\n\n"
            "JSON format: "
            '[{"combo":[<indices>],"description":"<1-2 sentence rationale>"}]\n\n'
            "Items:\n" + listing
        )

        images = [it["rgba"].convert("RGB") for it in items]
        messages = [
            {"role": "system", "content": "You are a precise fashion stylist."},
            {"role": "user", "content": [{"type": "text", "text": prompt}] + [{"type": "image"} for _ in images]},
        ]
        chat_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(text=[chat_text], images=images, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=256)
        text = self.processor.batch_decode(gen, skip_special_tokens=True)[0]

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                data = [data]
            return data
        except Exception:
            return [{"combo": [], "description": "Parsing failed"}]
