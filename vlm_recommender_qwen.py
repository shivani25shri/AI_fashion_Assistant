# vlm_recommender_qwen.py
# Local VLM outfit recommender using Qwen2.5-VL-3B-Instruct
# - Defaults to CPU (avoids Apple MPS crashes)
# - Optional 4-bit on CUDA
# - Stricter JSON control + resilient fallback
# - Threads/seed knobs for repeatable, CPU-friendly runs

from __future__ import annotations
import json, re, math, os, torch
from typing import List, Dict, Any, Optional
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def _resize_keep_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    if w >= h:
        return img.resize((max_side, max(1, int(h * max_side / w))), Image.LANCZOS)
    return img.resize((max(1, int(w * max_side / h)), max_side), Image.LANCZOS)


def _coerce_pil(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x
    return Image.fromarray(x)


def _strip_code_fence(text: str) -> str:
    """Return content inside ```json ... ``` or raw text if not fenced."""
    m = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.S)
    return m.group(1) if m else text


def _extract_first_json(text: str) -> Optional[str]:
    """
    Try to extract the first JSON array/object from a messy string by bracket matching.
    """
    text = text.strip()
    # Prefer fenced first
    fenced = _strip_code_fence(text)
    if fenced != text:
        return fenced

    # Try to find first JSON array/object by scanning
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        if start != -1:
            depth = 0
            for i, ch in enumerate(text[start:], start=start):
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        return candidate
    return None


class LocalVLM_Qwen:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: Optional[str] = None,
        use_4bit_if_cuda: bool = True,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        max_image_side: int = 320,
        seed: Optional[int] = 42,
        cpu_threads: Optional[int] = None,
    ):
        """
        device: "cpu" | "cuda" (recommended) | "mps" (not recommended for this model on some Macs)
        """
        # Stable default: CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "mps":
            # Qwen2.5-VL can be unstable on MPS; steer to CPU unless you explicitly want MPS
            device = "cpu"

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = float(temperature)
        self.max_image_side = int(max_image_side)

        if seed is not None:
            try:
                torch.manual_seed(seed)
            except Exception:
                pass

        if cpu_threads is not None and self.device == "cpu":
            try:
                torch.set_num_threads(int(cpu_threads))
            except Exception:
                pass

        load_args = {}
        if self.device == "cuda":
            if use_4bit_if_cuda:
                # 4-bit quantization reduces VRAM; requires bitsandbytes installed
                load_args.update(dict(load_in_4bit=True, device_map="auto"))
            else:
                load_args.update(dict(torch_dtype=torch.float16, device_map="auto"))
        else:
            # CPU
            load_args.update(dict(torch_dtype=torch.float32))  # keep it simple & stable

        # Processor + model
        # trust_remote_code is required for Qwen2.5-VL chat template and multimodal processing
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            **load_args
        ).to(self.device)

    def _build_prompt(self, n_items: int, listing: str) -> str:
        # Compact system: hard constraint + JSON schema
        return (
            "You are a fashion stylist. You will ONLY reply with STRICT JSON.\n"
            "Constraints:\n"
            "1) Create exactly 3 cohesive outfit combinations using the provided numbered items.\n"
            "2) Each combo MUST include at least one topwear and one bottomwear.\n"
            "3) You MAY add footwear and/or accessories.\n"
            f"4) Only use indices 1..{n_items}.\n\n"
            "JSON schema strictly as a JSON array:\n"
            '[{"combo":[<indices>],"description":"<1-2 sentence rationale>"}]\n\n'
            "Items:\n" + listing + "\n"
        )

    def _decode_to_json(self, text: str, n: int) -> List[Dict[str, Any]]:
        # Try fenced/first JSON
        candidate = _extract_first_json(text) or text
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                data = [data]
        except Exception:
            # Final fallback handled by _fallback_combo
            return []

        # sanitize
        clean: List[Dict[str, Any]] = []
        for rec in data:
            if not isinstance(rec, dict):
                continue
            combo = rec.get("combo", [])
            if not isinstance(combo, list):
                combo = []
            # keep 1..n, unique, int
            fixed = []
            for x in combo:
                try:
                    xi = int(x)
                    if 1 <= xi <= n and xi not in fixed:
                        fixed.append(xi)
                except Exception:
                    continue
            desc = rec.get("description", "")
            clean.append({"combo": fixed, "description": str(desc)})
        # keep non-empty combos
        clean = [r for r in clean if r["combo"]]
        return clean[:3] if clean else []

    @staticmethod
    def _fallback_combo(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tops = [i + 1 for i, it in enumerate(items) if it.get("section") == "topwear"]
        bots = [i + 1 for i, it in enumerate(items) if it.get("section") == "bottomwear"]
        foot = [i + 1 for i, it in enumerate(items) if it.get("section") == "footwear"]
        accs = [i + 1 for i, it in enumerate(items) if it.get("section") == "accessories"]

        combos: List[List[int]] = []

        if tops and bots:
            # a couple of sensible combos
            combos.append([tops[0], bots[0]] + ([foot[0]] if foot else []))
            if len(tops) > 1 and len(bots) > 1:
                combos.append([tops[1], bots[1]] + ([foot[1]] if len(foot) > 1 else []))
            if accs:
                combos.append([tops[0], bots[0], accs[0]])
        else:
            # Best effort if one category missing
            base = []
            if tops:
                base.append(tops[0])
            if bots:
                base.append(bots[0])
            if not base and (foot or accs):
                base += foot[:1] + accs[:1]
            if base:
                combos.append(base)

        # ensure exactly 3 outputs if possible (pad by echoing variations)
        out = []
        for c in combos[:3]:
            out.append({"combo": c, "description": "Auto fallback combo"})
        while len(out) < 3 and combos:
            out.append({"combo": combos[0], "description": "Auto fallback combo"})
        return out[:3] if out else [{"combo": [], "description": "No valid combo"}]

    def recommend(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        items: [{"section": "topwear|bottomwear|footwear|accessories", "rgba": PIL.Image}, ...]
        returns: [{"combo":[1-based indices], "description": str}, ...]   (exactly 3 when possible)
        """
        n = len(items)
        if n == 0:
            return []

        # Build listing and collect downscaled RGBs
        listing = "\n".join(f"{i + 1}. {it.get('section','unknown')}" for i, it in enumerate(items))
        images: List[Image.Image] = []
        image_content = []
        for it in items:
            img = _coerce_pil(it["rgba"]).convert("RGB")
            img = _resize_keep_max_side(img, self.max_image_side)
            images.append(img)
            image_content.append({"type": "image"})  # placeholder; processor inserts image tokens

        # Messages → chat template (Qwen requires trust_remote_code=True on processor)
        messages = [
            {"role": "system", "content": "You are a precise fashion stylist. Respond ONLY with valid JSON."},
            {"role": "user", "content": [{"type": "text", "text": self._build_prompt(n, listing)}] + image_content},
        ]
        chat_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[chat_text],
            images=images,
            return_tensors="pt"
        ).to(self.device)

        # Generation
        with torch.inference_mode():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=int(self.max_new_tokens),
                do_sample=(self.temperature > 0),
                temperature=float(self.temperature),
                repetition_penalty=1.05,
            )
        text = self.processor.batch_decode(gen, skip_special_tokens=True)[0]

        # Parse → JSON (strict) → sanitize
        parsed = self._decode_to_json(text, n)
        if parsed and len(parsed) >= 1:
            # Ensure up to 3 recs
            if len(parsed) > 3:
                parsed = parsed[:3]
            # If any combo violates constraints, we keep it but it already filtered invalid indices.
            return parsed

        # Fallback
        return self._fallback_combo(items)
