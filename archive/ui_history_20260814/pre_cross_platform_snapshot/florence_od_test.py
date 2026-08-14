from __future__ import annotations

import argparse
import base64
import json
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor

try:
    from peft import PeftModel
    HAVE_PEFT = True
except Exception:
    PeftModel = None
    HAVE_PEFT = False

try:
    from transformers import Florence2ForConditionalGeneration
    NATIVE_FLORENCE2 = True
except Exception:
    Florence2ForConditionalGeneration = None
    NATIVE_FLORENCE2 = False


DEFAULT_MODEL_ID = "/home/h3c/cbh_ws/florence_train/fused_model"
DEFAULT_ADAPTER_DIR = ""
DEFAULT_TASK = "<OD>"


def canonical_label(label: str, fallback: str = "watermark") -> str:
    label = (label or "").strip().lower()
    cleaned = []
    prev_us = False
    for ch in label:
        keep = ch.isascii() and (ch.isalnum() or ch == "_")
        if keep:
            cleaned.append(ch)
            prev_us = False
        else:
            if not prev_us:
                cleaned.append("_")
                prev_us = True
    out = "".join(cleaned).strip("_")
    return out or fallback


def parse_optional_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    s = canonical_label(str(label), fallback="")
    if s in ("", "none", "null"):
        return None
    return s


def merge_instances_labels(
    instances: Sequence[Dict[str, Any]],
    merge_all_labels_to: Optional[str] = None,
    fallback: str = "object",
) -> List[Dict[str, Any]]:
    target = parse_optional_label(merge_all_labels_to)
    out: List[Dict[str, Any]] = []
    for inst in instances:
        item = dict(inst)
        if target is not None:
            item["label"] = target
        else:
            item["label"] = canonical_label(str(item.get("label", "")), fallback=fallback)
        out.append(item)
    return out


def clamp_box_to_image(box: Sequence[float], image_size: Tuple[int, int]) -> List[float]:
    width, height = image_size
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(0.0, x1), float(width))
    y1 = min(max(0.0, y1), float(height))
    x2 = min(max(x1, x2), float(width))
    y2 = min(max(y1, y2), float(height))
    return [x1, y1, x2, y2]


def collect_detector_predictions(parsed: Any, task: str = "<OD>") -> List[Dict[str, Any]]:
    payload = parsed
    if isinstance(parsed, dict) and task in parsed:
        payload = parsed[task]

    preds: List[Dict[str, Any]] = []

    if isinstance(payload, dict):
        boxes = payload.get("bboxes") or payload.get("boxes") or []
        labels = payload.get("labels") or payload.get("classes") or payload.get("categories") or payload.get("bboxes_labels") or []
        scores = payload.get("scores") or []
        for i, box in enumerate(boxes):
            label = labels[i] if i < len(labels) else "object"
            if isinstance(label, dict):
                label = label.get("label") or label.get("text") or "object"
            score = scores[i] if i < len(scores) else None
            preds.append(
                {
                    "label": canonical_label(str(label), fallback="object"),
                    "bbox_xyxy": [float(v) for v in box],
                    "score": None if score is None else float(score),
                }
            )
        return preds

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            box = item.get("bbox_xyxy") or item.get("bboxes") or item.get("box") or item.get("bbox")
            if box is None:
                continue
            label = item.get("label") or item.get("text") or item.get("category") or "object"
            score = item.get("score")
            preds.append(
                {
                    "label": canonical_label(str(label), fallback="object"),
                    "bbox_xyxy": [float(v) for v in box],
                    "score": None if score is None else float(score),
                }
            )
        return preds

    return preds


def first_float_dtype(model: torch.nn.Module) -> torch.dtype:
    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.float32


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def guess_processor_source(model_id: str, adapter_dir: Optional[str]) -> str:
    if not adapter_dir:
        return model_id
    adapter_path = Path(adapter_dir)
    names = [
        "processor_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]
    if any((adapter_path / name).exists() for name in names):
        return str(adapter_path)
    return model_id


def _load_processor(
    processor_source: str,
    trust_remote_code: bool,
    use_fast_processor: bool,
) -> Tuple[Any, str]:
    tried_modes: List[str] = []

    def _attempt(**kwargs):
        return AutoProcessor.from_pretrained(processor_source, **kwargs)

    if use_fast_processor:
        fast_kwargs = {"use_fast": True}
        if trust_remote_code:
            fast_kwargs["trust_remote_code"] = True
        try:
            processor = _attempt(**fast_kwargs)
            return processor, "use_fast=True"
        except TypeError:
            tried_modes.append("use_fast=True")
        except Exception:
            tried_modes.append("use_fast=True")

        backend_kwargs = {"backend": "torchvision"}
        if trust_remote_code:
            backend_kwargs["trust_remote_code"] = True
        try:
            processor = _attempt(**backend_kwargs)
            return processor, 'backend="torchvision"'
        except TypeError:
            tried_modes.append('backend="torchvision"')
        except Exception:
            tried_modes.append('backend="torchvision"')

    slow_kwargs: Dict[str, Any] = {}
    if trust_remote_code:
        slow_kwargs["trust_remote_code"] = True
    if use_fast_processor:
        slow_kwargs["use_fast"] = False
    try:
        processor = _attempt(**slow_kwargs)
        if use_fast_processor and tried_modes:
            return processor, f"fallback use_fast=False after {', '.join(tried_modes)}"
        return processor, "use_fast=False"
    except TypeError:
        slow_kwargs.pop("use_fast", None)
        processor = _attempt(**slow_kwargs)
        if use_fast_processor and tried_modes:
            return processor, f"fallback default processor after {', '.join(tried_modes)}"
        return processor, "default"


def load_detector_stack(
    model_id: str,
    adapter_dir: Optional[str],
    use_fast_processor: bool,
) -> Tuple[torch.nn.Module, Any, torch.device, torch.dtype, str, str]:
    dtype = pick_dtype()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor_source = guess_processor_source(model_id, adapter_dir)
    native_err = None

    if NATIVE_FLORENCE2:
        try:
            model = Florence2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
            processor, processor_mode = _load_processor(
                processor_source,
                trust_remote_code=False,
                use_fast_processor=use_fast_processor,
            )
            if adapter_dir:
                if not HAVE_PEFT:
                    raise RuntimeError("adapter_dir was provided, but peft is not installed")
                model = PeftModel.from_pretrained(model, adapter_dir)
            model = model.to(device)
            model.eval()
            return model, processor, device, dtype, "native_florence2", processor_mode
        except Exception as e:
            native_err = e

    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype)
        processor, processor_mode = _load_processor(
            processor_source,
            trust_remote_code=True,
            use_fast_processor=use_fast_processor,
        )
        if adapter_dir:
            if not HAVE_PEFT:
                raise RuntimeError("adapter_dir was provided, but peft is not installed")
            model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.to(device)
        model.eval()
        return model, processor, device, dtype, "remote_code_fallback", processor_mode
    except Exception as e:
        if native_err is not None:
            raise RuntimeError(
                f"Failed to load Florence-2 with native path ({native_err}) and fallback path ({e})."
            ) from e
        raise


def move_batch_to_device(
    inputs: Dict[str, Any],
    device: torch.device,
    pixel_values_dtype: torch.dtype,
) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if key == "pixel_values":
                moved[key] = value.to(device=device, dtype=pixel_values_dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def draw_box(draw: ImageDraw.ImageDraw, box: Sequence[float], color: Tuple[int, int, int], width: int = 3) -> None:
    x1, y1, x2, y2 = [float(v) for v in box]
    for offset in range(width):
        draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)


def draw_text_block(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    fill: Tuple[int, int, int],
    bg: Tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle([left - 2, top - 2, right + 2, bottom + 2], fill=bg)
    draw.text((x, y), text, font=font, fill=fill)


def annotate_image(image: Image.Image, preds: Sequence[Dict[str, Any]]) -> Image.Image:
    vis = image.copy().convert("RGB")
    draw = ImageDraw.Draw(vis)
    font = ImageFont.load_default()
    pred_color = (255, 0, 0)
    text_fill = (255, 255, 255)

    for idx, pred in enumerate(preds):
        box = pred.get("bbox_xyxy", [])
        label = str(pred.get("label", f"pred_{idx}"))
        score = pred.get("score")
        text = f"P:{label}" if score is None else f"P:{label}:{score:.3f}"
        draw_box(draw, box, pred_color, width=3)
        draw_text_block(draw, (box[0], min(vis.height - 12, max(0.0, box[1]))), text, text_fill, pred_color, font)

    return vis


def image_to_base64_png(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def run_detector_sample(args: argparse.Namespace) -> Dict[str, Any]:
    image_path = Path(args.image).expanduser().resolve()
    image = Image.open(image_path).convert("RGB")
    merged_label = parse_optional_label(args.merge_all_labels_to)

    model, processor, device, requested_dtype, load_mode, processor_mode = load_detector_stack(
        args.model_id,
        args.adapter_dir,
        args.use_fast_processor,
    )
    pixel_values_dtype = first_float_dtype(model)
    use_amp = device.type == "cuda" and pixel_values_dtype in (torch.float16, torch.bfloat16)

    inputs = processor(text=args.task, images=image, return_tensors="pt")
    inputs = move_batch_to_device(inputs, device, pixel_values_dtype)

    with torch.inference_mode():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=pixel_values_dtype, enabled=True):
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                )
        else:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(generated_text, task=args.task, image_size=image.size)

    preds_raw = collect_detector_predictions(parsed, task=args.task)
    preds_raw = [
        {**p, "bbox_xyxy": clamp_box_to_image(p.get("bbox_xyxy", []), image.size)}
        for p in preds_raw
        if len(p.get("bbox_xyxy", [])) == 4
    ]
    preds = merge_instances_labels(preds_raw, merge_all_labels_to=merged_label, fallback="object")
    vis = annotate_image(image, preds_raw)

    return {
        "ok": True,
        "image": str(image_path),
        "image_size": {"width": image.width, "height": image.height},
        "model_id": args.model_id,
        "adapter_dir": args.adapter_dir,
        "task": args.task,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "load_mode": load_mode,
        "processor_mode": processor_mode,
        "device": str(device),
        "requested_dtype": str(requested_dtype),
        "pixel_values_dtype": str(pixel_values_dtype),
        "merge_all_labels_to": merged_label,
        "generated_text": generated_text,
        "parsed": parsed,
        "predictions_raw": preds_raw,
        "predictions": preds,
        "visualization": image_to_base64_png(vis),
    }


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single Florence-2 <OD> test with optional PEFT adapter.")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter_dir", type=str, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--merge_all_labels_to", type=str, default="watermark")
    parser.set_defaults(use_fast_processor=True)
    parser.add_argument("--use_slow_processor", dest="use_fast_processor", action="store_false")
    return parser.parse_cli_args()


def test_entry() -> None:
    args = parse_cli_args()
    try:
        result = run_detector_sample(args)
    except Exception as e:
        result = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    test_entry()
