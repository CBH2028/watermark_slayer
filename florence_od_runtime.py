from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
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

