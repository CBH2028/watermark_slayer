import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
import click
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Monkey-patch: cached_download was removed in huggingface_hub 0.24, add compatibility shim
import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

from transformers import AutoProcessor
try:
    from transformers import Florence2ForConditionalGeneration
except Exception:
    Florence2ForConditionalGeneration = None
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
import torch
from torch.nn import Module
import tqdm
from loguru import logger
from enum import Enum
import os
import tempfile
import shutil
import subprocess
import re
import json
import base64
from io import BytesIO

from florence_od_runtime import (
    DEFAULT_ADAPTER_DIR,
    DEFAULT_MODEL_ID,
    collect_detector_predictions,
    first_float_dtype,
    load_detector_stack,
    move_batch_to_device,
    clamp_box_to_image,
    canonical_label,
)

try:
    from cv2.typing import MatLike
except ImportError:
    MatLike = np.ndarray


LIVE_FRAME_PREFIX = "WM_SLAYER_LIVE_FRAME:"
LIVE_FRAME_MAX_SIDE = 960
LIVE_FRAME_JPEG_QUALITY = 82


def _preview_image_payload(image: Image.Image):
    """Build a compact browser image payload for live GUI updates."""
    preview_image = image.convert("RGB").copy()
    preview_image.thumbnail((LIVE_FRAME_MAX_SIDE, LIVE_FRAME_MAX_SIDE))
    buffer = BytesIO()
    preview_image.save(buffer, format="JPEG", quality=LIVE_FRAME_JPEG_QUALITY, optimize=True)
    return {
        "kind": "image",
        "mime": "image/jpeg",
        "data": base64.b64encode(buffer.getvalue()).decode("utf-8"),
    }


def _detection_items_from_boxes(image: Image.Image, boxes, output_label: str = "watermark"):
    """Convert plain xyxy boxes into the same shape used by Florence detections."""
    image_area = max(1, image.width * image.height)
    items = []
    for box in boxes or []:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in clamp_box_to_image(box, image.size)]
        area_percent = ((x2 - x1) * (y2 - y1) / image_area) * 100
        items.append({
            "bbox": [x1, y1, x2, y2],
            "area_percent": round(area_percent, 2),
            "accepted": True,
            "raw_label": output_label,
            "query_label": output_label,
            "output_label": output_label,
        })
    return items


def _mask_from_detection_items(image: Image.Image, detections):
    """Create an inpainting mask from accepted detection items."""
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    for item in detections or []:
        x1, y1, x2, y2 = item["bbox"]
        if item.get("accepted", True):
            draw.rectangle([x1, y1, x2, y2], fill=255)
        else:
            logger.warning(f"Skipping large bounding box: {item['bbox']} covering {item['area_percent']:.2f}% of the image")

    return mask


def _draw_detection_preview(image: Image.Image, detections):
    """Draw Florence boxes on a UI-only copy of the frame."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(2, min(8, annotated.width // 220))

    for item in detections or []:
        if not item.get("bbox") or len(item["bbox"]) != 4:
            continue
        x1, y1, x2, y2 = item["bbox"]
        accepted = item.get("accepted", True)
        color = (34, 197, 94) if accepted else (239, 68, 68)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        label = str(item.get("raw_label") or item.get("query_label") or item.get("output_label") or "watermark")
        area = item.get("area_percent")
        if isinstance(area, (int, float)):
            label = f"{label} {area:.1f}%"
        text_x = max(0, x1)
        text_y = max(0, y1 - 18)
        try:
            text_box = draw.textbbox((text_x, text_y), label)
            pad = 3
            draw.rectangle(
                [text_box[0] - pad, text_box[1] - pad, text_box[2] + pad, text_box[3] + pad],
                fill=(15, 23, 42),
            )
        except Exception:
            pass
        draw.text((text_x, text_y), label, fill=color)

    return annotated


def _live_detection_payload(detections):
    """Trim detection objects to JSON-safe values for frontend display."""
    payload = []
    for item in detections or []:
        if not item.get("bbox") or len(item["bbox"]) != 4:
            continue
        record = {
            "bbox": [int(round(value)) for value in item["bbox"]],
            "accepted": bool(item.get("accepted", True)),
            "raw_label": str(item.get("raw_label") or ""),
            "query_label": str(item.get("query_label") or ""),
            "output_label": str(item.get("output_label") or ""),
        }
        area = item.get("area_percent")
        if isinstance(area, (int, float, np.number)):
            record["area_percent"] = round(float(area), 2)
        score = item.get("score")
        if isinstance(score, (int, float, np.number)):
            record["score"] = float(score)
        payload.append(record)
    return payload


def _emit_live_frame(phase: str, frame_number: int, total_frames: int, before_image: Image.Image = None, after_image: Image.Image = None, detections=None):
    """Emit a structured live-frame event for the GUI bridge."""
    payload = {
        "phase": phase,
        "frame": frame_number,
        "frame_index": max(0, frame_number - 1),
        "total_frames": total_frames,
        "detections": _live_detection_payload(detections),
    }
    if before_image is not None:
        payload["before"] = _preview_image_payload(before_image)
    if after_image is not None:
        payload["after"] = _preview_image_payload(after_image)

    print(f"{LIVE_FRAME_PREFIX}{json.dumps(payload, separators=(',', ':'))}", flush=True)


def fetch_inpaint_model():
    """Download LaMA model using iopaint."""
    logger.info("Downloading LaMA model... (this may take a few minutes)")
    print("Downloading LaMA model (~196MB)... Please wait.")

    result = subprocess.run(
        [sys.executable, "-m", "iopaint", "download", "--model", "lama"],
        capture_output=False,  # Show download progress
        text=True
    )

    if result.returncode != 0:
        logger.error("Failed to download LaMA model")
        return False

    logger.info("LaMA model downloaded successfully")
    print("LaMA model downloaded!")
    return True


def load_inpaint_engine(device):
    """Load LaMA model, downloading if necessary."""
    try:
        return ModelManager(name="lama", device=device)
    except NotImplementedError as e:
        if "Unsupported model: lama" in str(e):
            print("LaMA model not available, attempting to download...")
            if fetch_inpaint_model():
                # Re-import to refresh model registry
                import importlib
                import iopaint.model
                importlib.reload(iopaint.model)
                # Try again
                return ModelManager(name="lama", device=device)
            else:
                raise RuntimeError("Failed to download LaMA model. Please run manually: python\\python.exe -m iopaint download --model lama")
        raise

class SlayerVisionTask(str, Enum):
    OD = "<OD>"
    """Object detection with labels emitted by the model"""

    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    """Detect bounding box for objects and OCR text"""

def run_detector_generation(
    task_prompt: SlayerVisionTask,
    image: MatLike,
    text_input: str,
    model: Florence2ForConditionalGeneration,
    processor: AutoProcessor,
    device: str,
    max_new_tokens: int = 256,
    num_beams: int = 3,
):
    if not isinstance(task_prompt, SlayerVisionTask):
        raise ValueError(f"task_prompt must be a SlayerVisionTask, but {task_prompt} is of type {type(task_prompt)}")

    prompt = task_prompt.value if text_input is None else task_prompt.value + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    device_obj = device if isinstance(device, torch.device) else torch.device(device)
    pixel_values_dtype = first_float_dtype(model)
    inputs = move_batch_to_device(inputs, device_obj, pixel_values_dtype)
    use_amp = device_obj.type == "cuda" and pixel_values_dtype in (torch.float16, torch.bfloat16)

    with torch.inference_mode():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=pixel_values_dtype, enabled=True):
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )
        else:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text, task=task_prompt.value, image_size=(image.width, image.height)
    )
    predictions_raw = collect_detector_predictions(parsed, task=task_prompt.value)
    return {
        "prompt": prompt,
        "task": task_prompt.value,
        "generated_text": generated_text,
        "parsed": parsed,
        "predictions_raw": predictions_raw,
    }


def parse_detector_answer(
    task_prompt: SlayerVisionTask,
    image: MatLike,
    text_input: str,
    model: Florence2ForConditionalGeneration,
    processor: AutoProcessor,
    device: str,
    max_new_tokens: int = 256,
    num_beams: int = 3,
):
    return run_detector_generation(
        task_prompt,
        image,
        text_input,
        model,
        processor,
        device,
        max_new_tokens,
        num_beams,
    )["parsed"]

def parse_raw_class_list(detection_classes: str):
    """Parse comma/space separated raw detection labels."""
    if not detection_classes:
        return []

    parts = re.split(r"[,;|\s]+", detection_classes)
    result = []
    seen = set()
    for part in parts:
        label = part.strip()
        if label and label not in seen:
            seen.add(label)
            result.append(label)
    return result


def choose_detector_task(detection_task: str, detection_classes=None):
    """Choose Florence task mode.

    auto keeps the legacy open-vocabulary path unless raw classes are selected.
    Fine-tuned datasets that use <OD> should select raw classes and therefore
    run object detection.
    """
    task = (detection_task or "auto").strip().lower()
    if task in {"od", "<od>", "object_detection", "object-detection"}:
        return SlayerVisionTask.OD
    if task in {"open_vocab", "open-vocab", "open_vocabulary", "open-vocabulary", "<open_vocabulary_detection>"}:
        return SlayerVisionTask.OPEN_VOCAB_DETECTION
    if detection_classes:
        return SlayerVisionTask.OD
    return SlayerVisionTask.OPEN_VOCAB_DETECTION


def _iter_detection_queries(detection_prompt: str, detection_classes=None):
    classes = detection_classes or []
    return classes if classes else [detection_prompt or "watermark"]


def _shape_detector_regions(predictions_raw, image, max_bbox_percent: float, query_label: str, output_label: str, selected_labels=None):
    results = []
    image_area = image.width * image.height
    selected_labels = selected_labels or set()

    for pred in predictions_raw:
        bbox = pred.get("bbox_xyxy", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in clamp_box_to_image(bbox, image.size)]
        bbox_area = (x2 - x1) * (y2 - y1)
        area_percent = (bbox_area / image_area) * 100
        raw_label = canonical_label(str(pred.get("label") or query_label), fallback=str(query_label))
        within_area_limit = area_percent <= max_bbox_percent
        selected_by_ui = (not selected_labels) or raw_label in selected_labels
        results.append({
            "bbox": [x1, y1, x2, y2],
            "area_percent": round(area_percent, 2),
            "accepted": within_area_limit and selected_by_ui,
            "within_area_limit": within_area_limit,
            "raw_label": raw_label,
            "query_label": query_label,
            "output_label": output_label,
            "selected_by_ui": selected_by_ui,
            "score": pred.get("score"),
        })

    return results


def _run_region_detection(image: MatLike, model: Florence2ForConditionalGeneration, processor: AutoProcessor, device: str, max_bbox_percent: float, detection_prompt: str = "watermark", detection_classes=None, detection_output_label: str = "watermark", detection_task: str = "auto", florence_max_new_tokens: int = 256, florence_num_beams: int = 3):
    task_prompt = choose_detector_task(detection_task, detection_classes)
    results = []
    raw_runs = []
    selected_labels = {canonical_label(str(label), fallback="") for label in (detection_classes or [])}

    if task_prompt == SlayerVisionTask.OD:
        raw_output = run_detector_generation(task_prompt, image, None, model, processor, device, florence_max_new_tokens, florence_num_beams)
        raw_runs.append(raw_output)
        results.extend(
            _shape_detector_regions(
                raw_output["predictions_raw"],
                image,
                max_bbox_percent,
                task_prompt.value,
                detection_output_label,
                selected_labels,
            )
        )
        if not any(item.get("accepted", False) for item in results) and (detection_prompt or "").strip():
            fallback_output = run_detector_generation(
                SlayerVisionTask.OPEN_VOCAB_DETECTION,
                image,
                detection_prompt,
                model,
                processor,
                device,
                florence_max_new_tokens,
                florence_num_beams,
            )
            raw_runs.append(fallback_output)
            results.extend(
                _shape_detector_regions(
                    fallback_output["predictions_raw"],
                    image,
                    max_bbox_percent,
                    detection_prompt,
                    detection_output_label,
                    set(),
                )
            )
        return {
            "detections": results,
            "raw_runs": raw_runs,
            "generated_text": "\n".join(run["generated_text"] for run in raw_runs),
            "parsed": [run["parsed"] for run in raw_runs],
            "predictions_raw": [pred for run in raw_runs for pred in run["predictions_raw"]],
        }

    for target in _iter_detection_queries(detection_prompt, detection_classes):
        raw_output = run_detector_generation(task_prompt, image, target, model, processor, device, florence_max_new_tokens, florence_num_beams)
        raw_runs.append(raw_output)
        results.extend(
            _shape_detector_regions(
                raw_output["predictions_raw"],
                image,
                max_bbox_percent,
                target,
                detection_output_label,
                selected_labels,
            )
        )

    return {
        "detections": results,
        "raw_runs": raw_runs,
        "generated_text": "\n".join(run["generated_text"] for run in raw_runs),
        "parsed": [run["parsed"] for run in raw_runs],
        "predictions_raw": [pred for run in raw_runs for pred in run["predictions_raw"]],
    }


def _list_detected_regions(image: MatLike, model: Florence2ForConditionalGeneration, processor: AutoProcessor, device: str, max_bbox_percent: float, detection_prompt: str = "watermark", detection_classes=None, detection_output_label: str = "watermark", detection_task: str = "auto", florence_max_new_tokens: int = 256, florence_num_beams: int = 3):
    return _run_region_detection(
        image,
        model,
        processor,
        device,
        max_bbox_percent,
        detection_prompt,
        detection_classes,
        detection_output_label,
        detection_task,
        florence_max_new_tokens,
        florence_num_beams,
    )["detections"]


def build_region_mask(image: MatLike, model: Florence2ForConditionalGeneration, processor: AutoProcessor, device: str, max_bbox_percent: float, detection_prompt: str = "watermark", detection_classes=None, detection_output_label: str = "watermark", detection_task: str = "auto", florence_max_new_tokens: int = 256, florence_num_beams: int = 3):
    """
    Detect watermarks and create a mask for inpainting.

    Args:
        image: PIL Image
        model: Florence-2 model
        processor: Florence-2 processor
        device: cuda or cpu
        max_bbox_percent: Maximum bbox size as percentage of image
        detection_prompt: Fallback text prompt for detection (e.g. "watermark", "watermark Sora logo", "Getty Images")
        detection_classes: Raw class labels from a fine-tuned Florence-2 model.
        detection_output_label: Unified label used by the application for selected raw classes.
    """
    detections = _list_detected_regions(image, model, processor, device, max_bbox_percent, detection_prompt, detection_classes, detection_output_label, detection_task, florence_max_new_tokens, florence_num_beams)
    return _mask_from_detection_items(image, detections)


def preview_detected_regions(image: MatLike, model: Florence2ForConditionalGeneration, processor: AutoProcessor, device: str, max_bbox_percent: float, detection_prompt: str = "watermark", detection_classes=None, detection_output_label: str = "watermark", detection_task: str = "auto", florence_max_new_tokens: int = 256, florence_num_beams: int = 3):
    """
    Detect watermarks and return bounding boxes WITHOUT creating mask or inpainting.
    Used for preview mode to show what would be detected.

    Returns:
        list of dicts with bbox info: [{"bbox": [x1,y1,x2,y2], "area_percent": float, "accepted": bool}, ...]
    """
    return _list_detected_regions(image, model, processor, device, max_bbox_percent, detection_prompt, detection_classes, detection_output_label, detection_task, florence_max_new_tokens, florence_num_beams)


def preview_detector_payload(image: MatLike, model: Florence2ForConditionalGeneration, processor: AutoProcessor, device: str, max_bbox_percent: float, detection_prompt: str = "watermark", detection_classes=None, detection_output_label: str = "watermark", detection_task: str = "auto", florence_max_new_tokens: int = 256, florence_num_beams: int = 3):
    """Return Florence raw output plus detection items derived from every raw OD box."""
    return _run_region_detection(image, model, processor, device, max_bbox_percent, detection_prompt, detection_classes, detection_output_label, detection_task, florence_max_new_tokens, florence_num_beams)

def inpaint_image_array(image: MatLike, mask: MatLike, model_manager: ModelManager):
    config = Config(
        ldm_steps=50,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1600,
    )
    result = model_manager(image, mask, config)

    if result.dtype in [np.float64, np.float32]:
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def cut_mask_to_transparency(image: Image.Image, mask: Image.Image):
    image = image.convert("RGBA")
    mask = mask.convert("L")
    transparent_image = Image.new("RGBA", image.size)
    for x in range(image.width):
        for y in range(image.height):
            if mask.getpixel((x, y)) > 0:
                transparent_image.putpixel((x, y), (0, 0, 0, 0))
            else:
                transparent_image.putpixel((x, y), image.getpixel((x, y)))
    return transparent_image

def is_supported_video(file_path):
    """Check if the file is a video based on its extension"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    return Path(file_path).suffix.lower() in video_extensions

def process_video_framewise(input_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, detection_prompt="watermark", detection_classes=None, detection_output_label="watermark", detection_task="auto", florence_max_new_tokens=256, florence_num_beams=3, progress_offset=0, progress_scale=100):
    """Process a video file by extracting frames, removing watermarks, and reconstructing the video"""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Error opening video file: {input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine output format
    if force_format:
        output_format = force_format.upper()
    else:
        output_format = "MP4"  # Default to MP4 for videos
    
    # Create output video file
    output_path = Path(output_path)
    if output_path.is_dir():
        output_file = output_path / f"{input_path.stem}_no_watermark.{output_format.lower()}"
    else:
        output_file = output_path.with_suffix(f".{output_format.lower()}")
    
    # Create a temporary file for the video without audio
    temp_dir = tempfile.mkdtemp()
    temp_video_path = Path(temp_dir) / f"temp_no_audio.{output_format.lower()}"
    
    # Set codec based on output format
    if output_format.upper() == "MP4":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif output_format.upper() == "AVI":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Default to MP4
    
    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))
    
    # Process each frame
    with tqdm.tqdm(total=total_frames, desc="Processing video frames") as pbar:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Run Florence and show the UI-only boxed frame before inpainting.
            detection_result = preview_detector_payload(pil_image, florence_model, florence_processor, device, max_bbox_percent, detection_prompt, detection_classes, detection_output_label, detection_task, florence_max_new_tokens, florence_num_beams)
            detections = detection_result["detections"]
            annotated_frame = _draw_detection_preview(pil_image, detections)
            _emit_live_frame("detected", frame_count + 1, total_frames, before_image=annotated_frame, detections=detections)

            # Get watermark mask
            mask_image = _mask_from_detection_items(pil_image, detections)
            
            # Process frame
            if transparent:
                # For video, we can't use transparency, so we'll fill with a color or background
                result_image = cut_mask_to_transparency(pil_image, mask_image)
                # Convert RGBA to RGB by filling transparent areas with white
                background = Image.new("RGB", result_image.size, (255, 255, 255))
                background.paste(result_image, mask=result_image.split()[3])
                result_image = background
            else:
                lama_result = inpaint_image_array(np.array(pil_image), np.array(mask_image), model_manager)
                result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))
            
            # Convert back to OpenCV format and write to output video
            frame_result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            out.write(frame_result)
            _emit_live_frame("processed", frame_count + 1, total_frames, after_image=result_image, detections=detections)
            
            # Update progress
            frame_count += 1
            pbar.update(1)
            local_progress = frame_count / total_frames
            progress = int(progress_offset + local_progress * progress_scale)
            print(f"Processing frame {frame_count}/{total_frames}, overall_progress:{progress}%")
    
    # Release resources
    cap.release()
    out.release()
    
    # Combine processed video with original audio using FFmpeg
    try:
        logger.info("Merging processed video with original audio...")
        
        # Check if FFmpeg is available
        try:
            subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("FFmpeg is not available. Video will be produced without audio.")
            shutil.copy(str(temp_video_path), str(output_file))
        else:
            # Re-encode MP4 as H.264/AAC so the GUI/browser has the best chance to play it.
            if output_format.upper() == "MP4":
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(temp_video_path),
                    "-i", str(input_path),
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-map", "0:v:0",
                    "-map", "1:a:0?",
                    "-shortest",
                    "-movflags", "+faststart",
                    str(output_file)
                ]
            else:
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(temp_video_path),
                    "-i", str(input_path),
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-map", "0:v:0",
                    "-map", "1:a:0?",
                    "-shortest",
                    str(output_file)
                ]

            # Execute FFmpeg
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Audio/video merge completed successfully!")
    except Exception as e:
        logger.error(f"Error during audio/video merge: {str(e)}")
        # In case of error, use video without audio
        shutil.copy(str(temp_video_path), str(output_file))
    finally:
        # Clean up temporary files
        try:
            os.remove(str(temp_video_path))
            os.rmdir(temp_dir)
        except:
            pass
    
    final_progress = progress_offset + progress_scale
    logger.info(f"input_path:{input_path}, output_path:{output_file}, overall_progress:{final_progress}")
    return output_file


def process_video_timeline_passes(input_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, detection_prompt="watermark", detection_classes=None, detection_output_label="watermark", detection_task="auto", detection_skip=1, fade_in_sec=0.0, fade_out_sec=0.0, florence_max_new_tokens=256, florence_num_beams=3, progress_offset=0, progress_scale=100):
    """
    Two-pass video processing with frame skip detection and fade in/out handling.

    Pass 1: Detect watermarks every N frames (sparse detection)
    Pass 2: Apply inpainting to all frames using interpolated masks

    This is more efficient for videos where watermarks don't change rapidly,
    and handles fade in/out watermarks by extending the mask temporally.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Error opening video file: {input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Convert seconds to frames
    fade_in_frames = int(fade_in_sec * fps)
    fade_out_frames = int(fade_out_sec * fps)

    logger.info(f"Two-pass processing: {total_frames} frames, skip={detection_skip}, fade_in={fade_in_frames}f, fade_out={fade_out_frames}f")

    # Determine output format
    if force_format:
        output_format = force_format.upper()
    else:
        output_format = "MP4"

    # Create output video file
    output_path = Path(output_path)
    if output_path.is_dir():
        output_file = output_path / f"{input_path.stem}_no_watermark.{output_format.lower()}"
    else:
        output_file = output_path.with_suffix(f".{output_format.lower()}")

    # ========== PASS 1: DETECTION (sparse) ==========
    logger.info("Pass 1: Detecting watermarks...")
    detections = {}  # frame_idx -> [bbox, bbox, ...]
    detection_frames = list(range(0, total_frames, detection_skip))

    with tqdm.tqdm(total=len(detection_frames), desc="Pass 1: Detection") as pbar:
        for frame_idx in detection_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detection_result = preview_detector_payload(pil_image, florence_model, florence_processor, device, max_bbox_percent, detection_prompt, detection_classes, detection_output_label, detection_task, florence_max_new_tokens, florence_num_beams)
            bboxes = detection_result["detections"]
            annotated_frame = _draw_detection_preview(pil_image, bboxes)
            _emit_live_frame("detected", frame_idx + 1, total_frames, before_image=annotated_frame, detections=bboxes)

            if bboxes:
                accepted_bboxes = [b["bbox"] for b in bboxes if b["accepted"]]
                if accepted_bboxes:
                    detections[frame_idx] = accepted_bboxes

            pbar.update(1)
            local_progress = (pbar.n / len(detection_frames)) * 0.5  # Pass 1 = 0-50% local
            progress = int(progress_offset + local_progress * progress_scale)
            print(f"Pass 1: frame {frame_idx}/{total_frames}, overall_progress:{progress}%")

    logger.info(f"Pass 1 complete: found watermarks in {len(detections)} detection points")

    # ========== TIMELINE EXPANSION ==========
    # Create frame->bbox mapping with fade in/out expansion
    frame_masks = {}  # frame_idx -> [bbox, ...]

    for det_frame, bboxes in detections.items():
        # Expand backwards (fade in) - watermark might be fading in before detection
        start_frame = max(0, det_frame - fade_in_frames)
        # Expand forwards (fade out) - continue masking after detection
        # Also include frames until next detection point
        end_frame = min(total_frames, det_frame + detection_skip + fade_out_frames)

        for f in range(start_frame, end_frame):
            if f not in frame_masks:
                frame_masks[f] = []
            # Add bboxes, avoiding duplicates
            for bbox in bboxes:
                if bbox not in frame_masks[f]:
                    frame_masks[f].append(bbox)

    logger.info(f"Timeline expanded: {len(frame_masks)} frames will have inpainting applied")

    # ========== PASS 2: INPAINTING ==========
    logger.info("Pass 2: Applying inpainting...")

    # Create temporary file for video without audio
    temp_dir = tempfile.mkdtemp()
    temp_video_path = Path(temp_dir) / f"temp_no_audio.{output_format.lower()}"

    # Set codec
    if output_format.upper() == "MP4":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif output_format.upper() == "AVI":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))

    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    with tqdm.tqdm(total=total_frames, desc="Pass 2: Inpainting") as pbar:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_detections = _detection_items_from_boxes(
                pil_image,
                frame_masks.get(frame_idx, []),
                detection_output_label,
            )
            annotated_frame = _draw_detection_preview(pil_image, frame_detections)
            _emit_live_frame("detected", frame_idx + 1, total_frames, before_image=annotated_frame, detections=frame_detections)

            if frame_idx in frame_masks:
                # This frame needs inpainting
                # Create mask from bboxes
                mask = _mask_from_detection_items(pil_image, frame_detections)

                # Apply inpainting or transparency
                if transparent:
                    result_image = cut_mask_to_transparency(pil_image, mask)
                    background = Image.new("RGB", result_image.size, (255, 255, 255))
                    background.paste(result_image, mask=result_image.split()[3])
                    result_image = background
                else:
                    lama_result = inpaint_image_array(np.array(pil_image), np.array(mask), model_manager)
                    result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))

                frame_result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            else:
                # No watermark detected for this frame, copy original
                result_image = pil_image
                frame_result = frame

            out.write(frame_result)
            _emit_live_frame("processed", frame_idx + 1, total_frames, after_image=result_image, detections=frame_detections)
            frame_idx += 1
            pbar.update(1)
            local_progress = 0.5 + (frame_idx / total_frames) * 0.5  # Pass 2 = 50-100% local
            progress = int(progress_offset + local_progress * progress_scale)
            print(f"Pass 2: frame {frame_idx}/{total_frames}, overall_progress:{progress}%")

    cap.release()
    out.release()

    # ========== MERGE WITH AUDIO ==========
    try:
        logger.info("Merging processed video with original audio...")
        try:
            subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("FFmpeg is not available. Video will be produced without audio.")
            shutil.copy(str(temp_video_path), str(output_file))
        else:
            if output_format.upper() == "MP4":
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(temp_video_path),
                    "-i", str(input_path),
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-map", "0:v:0",
                    "-map", "1:a:0?",
                    "-shortest",
                    "-movflags", "+faststart",
                    str(output_file)
                ]
            else:
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(temp_video_path),
                    "-i", str(input_path),
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-map", "0:v:0",
                    "-map", "1:a:0?",
                    "-shortest",
                    str(output_file)
                ]
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Audio/video merge completed successfully!")
    except Exception as e:
        logger.error(f"Error during audio/video merge: {str(e)}")
        shutil.copy(str(temp_video_path), str(output_file))
    finally:
        try:
            os.remove(str(temp_video_path))
            os.rmdir(temp_dir)
        except:
            pass

    final_progress = progress_offset + progress_scale
    logger.info(f"input_path:{input_path}, output_path:{output_file}, overall_progress:{final_progress}")
    return output_file


def process_single_asset(image_path: Path, output_path: Path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite, detection_prompt="watermark", detection_classes=None, detection_output_label="watermark", detection_task="auto", detection_skip=1, fade_in=0.0, fade_out=0.0, florence_max_new_tokens=256, florence_num_beams=3, progress_offset=0, progress_scale=100):
    # SAFETY: Never overwrite the input file
    if image_path.resolve() == output_path.resolve():
        logger.error(f"Cannot overwrite input file: {image_path}. Choose a different output path.")
        print(f"ERROR: Cannot overwrite input file! Choose a different output folder.")
        return

    if output_path.exists() and not overwrite:
        logger.info(f"Skipping existing file: {output_path}")
        return

    # Check if it's a video file
    if is_supported_video(image_path):
        # Use two-pass if detection_skip > 1 or fade handling is needed
        use_two_pass = detection_skip > 1 or fade_in > 0 or fade_out > 0
        if use_two_pass:
            return process_video_timeline_passes(image_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, detection_prompt, detection_classes, detection_output_label, detection_task, detection_skip, fade_in, fade_out, florence_max_new_tokens, florence_num_beams, progress_offset, progress_scale)
        else:
            return process_video_framewise(image_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, detection_prompt, detection_classes, detection_output_label, detection_task, florence_max_new_tokens, florence_num_beams, progress_offset, progress_scale)

    # Process image
    image = Image.open(image_path).convert("RGB")
    mask_image = build_region_mask(image, florence_model, florence_processor, device, max_bbox_percent, detection_prompt, detection_classes, detection_output_label, detection_task, florence_max_new_tokens, florence_num_beams)

    if transparent:
        result_image = cut_mask_to_transparency(image, mask_image)
    else:
        lama_result = inpaint_image_array(np.array(image), np.array(mask_image), model_manager)
        result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))

    # Determine output format
    if force_format:
        output_format = force_format.upper()
    elif transparent:
        output_format = "PNG"
    else:
        output_format = image_path.suffix[1:].upper()
        if output_format not in ["PNG", "WEBP", "JPG"]:
            output_format = "PNG"
    
    # Map JPG to JPEG for PIL compatibility
    if output_format == "JPG":
        output_format = "JPEG"

    if transparent and output_format == "JPG":
        logger.warning("Transparency detected. Defaulting to PNG for transparency support.")
        output_format = "PNG"

    new_output_path = output_path.with_suffix(f".{output_format.lower()}")
    result_image.save(new_output_path, format=output_format)
    # Report progress for this image (end of range)
    final_progress = progress_offset + progress_scale
    print(f"input_path:{image_path}, output_path:{new_output_path}, overall_progress:{final_progress}%")
    return new_output_path

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path(), required=False, default=None)
@click.option("--preview", is_flag=True, help="Preview mode: detect watermarks and output JSON with base64 image (no processing).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files in bulk mode.")
@click.option("--transparent", is_flag=True, help="Make watermark regions transparent instead of removing.")
@click.option("--max-bbox-percent", default=100.0, help="Maximum percentage of the image that a bounding box can cover.")
@click.option("--force-format", type=click.Choice(["PNG", "WEBP", "JPG", "MP4", "AVI"], case_sensitive=False), default=None, help="Force output format. Defaults to input format.")
@click.option("--detection-prompt", default="watermark", help="Text prompt for watermark detection (e.g. 'watermark', 'watermark Sora logo', 'Getty Images').")
@click.option("--detection-classes", default="", help="Comma separated model labels to accept (default single-class label: watermark).")
@click.option("--detection-output-label", default="watermark", help="Unified output label for selected raw detection classes.")
@click.option("--detection-task", default="auto", type=click.Choice(["auto", "od", "open_vocab"], case_sensitive=False), help="Florence detection task. auto uses <OD> when raw classes are selected.")
@click.option("--florence-model-id", default=DEFAULT_MODEL_ID, help="Base Florence-2 model path or Hub id.")
@click.option("--florence-adapter-dir", default=DEFAULT_ADAPTER_DIR, help="Optional PEFT adapter directory.")
@click.option("--florence-max-new-tokens", default=256, type=int, help="Florence generation max_new_tokens.")
@click.option("--florence-num-beams", default=3, type=int, help="Florence generation num_beams.")
@click.option("--use-slow-processor", is_flag=True, help="Force the legacy slow processor.")
@click.option("--detection-skip", default=1, type=int, help="Detect watermarks every N frames for videos (1-10). Higher = faster but may miss brief watermarks.")
@click.option("--fade-in", default=0.0, type=float, help="Extend mask backwards by N seconds to handle fade-in watermarks.")
@click.option("--fade-out", default=0.0, type=float, help="Extend mask forwards by N seconds to handle fade-out watermarks.")
def cli_entry(input_path: str, output_path: str, preview: bool, overwrite: bool, transparent: bool, max_bbox_percent: float, force_format: str, detection_prompt: str, detection_classes: str, detection_output_label: str, detection_task: str, florence_model_id: str, florence_adapter_dir: str, florence_max_new_tokens: int, florence_num_beams: int, use_slow_processor: bool, detection_skip: int, fade_in: float, fade_out: float):
    # Input validation
    if detection_skip < 1 or detection_skip > 10:
        logger.warning(f"detection_skip must be 1-10, got {detection_skip}. Using 1.")
        detection_skip = max(1, min(10, detection_skip))
    if fade_in < 0:
        fade_in = 0
    if fade_out < 0:
        fade_out = 0
    florence_max_new_tokens = max(1, int(florence_max_new_tokens or 256))
    florence_num_beams = max(1, int(florence_num_beams or 3))

    input_path = Path(input_path)
    raw_detection_classes = parse_raw_class_list(detection_classes)
    detection_output_label = detection_output_label or "watermark"
    task_type = choose_detector_task(detection_task, raw_detection_classes)

    # ========== PREVIEW MODE ==========
    if preview:
        import json
        import base64
        from io import BytesIO
        import random

        florence_model, florence_processor, device, _, load_mode, processor_mode = load_detector_stack(
            florence_model_id,
            florence_adapter_dir or None,
            use_fast_processor=not use_slow_processor,
        )
        logger.info(f"Florence-2 preview model loaded via {load_mode}, processor={processor_mode}")

        # Get sample image from input
        if input_path.is_dir():
            # Get a random image from directory
            images = list(input_path.glob("*.[jp][pn]g")) + list(input_path.glob("*.webp"))
            videos = list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi")) + list(input_path.glob("*.mov"))
            files = images + videos
            if not files:
                print(json.dumps({"error": "No supported files found in directory"}))
                return
            sample_path = random.choice(files)
        else:
            sample_path = input_path

        # Load image (extract frame if video)
        if is_supported_video(sample_path):
            cap = cv2.VideoCapture(str(sample_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Get frame from middle of video
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(json.dumps({"error": f"Could not read frame from video: {sample_path}"}))
                return
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            source_type = "video"
            source_frame = total_frames // 2
        else:
            pil_image = Image.open(sample_path).convert("RGB")
            source_type = "image"
            source_frame = None

        # Run detection from Florence raw generation output.
        detection_result = preview_detector_payload(pil_image, florence_model, florence_processor, device, max_bbox_percent, detection_prompt, raw_detection_classes, detection_output_label, detection_task, florence_max_new_tokens, florence_num_beams)
        detections = detection_result["detections"]

        # Draw bounding boxes on image
        draw = ImageDraw.Draw(pil_image)
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = (0, 255, 0) if det["accepted"] else (255, 0, 0)  # Green if accepted, red if rejected
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            # Draw label
            label = f"{det.get('raw_label', detection_prompt)} -> {det.get('output_label', detection_output_label)} {det['area_percent']:.1f}%"
            draw.text((x1, y1 - 15), label, fill=color)

        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Output JSON result
        result = {
            "image": img_base64,  # Just base64, GUI adds prefix
            "detections": detections,
            "source": str(sample_path),
            "source_type": source_type,
            "source_frame": source_frame,
            "prompt_used": task_type.value if task_type == SlayerVisionTask.OD else detection_prompt,
            "detection_task": task_type.value,
            "raw_classes": raw_detection_classes,
            "output_label": detection_output_label,
            "max_bbox_percent": max_bbox_percent,
            "florence_max_new_tokens": florence_max_new_tokens,
            "florence_num_beams": florence_num_beams,
            "generated_text": detection_result.get("generated_text", ""),
            "parsed": detection_result.get("parsed"),
            "predictions_raw": detection_result.get("predictions_raw", []),
            "raw_runs": detection_result.get("raw_runs", []),
        }
        print(json.dumps(result))
        return

    # ========== NORMAL PROCESSING MODE ==========
    output_path = Path(output_path)

    florence_model, florence_processor, device, _, load_mode, processor_mode = load_detector_stack(
        florence_model_id,
        florence_adapter_dir or None,
        use_fast_processor=not use_slow_processor,
    )
    print(f"Using device: {device}")
    logger.info(f"Florence-2 Model loaded via {load_mode}, processor={processor_mode}")

    if not transparent:
        model_manager = load_inpaint_engine(str(device))
        logger.info("LaMa model loaded")
    else:
        model_manager = None

    if input_path.is_dir():
        if not output_path.exists():
            output_path.mkdir(parents=True)

        # Include video files in the search
        images = list(input_path.glob("*.[jp][pn]g")) + list(input_path.glob("*.webp"))
        videos = list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi")) + list(input_path.glob("*.mov")) + list(input_path.glob("*.mkv"))
        files = images + videos
        total_files = len(files)

        for idx, file_path in enumerate(tqdm.tqdm(files, desc="Processing files")):
            output_file = output_path / file_path.name
            # Calculate progress range for this file
            progress_offset = int(idx / total_files * 100)
            progress_scale = int(100 / total_files)
            process_single_asset(file_path, output_file, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite, detection_prompt, raw_detection_classes, detection_output_label, detection_task, detection_skip, fade_in, fade_out, florence_max_new_tokens, florence_num_beams, progress_offset, progress_scale)
    else:
        # Single file mode - if output is a directory, construct file path
        if output_path.is_dir() or output_path.suffix == "":
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / input_path.name
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = output_path

        # Ensure video output has proper extension
        if is_supported_video(input_path) and output_file.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv']:
            if force_format and force_format.upper() in ["MP4", "AVI"]:
                output_file = output_file.with_suffix(f".{force_format.lower()}")
            else:
                output_file = output_file.with_suffix(".mp4")  # Default to mp4

        result_path = process_single_asset(input_path, output_file, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite, detection_prompt, raw_detection_classes, detection_output_label, detection_task, detection_skip, fade_in, fade_out, florence_max_new_tokens, florence_num_beams)
        final_output = result_path or output_file
        print(f"input_path:{input_path}, output_path:{final_output}, overall_progress:100")

if __name__ == "__main__":
    cli_entry()
