"""
Watermark Slayer GUI.
PyWebview frontend for the local processing workflow.
"""

import logging

# Suppress noisy pywebview WebView2 COM warnings (thread safety noise, doesn't affect functionality)
class WebviewNoiseFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        # Filter out WebView2 COM interface errors that spam the console
        if 'Error while processing window.native' in msg:
            return False
        if 'CoreWebView2 members can only be accessed' in msg:
            return False
        return True

logging.getLogger('pywebview').addFilter(WebviewNoiseFilter())

import webview
import threading
import subprocess
import sys
import os
import json
import yaml
import base64
import mimetypes
import re
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from PIL import Image, ImageOps

# Only psutil for system info (lightweight)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.yml")
DEFAULT_FLORENCE_MODEL_ID = "/home/h3c/cbh_ws/florence_train/fused_model"
DEFAULT_FLORENCE_ADAPTER_DIR = ""
DEFAULT_FLORENCE_MAX_NEW_TOKENS = 256
DEFAULT_FLORENCE_NUM_BEAMS = 3
DEFAULT_OUTPUT_PATH = "/home/h3c/cbh_ws/water_marked/outputs"
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
MEDIA_PREVIEW_MAX_SIDE = 1600
LIVE_FRAME_PREFIX = "WM_SLAYER_LIVE_FRAME:"


def normalize_runtime_path(path):
    """Normalize user-facing paths to Ubuntu/WSL style slash paths."""
    if path is None:
        return ''

    value = str(path).strip()
    if not value:
        return ''

    parsed = urlparse(value)
    if parsed.scheme == 'file':
        if parsed.netloc and parsed.netloc not in ('localhost', ''):
            value = f"//{parsed.netloc}{unquote(parsed.path)}"
        else:
            value = unquote(parsed.path)

    value = value.replace('\\', '/')
    drive_match = re.match(r'^/?([A-Za-z]):/(.*)$', value)
    if drive_match:
        drive, rest = drive_match.groups()
        value = f"/mnt/{drive.lower()}/{rest}"

    return value


def normalize_config_paths(config):
    """Keep path-like values in ui.yml portable for the Ubuntu runtime."""
    if not isinstance(config, dict):
        return config

    normalized = dict(config)
    for key in (
        'input_path',
        'output_path',
        'florence_model_id',
        'florence_adapter_dir',
    ):
        if key in normalized and normalized[key] is not None:
            normalized[key] = normalize_runtime_path(normalized[key])
    return normalized


class SlayerBridge:
    """Python API exposed to JavaScript frontend"""

    def __init__(self):
        self._frontend_window = None
        self._slayer_process = None
        self._job_active = False
        self._ui_state = self._read_ui_state()
        self._media_routes = {}
        self._media_tokens = {}
        self._media_server = None
        self._media_server_thread = None
        self._media_server_port = None

    def attach_window(self, window):
        """Set the webview window reference"""
        self._frontend_window = window

    def _read_ui_state(self):
        """Load saved configuration from YAML file"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return normalize_config_paths(yaml.safe_load(f) or {})
            except Exception:
                pass
        return {}

    def _write_ui_state(self, config):
        """Save configuration to YAML file"""
        try:
            config = normalize_config_paths(config)
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def log_frontend_message(self, msg):
        """Print debug message from JavaScript"""
        print(f"[JS DEBUG] {msg}")

    def fetch_ui_state(self):
        """Return saved configuration to frontend"""
        self._ui_state = normalize_config_paths(self._ui_state)
        return self._ui_state

    def store_ui_state(self, config):
        """Save configuration from frontend"""
        self._ui_state = normalize_config_paths(config)
        self._write_ui_state(self._ui_state)

    def choose_media_file(self):
        """Open file browser dialog"""
        if not self._frontend_window:
            return None

        file_types = (
            'All supported files (*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.mp4;*.avi;*.mov;*.mkv;*.flv;*.wmv;*.webm)',
            'Images (*.png;*.jpg;*.jpeg;*.webp;*.bmp)',
            'Videos (*.mp4;*.avi;*.mov;*.mkv;*.flv;*.wmv;*.webm)',
            'All files (*.*)'
        )

        result = self._frontend_window.create_file_dialog(
            webview.FileDialog.OPEN,
            file_types=file_types
        )
        return normalize_runtime_path(result[0]) if result else None

    def choose_directory(self):
        """Open folder browser dialog"""
        if not self._frontend_window:
            return None

        result = self._frontend_window.create_file_dialog(webview.FileDialog.FOLDER)
        return normalize_runtime_path(result[0]) if result else None

    def _ensure_media_server(self):
        """Start a local read-only HTTP server for video playback."""
        if self._media_server:
            return

        routes = self._media_routes

        class MediaHandler(BaseHTTPRequestHandler):
            server_version = "WatermarkSlayerMedia/1.0"

            def log_message(self, *_args):
                return

            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Range')
                self.end_headers()

            def do_HEAD(self):
                self._serve_media(send_body=False)

            def do_GET(self):
                self._serve_media(send_body=True)

            def _serve_media(self, send_body=True):
                parsed = urlparse(self.path)
                parts = parsed.path.strip('/').split('/')
                if len(parts) < 2 or parts[0] != 'media':
                    self.send_error(404)
                    return

                media_path = routes.get(parts[1])
                if not media_path:
                    self.send_error(404)
                    return

                path = Path(media_path)
                if not path.exists() or not path.is_file():
                    self.send_error(404)
                    return

                file_size = path.stat().st_size
                start = 0
                end = file_size - 1
                status = 200

                range_header = self.headers.get('Range', '')
                range_match = re.match(r'bytes=(\d*)-(\d*)', range_header)
                if range_match:
                    raw_start, raw_end = range_match.groups()
                    if raw_start:
                        start = int(raw_start)
                    if raw_end:
                        end = int(raw_end)
                    end = min(end, file_size - 1)
                    if start > end or start >= file_size:
                        self.send_response(416)
                        self.send_header('Content-Range', f'bytes */{file_size}')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        return
                    status = 206

                content_length = end - start + 1
                mime = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
                self.send_response(status)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(content_length))
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Cache-Control', 'no-store')
                if status == 206:
                    self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                self.end_headers()

                if not send_body:
                    return

                with path.open('rb') as media_file:
                    media_file.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk = media_file.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        try:
                            self.wfile.write(chunk)
                        except (BrokenPipeError, ConnectionResetError):
                            break
                        remaining -= len(chunk)

        self._media_server = ThreadingHTTPServer(('127.0.0.1', 0), MediaHandler)
        self._media_server.daemon_threads = True
        self._media_server_port = self._media_server.server_address[1]
        self._media_server_thread = threading.Thread(
            target=self._media_server.serve_forever,
            daemon=True
        )
        self._media_server_thread.start()

    def _media_server_url(self, media_path):
        """Return a browser-safe local URL for a media file."""
        self._ensure_media_server()
        resolved = str(Path(media_path).resolve())
        token = self._media_tokens.get(resolved)
        if not token:
            token = secrets.token_urlsafe(16)
            self._media_tokens[resolved] = token
            self._media_routes[token] = resolved

        name = quote(Path(media_path).name)
        return f"http://127.0.0.1:{self._media_server_port}/media/{token}/{name}"

    def _video_poster_payload(self, media_path):
        """Extract a compact first-frame poster for video fallback display."""
        try:
            import cv2

            cap = cv2.VideoCapture(str(media_path))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return {}

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((MEDIA_PREVIEW_MAX_SIDE, MEDIA_PREVIEW_MAX_SIDE))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=86, optimize=True)
            data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return {
                'poster_mime': 'image/jpeg',
                'poster_data': data,
                'poster_data_url': f"data:image/jpeg;base64,{data}",
            }
        except Exception:
            return {}

    def _build_media_payload(self, path, include_data=True):
        """Return a browser-displayable payload for an image or video path."""
        if not path:
            return {'error': 'No media path specified'}

        path = normalize_runtime_path(path)
        media_path = Path(path).expanduser()
        if not media_path.exists():
            return {'error': f'Media path does not exist: {path}'}

        suffix = media_path.suffix.lower()
        payload = {
            'path': normalize_runtime_path(media_path.resolve()),
            'name': media_path.name,
            'suffix': suffix,
        }

        if suffix in IMAGE_EXTENSIONS:
            mime = 'image/png'
            payload.update({
                'kind': 'image',
                'mime': mime,
                'url': media_path.resolve().as_uri(),
            })
            if include_data:
                with Image.open(media_path) as image:
                    image = ImageOps.exif_transpose(image)
                    if image.mode not in ("RGB", "RGBA"):
                        image = image.convert("RGB")
                    image.thumbnail((MEDIA_PREVIEW_MAX_SIDE, MEDIA_PREVIEW_MAX_SIDE))
                    buffer = BytesIO()
                    image.save(buffer, format="PNG")
                data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                payload['data'] = data
                payload['data_url'] = f"data:{mime};base64,{data}"
            return payload

        if suffix in VIDEO_EXTENSIONS:
            payload.update({
                'kind': 'video',
                'mime': mimetypes.guess_type(str(media_path))[0] or 'video/mp4',
                'url': self._media_server_url(media_path),
            })
            if include_data:
                payload.update(self._video_poster_payload(media_path))
            return payload

        return {'error': f'Unsupported media type: {path}'}

    def fetch_media_payload(self, path):
        """Return image base64 or video file URL for comparison UI."""
        try:
            return self._build_media_payload(path)
        except Exception as e:
            return {'error': str(e)}

    def _would_clobber_source(self, input_path, output_path):
        """Check if output would overwrite the input file."""
        supported_ext = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

        if os.path.isfile(input_path):
            # Single file mode
            output_ext = os.path.splitext(output_path)[1].lower()
            is_output_dir = os.path.isdir(output_path) or (output_ext == '' or output_ext not in supported_ext)

            if is_output_dir:
                output_file = os.path.join(output_path, os.path.basename(input_path))
            else:
                output_file = output_path
            # Compare resolved paths
            return os.path.normcase(os.path.abspath(input_path)) == os.path.normcase(os.path.abspath(output_file))
        else:
            # Directory mode - check if input and output folders are the same
            return os.path.normcase(os.path.abspath(input_path)) == os.path.normcase(os.path.abspath(output_path))

    def _find_output_conflicts(self, input_path, output_path):
        """Check if output files already exist. Returns list of conflicting filenames."""
        conflicts = []
        supported_ext = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

        if os.path.isfile(input_path):
            # Single file mode
            input_name = os.path.basename(input_path)
            # Check if output_path is an existing directory OR looks like a directory path (no file extension)
            output_ext = os.path.splitext(output_path)[1].lower()
            is_output_dir = os.path.isdir(output_path) or (output_ext == '' or output_ext not in supported_ext)

            if is_output_dir:
                output_file = os.path.join(output_path, input_name)
            else:
                output_file = output_path

            # Check for file with same name OR alternate extension (jpg<->jpeg)
            files_to_check = [output_file]
            base, ext = os.path.splitext(output_file)
            if ext.lower() == '.jpg':
                files_to_check.append(base + '.jpeg')
            elif ext.lower() == '.jpeg':
                files_to_check.append(base + '.jpg')

            for check_file in files_to_check:
                if os.path.exists(check_file):
                    conflicts.append(os.path.basename(check_file))
                    break
        else:
            # Directory/batch mode
            if os.path.isdir(input_path):
                for fname in os.listdir(input_path):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in supported_ext:
                        output_file = os.path.join(output_path, fname)
                        if os.path.exists(output_file):
                            conflicts.append(fname)

        return conflicts

    def fetch_capability_snapshot(self):
        """Get static system info (CUDA, FFmpeg, GPU) - call once on startup"""
        info = {
            'cuda': False,
            'gpu_name': None,
            'ffmpeg': False
        }

        # Windows: hide console windows for subprocesses
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0

        # Check CUDA via subprocess (avoid importing torch in GUI)
        try:
            result = subprocess.run(
                [sys.executable, '-c', 'import torch; print("CUDA:" + str(torch.cuda.is_available()) + ":" + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""))'],
                capture_output=True, text=True, timeout=10, creationflags=creationflags
            )
            if result.returncode == 0 and 'CUDA:' in result.stdout:
                parts = result.stdout.strip().split(':')
                info['cuda'] = parts[1] == 'True'
                if len(parts) > 2 and parts[2]:
                    info['gpu_name'] = parts[2]
        except Exception:
            pass

        # Check FFmpeg
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, creationflags=creationflags)
            info['ffmpeg'] = True
        except (subprocess.SubprocessError, FileNotFoundError):
            info['ffmpeg'] = False

        return info

    def fetch_usage_snapshot(self):
        """Get dynamic system info (RAM, CPU) - call periodically"""
        info = {
            'ram_percent': 0,
            'cpu_percent': 0
        }

        if PSUTIL_AVAILABLE:
            try:
                info['ram_percent'] = psutil.virtual_memory().percent
                info['cpu_percent'] = psutil.cpu_percent()
            except Exception:
                pass

        return info

    def launch_slayer_job(self, settings):
        """Start watermark removal processing"""
        if self._job_active:
            return {'error': 'Already running'}

        input_path = normalize_runtime_path(settings.get('input', ''))
        output_path = normalize_runtime_path(settings.get('output', '')) or DEFAULT_OUTPUT_PATH

        if not input_path:
            return {'error': 'No input path specified'}

        # Use input directory as output if not specified
        if not output_path:
            output_path = DEFAULT_OUTPUT_PATH

        # SAFETY: Check if output would overwrite input
        overwrite = settings.get('overwrite', False)
        would_overwrite_input = self._would_clobber_source(input_path, output_path)
        if would_overwrite_input:
            return {'error': 'Cannot overwrite input file! Choose a different output folder.'}

        # Check for file conflicts if overwrite is not enabled
        if not overwrite:
            conflicts = self._find_output_conflicts(input_path, output_path)
            if conflicts:
                conflict_list = ', '.join(conflicts[:3])
                more = f" (+{len(conflicts)-3} more)" if len(conflicts) > 3 else ""
                error_msg = f'Output files already exist: {conflict_list}{more}. Enable "Overwrite" or choose different output folder.'
                return {'error': error_msg}

        # Get settings
        detection_prompt = settings.get('detection_prompt', 'watermark')
        detection_classes = settings.get('detection_classes', [])
        if isinstance(detection_classes, str):
            detection_classes_arg = detection_classes
        else:
            detection_classes_arg = ','.join([str(item).strip() for item in detection_classes if str(item).strip()])
        detection_output_label = settings.get('detection_output_label', 'watermark')
        detection_task = settings.get('detection_task', 'auto')
        florence_model_id = normalize_runtime_path(settings.get('florence_model_id', DEFAULT_FLORENCE_MODEL_ID))
        florence_adapter_dir = normalize_runtime_path(settings.get('florence_adapter_dir', DEFAULT_FLORENCE_ADAPTER_DIR))
        florence_max_new_tokens = int(settings.get('florence_max_new_tokens', DEFAULT_FLORENCE_MAX_NEW_TOKENS) or DEFAULT_FLORENCE_MAX_NEW_TOKENS)
        florence_num_beams = int(settings.get('florence_num_beams', DEFAULT_FLORENCE_NUM_BEAMS) or DEFAULT_FLORENCE_NUM_BEAMS)
        florence_use_fast_processor = bool(settings.get('florence_use_fast_processor', True))
        detection_skip = settings.get('detection_skip', 1)
        fade_in = settings.get('fade_in', 0)
        fade_out = settings.get('fade_out', 0)

        # Save config
        self.store_ui_state({
            'input_path': input_path,
            'output_path': output_path,
            'overwrite': settings.get('overwrite', False),
            'transparent': settings.get('transparent', False),
            'max_bbox_percent': settings.get('max_bbox', 100),
            'force_format': settings.get('format', 'None'),
            'mode': settings.get('mode', 'single'),
            'detection_prompt': detection_prompt,
            'detection_group': settings.get('detection_group', 'watermark'),
            'detection_classes': detection_classes_arg.split(',') if detection_classes_arg else [],
            'detection_output_label': detection_output_label,
            'detection_task': detection_task,
            'florence_model_id': florence_model_id,
            'florence_adapter_dir': florence_adapter_dir,
            'florence_max_new_tokens': florence_max_new_tokens,
            'florence_num_beams': florence_num_beams,
            'florence_use_fast_processor': florence_use_fast_processor,
            'detection_skip': detection_skip,
            'fade_in': fade_in,
            'fade_out': fade_out,
            'theme': settings.get('theme', 'dark'),
            'lang': settings.get('lang', 'zh')
        })

        # Build command
        cmd = [sys.executable, 'watermark_slayer.py', input_path, output_path]

        if settings.get('overwrite'):
            cmd.append('--overwrite')

        if settings.get('transparent'):
            cmd.append('--transparent')

        max_bbox = settings.get('max_bbox', 100)
        cmd.append(f'--max-bbox-percent={int(max_bbox)}')

        format_opt = settings.get('format', 'None')
        if format_opt and format_opt != 'None':
            cmd.append(f'--force-format={format_opt}')

        if detection_prompt and detection_prompt != 'watermark':
            cmd.append(f'--detection-prompt={detection_prompt}')

        if detection_classes_arg:
            cmd.append(f'--detection-classes={detection_classes_arg}')
        cmd.append(f'--detection-output-label={detection_output_label or "watermark"}')
        cmd.append(f'--detection-task={detection_task or "auto"}')

        if florence_model_id:
            cmd.append(f'--florence-model-id={florence_model_id}')
        cmd.append(f'--florence-adapter-dir={florence_adapter_dir or ""}')
        cmd.append(f'--florence-max-new-tokens={florence_max_new_tokens}')
        cmd.append(f'--florence-num-beams={florence_num_beams}')
        if not florence_use_fast_processor:
            cmd.append('--use-slow-processor')

        if detection_skip and int(detection_skip) > 1:
            cmd.append(f'--detection-skip={int(detection_skip)}')

        if fade_in and float(fade_in) > 0:
            cmd.append(f'--fade-in={float(fade_in)}')

        if fade_out and float(fade_out) > 0:
            cmd.append(f'--fade-out={float(fade_out)}')

        # Start processing in background thread
        self._job_active = True
        threading.Thread(target=self._stream_worker, args=(cmd,), daemon=True).start()
        return {'status': 'started'}

    def _stream_worker(self, cmd):
        """Run the subprocess and stream output to frontend"""
        try:
            # Log the CLI command for educational purposes
            cli_display = ' '.join(cmd[1:])  # Skip python executable
            cli_display = cli_display.replace('watermark_slayer.py ', 'python watermark_slayer.py \\\n    ')
            cli_display = cli_display.replace(' --', ' \\\n    --')
            self._send_frontend_event(f'addLog("$ {json.dumps(cli_display)[1:-1]}", "text-info")')

            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            working_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(working_dir, 'watermark_slayer.py')

            # Verify script exists
            if not os.path.exists(script_path):
                self._send_frontend_event(f'addLog("ERROR: watermark_slayer.py not found at {json.dumps(script_path)}", "text-error")')
                self._send_frontend_event('processingComplete()')
                return

            self._slayer_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=working_dir
            )

            for line in iter(self._slayer_process.stdout.readline, ''):
                if not self._job_active:
                    break

                line = line.strip()
                if not line:
                    continue

                if line.startswith(LIVE_FRAME_PREFIX):
                    try:
                        live_frame = json.loads(line[len(LIVE_FRAME_PREFIX):])
                        self._send_frontend_event(f'processingFrame({json.dumps(live_frame)})')
                    except json.JSONDecodeError:
                        pass
                    continue

                # Parse progress
                if 'overall_progress:' in line:
                    try:
                        progress_str = line.split('overall_progress:')[1].strip()
                        progress = int(progress_str.replace('%', ''))
                        self._send_frontend_event(f'updateProgress({progress})')
                    except (ValueError, IndexError):
                        pass

                output_match = re.search(r'output_path:([^,\r\n]+)', line)
                if output_match:
                    output_path = normalize_runtime_path(output_match.group(1).strip())
                    self._send_frontend_event(f'processingOutputPath({json.dumps(output_path)})')

                # Send log line to frontend
                escaped = json.dumps(line)

                if 'error' in line.lower() or 'failed' in line.lower():
                    color = 'text-error'
                elif 'warning' in line.lower():
                    color = 'text-yellow-400'
                elif 'success' in line.lower() or 'done' in line.lower() or 'saved' in line.lower():
                    color = 'text-success'
                else:
                    color = 'text-gray-400'

                self._send_frontend_event(f'addLog({escaped}, "{color}")')

            self._slayer_process.wait()
            self._send_frontend_event('processingComplete()')

        except Exception as e:
            import traceback
            error_msg = json.dumps(f"Error: {str(e)}")
            self._send_frontend_event(f'addLog({error_msg}, "text-error")')
            # Log full traceback for debugging
            tb = json.dumps(traceback.format_exc())
            self._send_frontend_event(f'addLog({tb}, "text-gray-500")')
            self._send_frontend_event('processingComplete()')

        finally:
            self._job_active = False
            self._slayer_process = None

    def _send_frontend_event(self, js_code):
        """Safely call JavaScript in the frontend"""
        if self._frontend_window:
            try:
                self._frontend_window.evaluate_js(js_code)
            except Exception:
                pass

    def halt_slayer_job(self):
        """Stop the current processing"""
        self._job_active = False

        if self._slayer_process:
            try:
                self._slayer_process.terminate()
                try:
                    self._slayer_process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self._slayer_process.kill()
            except Exception:
                pass

        return {'status': 'stopped'}

    def preview_slayer_detection(self, settings):
        """
        Preview watermark detection via CLI subprocess.
        Returns image with bounding boxes drawn as base64.
        """
        input_path = normalize_runtime_path(settings.get('input', ''))
        detection_prompt = settings.get('detection_prompt', 'watermark')
        detection_classes = settings.get('detection_classes', [])
        if isinstance(detection_classes, str):
            detection_classes_arg = detection_classes
        else:
            detection_classes_arg = ','.join([str(item).strip() for item in detection_classes if str(item).strip()])
        detection_output_label = settings.get('detection_output_label', 'watermark')
        detection_task = settings.get('detection_task', 'auto')
        florence_model_id = normalize_runtime_path(settings.get('florence_model_id', DEFAULT_FLORENCE_MODEL_ID))
        florence_adapter_dir = normalize_runtime_path(settings.get('florence_adapter_dir', DEFAULT_FLORENCE_ADAPTER_DIR))
        florence_max_new_tokens = int(settings.get('florence_max_new_tokens', DEFAULT_FLORENCE_MAX_NEW_TOKENS) or DEFAULT_FLORENCE_MAX_NEW_TOKENS)
        florence_num_beams = int(settings.get('florence_num_beams', DEFAULT_FLORENCE_NUM_BEAMS) or DEFAULT_FLORENCE_NUM_BEAMS)
        florence_use_fast_processor = bool(settings.get('florence_use_fast_processor', True))
        max_bbox = settings.get('max_bbox', 100)

        if not input_path:
            return {'error': 'No input path specified'}

        try:
            # Call CLI with --preview flag
            cmd = [
                sys.executable, 'watermark_slayer.py',
                input_path, '--preview',
                '--max-bbox-percent', str(int(max_bbox)),
                '--detection-prompt', detection_prompt
            ]
            if detection_classes_arg:
                cmd.extend(['--detection-classes', detection_classes_arg])
            cmd.extend(['--detection-output-label', detection_output_label or 'watermark'])
            cmd.extend(['--detection-task', detection_task or 'auto'])
            if florence_model_id:
                cmd.extend(['--florence-model-id', florence_model_id])
            cmd.extend(['--florence-adapter-dir', florence_adapter_dir or ''])
            cmd.extend(['--florence-max-new-tokens', str(florence_max_new_tokens)])
            cmd.extend(['--florence-num-beams', str(florence_num_beams)])
            if not florence_use_fast_processor:
                cmd.append('--use-slow-processor')

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )

            if result.returncode != 0:
                return {'error': result.stderr or 'Preview failed'}

            # Parse JSON output from CLI
            output = result.stdout.strip()
            # Find JSON in output (may have log lines before it)
            for line in output.split('\n'):
                if line.startswith('{'):
                    return json.loads(line)

            return {'error': 'No preview data returned'}

        except subprocess.TimeoutExpired:
            return {'error': 'Preview timed out'}
        except Exception as e:
            return {'error': str(e)}

def launch_gui():
    """Main entry point"""
    bridge = SlayerBridge()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ui_path = os.path.join(script_dir, 'ui', 'index.html')

    window = webview.create_window(
        'Watermark Slayer',
        ui_path,
        js_api=bridge,
        width=950,
        height=860,
        min_size=(800, 600),
        background_color='#050505'
    )

    bridge.attach_window(window)
    webview.start()


if __name__ == '__main__':
    launch_gui()
