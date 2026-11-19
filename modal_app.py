"""Modal upload API and worker for LTX-2.5 IC-LoRA video inpainting.

RunPod's handler.py and Dockerfile are unchanged. Model weights stay in the
separate ltx-model-cache Volume; website job inputs and outputs use
job-specific paths in ltx-job-files.
"""

from __future__ import annotations

import hmac
import base64
import json
import os
import re
import subprocess
import sys
import shutil
import secrets
import uuid
from fractions import Fraction
from urllib.parse import urlparse
from urllib.request import Request as URLRequest, urlopen
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import modal


APP_NAME = "ltx-inpaint-modal"
HF_SECRET_NAME = os.environ.get("MODAL_HF_SECRET", "huggingface-secret")
UPLOAD_SECRET_NAME = os.environ.get("MODAL_UPLOAD_SECRET", "ltx-web-upload-auth")
MODEL_VOLUME_NAME = "ltx-model-cache"
JOB_VOLUME_NAME = os.environ.get("MODAL_JOB_VOLUME", "ltx-job-files-v2")
GPU_TYPE = os.environ.get("MODAL_GPU", "RTX-PRO-6000")
JOB_ROOT = Path("/job-files")
MODEL_ROOT = Path("/models")

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)
job_volume = modal.Volume.from_name(JOB_VOLUME_NAME)
image = modal.Image.from_dockerfile("Dockerfile.modal")

REQUIRED_MODEL_FILES = (
    "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "loras/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
)

JOB_RETENTION_DAYS = 30
JOB_CLEANUP_SCHEDULE_UTC = "0 4 * * *"


def _job_file(value: str, field: str, job_id: str, *, must_exist: bool) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path inside {JOB_ROOT}")

    path = Path(value.strip())
    if not path.is_absolute():
        path = JOB_ROOT / path
    # Modal exposes a mounted Volume through an internal symlink. Resolve
    # relative path segments without following that mount symlink, or valid
    # /job-files paths would appear outside JOB_ROOT.
    path = Path(os.path.abspath(path))
    job_input_root = JOB_ROOT / "jobs" / job_id / "input"
    try:
        path.relative_to(job_input_root)
    except ValueError as exc:
        raise ValueError(f"{field} must be inside {job_input_root}") from exc
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"{field} does not exist in the Modal job volume: {path}")
    return path


def _ensure_models() -> None:
    missing = [rel for rel in REQUIRED_MODEL_FILES if not (MODEL_ROOT / rel).is_file()]
    if not missing:
        print(f"All required model files are present in {MODEL_ROOT}.", flush=True)
        return

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            f"HF_TOKEN is missing. Check that Modal Secret '{HF_SECRET_NAME}' "
            "contains a key named HF_TOKEN."
        )

    print(
        f"Downloading {len(missing)} missing model file(s) into Modal Volume "
        f"'{MODEL_VOLUME_NAME}' at {MODEL_ROOT}.",
        flush=True,
    )
    result = subprocess.run(
        [sys.executable, "/app/setup_and_download.py", "--download-only", "--yes"],
        cwd="/app",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Model download failed with exit code {result.returncode}.")

    missing = [rel for rel in REQUIRED_MODEL_FILES if not (MODEL_ROOT / rel).is_file()]
    if missing:
        raise RuntimeError("Model download did not produce: " + ", ".join(missing))

    # Commit before inference so this complete model snapshot is available to
    # the next Modal container, even if the job later fails during generation.
    model_volume.commit()



def _render_static_mask_video(source_path: Path, png_path: Path, output_path: Path) -> tuple[int, int, int, float]:
    """Repeat a static white-replace/black-preserve PNG for every source frame."""
    import av
    from PIL import Image

    source = av.open(str(source_path))
    encoder = None
    frame_count = 0
    completed = False
    try:
        source_stream = source.streams.video[0]
        fps = source_stream.average_rate or Fraction(24, 1)
        if float(fps) <= 0:
            fps = Fraction(24, 1)
        if source_stream.width < 64 or source_stream.height < 64 or source_stream.width * source_stream.height > 40_000_000:
            raise ValueError("Video dimensions are outside the supported range.")
        max_frames = int(float(fps) * 60) + 2
        with Image.open(png_path) as opened_mask:
            mask = opened_mask.convert("L")
        if mask.size != (source_stream.width, source_stream.height):
            raise ValueError(
                f"Mask dimensions {mask.width}x{mask.height} do not match video "
                f"{source_stream.width}x{source_stream.height}."
            )
        mask = mask.point(lambda value: 255 if value >= 128 else 0).convert("RGB")
        output_path.unlink(missing_ok=True)
        encoder = av.open(str(output_path), mode="w")
        mask_stream = encoder.add_stream("libx264", rate=fps)
        mask_stream.width = source_stream.width
        mask_stream.height = source_stream.height
        mask_stream.pix_fmt = "yuv420p" if source_stream.width % 2 == 0 and source_stream.height % 2 == 0 else "yuv444p"
        mask_stream.options = {"crf": "0", "preset": "ultrafast"}
        for _ in source.decode(video=0):
            if frame_count >= max_frames:
                raise ValueError("Videos longer than 60 seconds are not supported.")
            frame = av.VideoFrame.from_image(mask).reformat(format=mask_stream.pix_fmt)
            frame.pts = frame_count
            frame.time_base = Fraction(fps.denominator, fps.numerator)
            for packet in mask_stream.encode(frame):
                encoder.mux(packet)
            frame_count += 1
        for packet in mask_stream.encode():
            encoder.mux(packet)
        if frame_count == 0:
            raise ValueError("Source video contains no frames.")
        completed = True
        return source_stream.width, source_stream.height, frame_count, float(fps)
    finally:
        if encoder is not None:
            encoder.close()
        source.close()
        if not completed:
            output_path.unlink(missing_ok=True)

@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=12.0,
    memory=16384,
    timeout=60 * 60,
    scaledown_window=2,
    # Allow five independent video jobs to run in parallel. Additional inputs
    # sent to this same deployed Function wait in Modal's input queue.
    max_containers=5,
    volumes={
        str(MODEL_ROOT): model_volume,
        str(JOB_ROOT): job_volume,
    },
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def process_job(
    job_id: str,
    video_path: str,
    mask_path: str,
    mode: str = "standard",
    prompt: str | None = None,
    seed: int | None = None,
) -> dict[str, str]:
    """Inpaint a job's uploaded video and mask from the Modal job-files Volume."""
    import torch

    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", job_id):
        raise ValueError("job_id may contain only letters, numbers, underscores, and hyphens")
    if mode not in {"low-vram", "standard"}:
        raise ValueError("mode must be 'low-vram' or 'standard'")
    # A deployed worker can be reused after a web upload has committed new
    # files to the shared job Volume. Reload so this container sees them.
    job_volume.reload()
    job_manifest_path = JOB_ROOT / "jobs" / job_id / "job.json"
    if prompt is None:
        try:
            job_manifest = json.loads(job_manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            job_manifest = {}
        prompt = job_manifest.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A non-empty user prompt is required for every job.")
    prompt = prompt.strip()
    if len(prompt) > 4096:
        raise ValueError("prompt must be 4096 characters or fewer")
    source = _job_file(video_path, "video_path", job_id, must_exist=True)
    mask = _job_file(mask_path, "mask_path", job_id, must_exist=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Configure this Modal function with a CUDA GPU.")
    gpu = torch.cuda.get_device_properties(0)
    print(
        f"Using CUDA GPU {gpu.name} with {gpu.total_memory / (1024**3):.1f} GiB VRAM.",
        flush=True,
    )

    def update_manifest(**updates: object) -> None:
        try:
            manifest = json.loads(job_manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            manifest = {"job_id": job_id, "prompt": prompt}
        manifest.update(updates)
        job_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    output = JOB_ROOT / "jobs" / job_id / "output" / "inpainted.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    update_manifest(status="running", progress=10, started_at=datetime.now(timezone.utc).isoformat())
    job_volume.commit()
    try:
        if mask.suffix.lower() == ".png":
            mask_video_path = mask.with_name("mask_video.mp4")
            width, height, frame_count, fps = _render_static_mask_video(source, mask, mask_video_path)
            manifest = json.loads(job_manifest_path.read_text(encoding="utf-8"))
            manifest["input_paths"]["mask_video"] = str(mask_video_path)
            manifest["video_info"] = {
                "width": width,
                "height": height,
                "frames": frame_count,
                "fps": fps,
                "duration_seconds": frame_count / fps,
            }
            job_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            job_volume.commit()
            mask = mask_video_path
        _ensure_models()
        command = [
            sys.executable,
            "/app/run_inpaint.py",
            "--source", str(source),
            "--mask", str(mask),
            "--output", str(output),
            "--use-conv-vae",
        ]
        if mode == "low-vram":
            command.append("--low-vram")
        command.extend(["--prompt", prompt])
        if seed is not None:
            command.extend(["--seed", str(int(seed))])

        print(f"Starting Modal inpaint job {job_id}: video={source}, mask={mask}", flush=True)
        print(f"Prompt received by Modal worker: {prompt}", flush=True)
        print(f"Mode: {mode}; convolutional VAE enabled.", flush=True)
        result = subprocess.run(command, cwd="/app", check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Inpainting failed with exit code {result.returncode}.")
        if not output.is_file():
            raise RuntimeError(f"Inpainting did not create the expected output: {output}")

        update_manifest(
            status="completed",
            progress=100,
            completed_at=datetime.now(timezone.utc).isoformat(),
            output_path=str(output),
        )
        job_volume.commit()
        return {"job_id": job_id, "output_path": str(output), "volume_name": JOB_VOLUME_NAME}
    except Exception as exc:
        update_manifest(status="failed", error=str(exc)[:500], failed_at=datetime.now(timezone.utc).isoformat())
        job_volume.commit()
        raise


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    cpu=1.0,
    memory=1024,
    timeout=10 * 60,
    schedule=modal.Cron(JOB_CLEANUP_SCHEDULE_UTC),
    volumes={str(JOB_ROOT): job_volume},
)
def cleanup_expired_jobs() -> dict[str, int]:
    """Remove terminal job folders older than 30 days from the job volume.

    Model weights live in a separate Volume, so this function only scans the
    direct children of /job-files/jobs. Queued and running jobs use their
    creation/start time for retention, so stale jobs do not accumulate.
    Incomplete upload folders without a manifest use their last modification
    time, so abandoned partial uploads eventually expire too.
    """
    # Reload first so this scheduled container evaluates the latest committed
    # job manifests, including jobs written by other Modal containers.
    job_volume.reload()
    jobs_root = JOB_ROOT / "jobs"
    if not jobs_root.is_dir():
        print(f"No job folder found at {jobs_root}; nothing to clean.", flush=True)
        return {"deleted_jobs": 0, "deleted_bytes": 0, "skipped_jobs": 0}

    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - JOB_RETENTION_DAYS * 24 * 60 * 60
    deleted_jobs = 0
    deleted_bytes = 0
    skipped_jobs = 0

    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            continue

        manifest_path = job_dir / "job.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                status = manifest.get("status")
                if status not in {"uploaded", "queued", "running", "completed", "failed"}:
                    skipped_jobs += 1
                    continue
                # Use the latest lifecycle start point: a recently running
                # job remains safe even if it sat queued for a long time.
                created_at = manifest.get("started_at") if status == "running" else manifest.get("created_at")
                if not isinstance(created_at, str):
                    skipped_jobs += 1
                    continue
                created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                modified_at = created.timestamp()
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                skipped_jobs += 1
                continue
        else:
            # A failed upload may leave a folder before its manifest is saved.
            # Base its retention on the directory's last modification time.
            try:
                modified_at = job_dir.stat().st_mtime
            except OSError:
                skipped_jobs += 1
                continue

        if modified_at >= cutoff:
            continue

        for file_path in job_dir.rglob("*"):
            if file_path.is_file():
                try:
                    deleted_bytes += file_path.stat().st_size
                except OSError:
                    pass
        shutil.rmtree(job_dir)
        deleted_jobs += 1
        print(f"Deleted expired job folder {job_dir.name}.", flush=True)

    if deleted_jobs:
        job_volume.commit()
    print(
        f"Job cleanup complete: deleted {deleted_jobs} job(s) / "
        f"{deleted_bytes} bytes; skipped {skipped_jobs}; "
        f"retention {JOB_RETENTION_DAYS} days.",
        flush=True,
    )
    return {
        "deleted_jobs": deleted_jobs,
        "deleted_bytes": deleted_bytes,
        "skipped_jobs": skipped_jobs,
    }

@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=60 * 60,
    max_containers=5,
    volumes={str(JOB_ROOT): job_volume},
    secrets=[modal.Secret.from_name(UPLOAD_SECRET_NAME)],
)
@modal.asgi_app()
def upload_api():
    """Authenticated website API that creates a job and stores its two inputs."""
    from fastapi import FastAPI, HTTPException, Request
    from starlette.datastructures import UploadFile

    api = FastAPI(title="LTX Inpaint Upload API")

    def safe_display_name(filename: str | None, fallback: str) -> str:
        # Keep client filenames only as display metadata; never use them in paths.
        name = PurePosixPath((filename or "").replace("\\", "/")).name
        name = "".join(char for char in name if char.isprintable() and char != "\x00")
        return name[:255] or fallback

    def safe_extension(filename: str | None) -> str:
        extension = PurePosixPath((filename or "").replace("\\", "/")).suffix.lower()
        if extension not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
            raise HTTPException(status_code=400, detail="Video and mask must use a supported video file extension")
        return extension

    def save_upload(upload: UploadFile, destination: Path) -> int:
        total = 0
        upload.file.seek(0)
        with destination.open("xb") as output_file:
            while True:
                chunk = upload.file.read(8 * 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > 2 * 1024 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="Each uploaded file must be 2 GiB or smaller")
                output_file.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="Uploaded files must not be empty")
        return total

    async def create_job(request):
        expected_token = os.environ.get("WEB_UPLOAD_TOKEN", "")
        if len(expected_token) < 32:
            raise HTTPException(
                status_code=503,
                detail="WEB_UPLOAD_TOKEN must be at least 32 characters; rotate the upload secret",
            )
        authorization = request.headers.get("authorization", "")
        scheme, separator, provided_token = authorization.partition(" ")
        if not expected_token:
            raise HTTPException(status_code=503, detail="Upload API token is not configured")
        if (
            not separator
            or scheme.lower() != "bearer"
            or not hmac.compare_digest(provided_token, expected_token)
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

        form = await request.form()
        video = form.get("video")
        mask = form.get("mask")
        prompt = form.get("prompt")
        if not isinstance(video, UploadFile) or not isinstance(mask, UploadFile):
            raise HTTPException(status_code=400, detail="Multipart fields 'video' and 'mask' are required")
        if not isinstance(prompt, str) or not prompt.strip():
            raise HTTPException(status_code=400, detail="A non-empty multipart field 'prompt' is required")
        prompt = prompt.strip()
        if len(prompt) > 4096:
            raise HTTPException(status_code=400, detail="Prompt must be 4096 characters or fewer")

        job_id = uuid.uuid4().hex
        job_dir = JOB_ROOT / "jobs" / job_id
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=False)
        try:
            video_extension = safe_extension(video.filename)
            mask_extension = safe_extension(mask.filename)
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
        video_path = input_dir / f"source_video{video_extension}"
        mask_path = input_dir / f"mask_video{mask_extension}"

        try:
            video_bytes = save_upload(video, video_path)
            mask_bytes = save_upload(mask, mask_path)
            output_path = job_dir / "output" / "inpainted.mp4"
            manifest = {
                "job_id": job_id,
                "status": "uploaded",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "prompt": prompt,
                "original_filenames": {
                    "video": safe_display_name(video.filename, "video"),
                    "mask": safe_display_name(mask.filename, "mask"),
                },
                "input_paths": {
                    "video": str(video_path),
                    "mask": str(mask_path),
                },
                "output_path": str(output_path),
                "file_sizes_bytes": {"video": video_bytes, "mask": mask_bytes},
            }
            manifest_path = job_dir / "job.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            job_volume.commit()
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            try:
                job_volume.commit()
            except Exception:
                pass
            raise
        finally:
            await video.close()
            await mask.close()

        return manifest

    # FastAPI needs the actual Request class in the annotation. With postponed
    # annotations, a locally imported Request would otherwise remain an
    # unresolved string and be interpreted as a required query parameter.
    create_job.__annotations__["request"] = Request
    api.post("/api/jobs", status_code=201)(create_job)

    @api.get("/health")
    async def health():
        return {"status": "ok"}

    return api


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=60 * 60,
    max_containers=5,
    volumes={str(JOB_ROOT): job_volume},
    secrets=[modal.Secret.from_name(UPLOAD_SECRET_NAME)],
)
@modal.asgi_app()
def object_remover_api():
    """Private job API for the Viresto static-mask editor."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse
    from starlette.middleware.cors import CORSMiddleware
    from starlette.datastructures import UploadFile
    from PIL import Image
    api = FastAPI(title="Viresto Object Remover Jobs")
    # The upload route is protected by a short-lived, job-scoped HMAC ticket.
    # No Modal service credential is ever sent to the browser.
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        allow_credentials=False,
    )
    max_video_bytes = 100 * 1024 * 1024
    max_mask_bytes = 12 * 1024 * 1024
    allowed_video_extensions = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

    def authorize(request: Request) -> None:
        expected = os.environ.get("WEB_UPLOAD_TOKEN", "")
        scheme, separator, provided = request.headers.get("authorization", "").partition(" ")
        if len(expected) < 32 or not separator or scheme.lower() != "bearer" or not hmac.compare_digest(provided, expected):
            raise HTTPException(status_code=401, detail="Invalid service authorization")

    def authorize_upload_ticket(request: Request, job_id: str) -> dict[str, object]:
        expected = os.environ.get("WEB_UPLOAD_TOKEN", "")
        scheme, separator, token = request.headers.get("authorization", "").partition(" ")
        if len(expected) < 32 or not separator or scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid upload authorization")
        payload_part, dot, signature_part = token.partition(".")
        if not dot or not payload_part or not signature_part:
            raise HTTPException(status_code=401, detail="Invalid upload authorization")
        signing_key = hmac.new(expected.encode(), b"object-remover-upload-ticket", "sha256").digest()
        signature = hmac.new(signing_key, payload_part.encode(), "sha256").digest()
        expected_signature = base64.urlsafe_b64encode(signature).decode().rstrip("=")
        if not hmac.compare_digest(signature_part, expected_signature):
            raise HTTPException(status_code=401, detail="Invalid upload authorization")
        try:
            padded = payload_part + "=" * (-len(payload_part) % 4)
            claims = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=401, detail="Invalid upload authorization") from exc
        if not isinstance(claims, dict) or (
            claims.get("scope") != "object-remover-upload"
            or claims.get("job_id") != job_id
            or int(claims.get("exp", 0)) <= int(datetime.now(timezone.utc).timestamp())
            or not isinstance(claims.get("user_id"), str)
            or not claims["user_id"]
        ):
            raise HTTPException(status_code=401, detail="Upload authorization expired or does not match this job")
        return claims

    def save_direct_upload(upload: UploadFile, destination: Path, max_bytes: int, expected_type: str) -> int:
        content_type = upload.content_type or "application/octet-stream"
        if expected_type == "video/":
            if not content_type.startswith("video/") and content_type != "application/octet-stream":
                raise HTTPException(status_code=415, detail="Source file must be a video")
        elif content_type != expected_type:
            raise HTTPException(status_code=415, detail=f"File must be {expected_type}")
        total = 0
        upload.file.seek(0)
        with destination.open("xb") as output_file:
            while True:
                chunk = upload.file.read(8 * 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="Uploaded file exceeds the size limit")
                output_file.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="Uploaded file must not be empty")
        return total

    async def upload_direct_job(job_id: str, request: Request):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise HTTPException(status_code=400, detail="Invalid job ID")
        claims = authorize_upload_ticket(request, job_id)
        video_name = PurePosixPath(str(claims.get("video_name", ""))).name
        extension = PurePosixPath(video_name).suffix.lower()
        if extension not in allowed_video_extensions:
            raise HTTPException(status_code=415, detail="Unsupported source video extension")
        try:
            video_size = int(claims.get("video_size", 0))
            width = int(claims.get("width", 0))
            height = int(claims.get("height", 0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=401, detail="Invalid upload authorization") from exc
        if not 0 < video_size <= max_video_bytes or not (64 <= width <= 8192 and 64 <= height <= 8192 and width * height <= 40_000_000):
            raise HTTPException(status_code=401, detail="Invalid upload authorization")
        content_length = request.headers.get("content-length")
        try:
            request_bytes = int(content_length or "")
        except ValueError as exc:
            raise HTTPException(status_code=411, detail="Content-Length is required for direct uploads") from exc
        if request_bytes > video_size + max_mask_bytes + 1024 * 1024:
            raise HTTPException(status_code=413, detail="Upload request exceeds the authorized size limit")

        await job_volume.reload.aio()
        job_dir = JOB_ROOT / "jobs" / job_id
        manifest_path = job_dir / "job.json"
        if manifest_path.exists():
            raise HTTPException(status_code=409, detail="This job already has uploaded files")

        form = await request.form(max_files=2, max_fields=0)
        if set(form.keys()) != {"video", "mask"} or len(form.getlist("video")) != 1 or len(form.getlist("mask")) != 1:
            raise HTTPException(status_code=400, detail="Upload must contain exactly one video and one mask")
        video = form.get("video")
        mask = form.get("mask")
        if not isinstance(video, UploadFile) or not isinstance(mask, UploadFile):
            raise HTTPException(status_code=400, detail="Multipart fields 'video' and 'mask' are required")
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=False)
        video_path = input_dir / f"source_video{extension}"
        mask_path = input_dir / "static_mask.png"
        try:
            actual_video_size = save_direct_upload(video, video_path, max_video_bytes, "video/")
            if actual_video_size != video_size:
                raise HTTPException(status_code=422, detail="Video size does not match the authorized upload")
            mask_bytes = save_direct_upload(mask, mask_path, max_mask_bytes, "image/png")
            try:
                with Image.open(mask_path) as opened_mask:
                    if opened_mask.format != "PNG" or opened_mask.size != (width, height):
                        raise HTTPException(status_code=422, detail="Mask dimensions do not match the video")
                    opened_mask.verify()
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=422, detail="Mask is not a valid PNG image") from exc

            now = datetime.now(timezone.utc).isoformat()
            output_path = job_dir / "output" / "inpainted.mp4"
            manifest = {
                "job_id": job_id,
                "user_id": claims["user_id"],
                "status": "uploaded",
                "progress": 0,
                "created_at": now,
                "original_filenames": {"video": video_name, "mask": "static_mask.png"},
                "input_paths": {"video": str(video_path), "mask": str(mask_path)},
                "video_info": {"width": width, "height": height},
                "output_path": str(output_path),
                "output_token": secrets.token_urlsafe(32),
                "file_sizes_bytes": {"video": actual_video_size, "mask": mask_bytes},
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            await job_volume.commit.aio()
            return {"job_id": job_id, "status": "uploaded"}
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            await job_volume.commit.aio()
            raise

    async def start_uploaded_job(job_id: str, request: Request):
        authorize(request)
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        data = await request.json()
        user_id = str(data.get("user_id", ""))
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4096:
            raise HTTPException(status_code=400, detail="A prompt of 1 to 4096 characters is required")
        prompt = prompt.strip()
        await job_volume.reload.aio()
        job_dir = JOB_ROOT / "jobs" / job_id
        manifest_path = job_dir / "job.json"
        if not manifest_path.is_file():
            raise HTTPException(status_code=404, detail="Uploaded job not found")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("user_id") != user_id:
            raise HTTPException(status_code=404, detail="Uploaded job not found")
        if manifest.get("status") in {"queued", "running", "completed"}:
            return {
                "job_id": job_id,
                "status": manifest["status"],
                "modal_call_id": manifest.get("modal_call_id"),
            }
        if manifest.get("status") != "uploaded":
            raise HTTPException(status_code=409, detail="Job is not ready to start")
        input_paths = manifest.get("input_paths") or {}
        video_path = Path(str(input_paths.get("video", "")))
        mask_path = Path(str(input_paths.get("mask", "")))
        if not video_path.is_file() or not mask_path.is_file():
            raise HTTPException(status_code=409, detail="Video or mask is missing from the Modal job volume")

        manifest.update(status="queued", progress=0, prompt=prompt, queued_at=datetime.now(timezone.utc).isoformat())
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        await job_volume.commit.aio()
        try:
            function_call = process_job.spawn(job_id, str(video_path), str(mask_path), "standard", prompt, None)
        except Exception as exc:
            await job_volume.reload.aio()
            failed = json.loads(manifest_path.read_text(encoding="utf-8"))
            failed.update(status="failed", error=str(exc)[:500], failed_at=datetime.now(timezone.utc).isoformat())
            manifest_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")
            await job_volume.commit.aio()
            raise HTTPException(status_code=500, detail="Could not start the inpainting job") from exc
        try:
            await job_volume.reload.aio()
            latest = json.loads(manifest_path.read_text(encoding="utf-8"))
            latest["modal_call_id"] = function_call.object_id
            manifest_path.write_text(json.dumps(latest, indent=2), encoding="utf-8")
            await job_volume.commit.aio()
        except Exception as exc:
            # The GPU call has already been accepted. A manifest update failure
            # must not make the caller refund the job or submit a duplicate.
            print(f"Could not save Modal call ID for {job_id}: {exc}", flush=True)
            latest = manifest
        return {"job_id": job_id, "status": latest.get("status", "queued"), "modal_call_id": function_call.object_id}

    def download_signed_url(url: str, destination: Path, max_bytes: int, expected_type: str) -> int:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "storage.googleapis.com" or not parsed.query:
            raise HTTPException(status_code=400, detail="Only signed Firebase Storage download URLs are accepted")
        try:
            request = URLRequest(url, headers={"User-Agent": "Viresto-Object-Remover/1"})
            with urlopen(request, timeout=60) as response:
                if response.geturl() != url:
                    raise HTTPException(status_code=400, detail="Redirected storage URLs are not accepted")
                content_type = response.headers.get_content_type()
                if expected_type == "video/" and not content_type.startswith("video/"):
                    raise HTTPException(status_code=415, detail="Source URL is not a video")
                if expected_type != "video/" and content_type != expected_type:
                    raise HTTPException(status_code=415, detail=f"Mask must be {expected_type}")
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > max_bytes:
                    raise HTTPException(status_code=413, detail="Uploaded file exceeds the size limit")
                total = 0
                with destination.open("xb") as output_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > max_bytes:
                            raise HTTPException(status_code=413, detail="Uploaded file exceeds the size limit")
                        output_file.write(chunk)
                if total == 0:
                    raise HTTPException(status_code=400, detail="Uploaded file is empty")
                return total
        except HTTPException:
            raise
        except Exception as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Could not read the signed Firebase Storage file") from exc

    async def submit_job(request):
        authorize(request)
        data = await request.json()
        job_id = str(data.get("job_id", ""))
        user_id = str(data.get("user_id", ""))
        prompt = data.get("prompt")
        video_name = PurePosixPath(str(data.get("video_name", ""))).name
        extension = PurePosixPath(video_name).suffix.lower()
        if not re.fullmatch(r"[a-f0-9]{32}", job_id) or not user_id or len(user_id) > 128:
            raise HTTPException(status_code=400, detail="Invalid job identity")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4096:
            raise HTTPException(status_code=400, detail="A prompt of 1 to 4096 characters is required")
        if extension not in allowed_video_extensions:
            raise HTTPException(status_code=415, detail="Unsupported source video extension")

        job_dir = JOB_ROOT / "jobs" / job_id
        manifest_path = job_dir / "job.json"
        await job_volume.reload.aio()
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("user_id") != user_id:
                raise HTTPException(status_code=404, detail="Job not found")
            return {"job_id": job_id, "status": existing.get("status", "queued")}

        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=False)
        video_path = input_dir / f"source_video{extension}"
        static_mask_path = input_dir / "static_mask.png"
        try:
            video_bytes = download_signed_url(str(data.get("video_url", "")), video_path, max_video_bytes, "video/")
            mask_bytes = download_signed_url(str(data.get("mask_image_url", "")), static_mask_path, max_mask_bytes, "image/png")
            with Image.open(static_mask_path) as opened_mask:
                if opened_mask.size != (int(data.get("width", 0)), int(data.get("height", 0))):
                    raise HTTPException(status_code=422, detail="Uploaded mask dimensions do not match the editor canvas")
            now = datetime.now(timezone.utc).isoformat()
            output_path = job_dir / "output" / "inpainted.mp4"
            output_token = secrets.token_urlsafe(32)
            manifest = {
                "job_id": job_id,
                "user_id": user_id,
                "status": "queued",
                "progress": 0,
                "created_at": now,
                "prompt": prompt.strip(),
                "original_filenames": {"video": video_name, "mask": "static_mask.png"},
                "input_paths": {"video": str(video_path), "mask": str(static_mask_path)},
                "video_info": {"width": int(data.get("width", 0)), "height": int(data.get("height", 0))},
                "output_path": str(output_path),
                "output_token": output_token,
                "file_sizes_bytes": {"video": video_bytes, "mask": mask_bytes},
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            job_volume.commit()
            function_call = process_job.spawn(
                job_id, str(video_path), str(static_mask_path), "standard", prompt.strip(), None
            )
            return {
                "job_id": job_id,
                "status": "queued",
                "modal_call_id": function_call.object_id,
            }
        except HTTPException:
            shutil.rmtree(job_dir, ignore_errors=True)
            job_volume.commit()
            raise
        except Exception as exc:
            manifest = {
                "job_id": job_id,
                "user_id": user_id,
                "status": "failed",
                "error": str(exc)[:500],
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            job_volume.commit()
            raise HTTPException(status_code=500, detail="Could not create the inpainting job") from exc

    async def get_job(job_id: str, request, call_id: str | None = None):
        authorize(request)
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        await job_volume.reload.aio()
        manifest_path = JOB_ROOT / "jobs" / job_id / "job.json"
        if not manifest_path.is_file():
            raise HTTPException(status_code=404, detail="Job not found")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = manifest.get("status", "queued")
        error = manifest.get("error")
        if status in {"queued", "running"} and call_id:
            if not re.fullmatch(r"fc-[A-Za-z0-9]+", call_id):
                raise HTTPException(status_code=400, detail="Invalid Modal function call ID")
            try:
                call = modal.FunctionCall.from_id(call_id)
                call_result = await call.get.aio(timeout=0)
                if isinstance(call_result, dict) and call_result.get("output_path"):
                    status = "completed"
            except modal.exception.FunctionTimeoutError as exc:
                status = "failed"
                error = f"Modal function timed out: {exc}"[:500]
            except modal.exception.ExecutionError as exc:
                status = "failed"
                error = f"Modal function terminated: {exc}"[:500]
            except modal.exception.OutputExpiredError:
                status = "failed"
                error = "Modal function ended without saving a result."
            except modal.exception.NotFoundError:
                status = "failed"
                error = "Modal function call no longer exists; the job did not finish."
            except modal.exception.TimeoutError:
                # The Modal call is still queued or running.
                pass
            except Exception as exc:
                # A transient Modal control-plane error must not falsely fail a live job.
                print(f"Could not reconcile Modal call {call_id}: {exc}", flush=True)
            if status in {"failed", "completed"}:
                manifest.update(status=status)
                if status == "failed":
                    manifest.update(
                        error=error,
                        failed_at=datetime.now(timezone.utc).isoformat(),
                    )
                else:
                    manifest.update(
                        progress=100,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                    )
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                await job_volume.commit.aio()
        if manifest.get("status") == "completed":
            output_url = str(request.base_url).rstrip("/") + f"/api/jobs/{job_id}/output?token={manifest.get('output_token', '')}"
        else:
            output_url = None
        return {
            "job_id": job_id,
            "status": status,
            "progress": manifest.get("progress", 0),
            "error": error,
            "output_url": output_url,
        }

    @api.get("/api/jobs/{job_id}/output")
    async def get_output(job_id: str, token: str):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        await job_volume.reload.aio()
        job_dir = JOB_ROOT / "jobs" / job_id
        manifest_path = job_dir / "job.json"
        if not manifest_path.is_file():
            raise HTTPException(status_code=404, detail="Job not found")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = str(manifest.get("output_token", ""))
        if manifest.get("status") != "completed" or not expected or not hmac.compare_digest(token, expected):
            raise HTTPException(status_code=404, detail="Result not found")
        output_path = job_dir / "output" / "inpainted.mp4"
        if not output_path.is_file():
            raise HTTPException(status_code=404, detail="Result not found")
        return FileResponse(output_path, media_type="video/mp4", filename="inpainted.mp4")

    @api.get("/health")
    async def health():
        return {"status": "ok"}

    submit_job.__annotations__["request"] = Request
    get_job.__annotations__["request"] = Request
    upload_direct_job.__annotations__["request"] = Request
    start_uploaded_job.__annotations__["request"] = Request
    api.post("/api/jobs/from-urls", status_code=202)(submit_job)
    api.post("/api/jobs/{job_id}/upload", status_code=201)(upload_direct_job)
    api.post("/api/jobs/{job_id}/start", status_code=202)(start_uploaded_job)
    api.get("/api/jobs/{job_id}")(get_job)
    return api


@app.local_entrypoint()
def main(
    job_id: str,
    video_path: str,
    mask_path: str,
    mode: str = "standard",
    prompt: str | None = None,
    seed: int | None = None,
) -> None:
    result = process_job.remote(job_id, video_path, mask_path, mode, prompt, seed)
    print(result)
