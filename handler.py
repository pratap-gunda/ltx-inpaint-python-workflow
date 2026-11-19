"""RunPod Serverless entry point for LTX IC-LoRA video inpainting.

The request supplies paths that already exist on the attached network volume.
Example input:
    {"video_path": "/runpod-volume/inputs/source_video.mp4",
     "mask_path": "/runpod-volume/inputs/mask_video.mp4"}
"""

from __future__ import annotations

import os
import fcntl
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import runpod
import torch


APP_DIR = Path(__file__).resolve().parent
VOLUME_ROOT = Path(os.environ.get("RUNPOD_VOLUME_ROOT", "/runpod-volume")).resolve()
OUTPUT_DIR = VOLUME_ROOT / "outputs"
RUN_SCRIPT = APP_DIR / "run_inpaint.py"
SETUP_SCRIPT = APP_DIR / "setup_and_download.py"
MODEL_DIR = Path(os.environ.get("LTX_MODELS_DIR", VOLUME_ROOT / "models")).resolve()
REQUIRED_MODEL_FILES = (
    "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "loras/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
)


def _volume_path(value: Any, field: str, *, must_exist: bool) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"input.{field} must be a non-empty path string")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = VOLUME_ROOT / path
    path = path.resolve()

    try:
        path.relative_to(VOLUME_ROOT)
    except ValueError as exc:
        raise ValueError(f"input.{field} must be inside {VOLUME_ROOT}") from exc

    if must_exist and not path.is_file():
        raise FileNotFoundError(f"input.{field} does not exist on the worker: {path}")
    return path


def _missing_models() -> list[Path]:
    return [MODEL_DIR / relative_path for relative_path in REQUIRED_MODEL_FILES
            if not (MODEL_DIR / relative_path).is_file()]


def _materialize_gemma_encoder() -> None:
    """Replace an HF-cache symlink with a filename-preserving hard link."""
    path = MODEL_DIR / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    if not path.is_symlink():
        return

    cache_blob = path.resolve(strict=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            os.link(cache_blob, temp_path)
            print(f"Materialized Gemma text encoder with hard link: {path}", flush=True)
        except OSError:
            shutil.copy2(cache_blob, temp_path)
            print(f"Materialized Gemma text encoder copy: {path}", flush=True)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _ensure_models() -> None:
    """Download missing weights directly into the persistent network volume."""
    missing = _missing_models()
    if not missing:
        _materialize_gemma_encoder()
        return

    if not os.environ.get("HF_TOKEN"):
        names = ", ".join(path.name for path in missing)
        raise RuntimeError(
            f"Required models are missing from {MODEL_DIR}: {names}. "
            "Set HF_TOKEN as a private Serverless environment secret; the first job "
            "will download the weights directly to the attached network volume."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = VOLUME_ROOT / ".ltx-model-download.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        missing = _missing_models()
        if missing:
            print(
                f"Downloading {len(missing)} missing model file(s) into {MODEL_DIR}...",
                flush=True,
            )
            result = subprocess.run(
                [sys.executable, str(SETUP_SCRIPT), "--download-only", "--yes"],
                cwd=APP_DIR,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Model download command failed with exit code {result.returncode}")
            still_missing = _missing_models()
            if still_missing:
                names = ", ".join(path.name for path in still_missing)
                raise RuntimeError(f"Model download did not produce required files: {names}")
            _materialize_gemma_encoder()
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """Run one inpainting job and return the path to the saved output."""
    payload = job.get("input") or {}
    if not isinstance(payload, dict):
        raise ValueError("job.input must be a JSON object")

    if not VOLUME_ROOT.is_dir():
        raise RuntimeError(
            f"Network volume is not mounted at {VOLUME_ROOT}. Attach the volume to the endpoint."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this worker; assign a CUDA-enabled GPU.")
    gpu = torch.cuda.get_device_properties(0)
    gpu_vram_gib = gpu.total_memory / (1024**3)
    print(
        f"Using CUDA GPU {gpu.name} with {gpu_vram_gib:.1f} GiB VRAM; "
        "the low-VRAM inference preset will manage memory usage.",
        flush=True,
    )

    source = _volume_path(payload.get("video_path"), "video_path", must_exist=True)
    mask = _volume_path(payload.get("mask_path"), "mask_path", must_exist=True)
    _ensure_models()
    job_id = str(job.get("id") or uuid.uuid4().hex)
    output_value = payload.get("output_path") or f"outputs/{job_id}_inpainted.mp4"
    output = _volume_path(output_value, "output_path", must_exist=False)
    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(RUN_SCRIPT),
        "--source",
        str(source),
        "--mask",
        str(mask),
        "--output",
        str(output),
        # Keep both stages enabled; use the configuration that has worked locally.
        "--low-vram",
        "--use-conv-vae",
    ]

    prompt = payload.get("prompt")
    if prompt is not None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("input.prompt must be a non-empty string when supplied")
        command.extend(["--prompt", prompt])

    seed = payload.get("seed")
    if seed is not None:
        command.extend(["--seed", str(int(seed))])

    quantization = payload.get("quantization")
    if quantization is not None:
        if not isinstance(quantization, str) or quantization not in {
            "fp8-cast",
            "fp8-scaled-mm",
            "nvfp4-cast",
            "nvfp4-prequant",
        }:
            raise ValueError("input.quantization is not a supported quantization mode")
        command.extend(["--quantization", quantization])

    print(f"Starting inpaint job {job_id}: video={source}, mask={mask}, output={output}", flush=True)
    result = subprocess.run(command, cwd=APP_DIR, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Inpainting process failed with exit code {result.returncode}")
    if not output.is_file():
        raise RuntimeError(f"Inpainting completed without creating the expected output: {output}")

    return {"job_id": job_id, "output_path": str(output)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
