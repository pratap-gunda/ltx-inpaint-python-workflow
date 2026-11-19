"""
LTX-2.5 IC-LoRA Video Inpainting — Setup & Download Script
============================================================
This script:
  1. Installs all required Python dependencies (torch, huggingface_hub, etc.)
  2. Clones the official Lightricks/LTX-2 repository
  3. Installs the ltx-pipelines and ltx-core packages
  4. Downloads all required model weights into the local models/ folder

Usage:
    python setup_and_download.py

Requirements:
    - Python >= 3.10
    - CUDA-capable GPU with drivers installed
    - ~55 GB free disk space for model downloads
    - HuggingFace account (for gated model access)
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("LTX_MODELS_DIR", SCRIPT_DIR / "models")).expanduser().resolve()
LTX2_REPO_DIR = Path(os.environ.get("LTX2_REPO_DIR", SCRIPT_DIR / "LTX-2")).expanduser().resolve()

# Model download definitions: (repo_id, filename, subfolder_in_repo, local_target_path)
MODEL_DOWNLOADS = [
    # --- From Lightricks/LTX-2.5 ---
    {
        "repo_id": "Lightricks/LTX-2.5",
        "filename": "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "subfolder": "diffusion_models",
        "local_path": MODELS_DIR / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "description": "Base DiT Transformer (22B distilled)",
    },
    {
        "repo_id": "Lightricks/LTX-2.5",
        "filename": "ltx-2.5-audio-vae-bf16.safetensors",
        "subfolder": "vae",
        "local_path": MODELS_DIR / "vae" / "ltx-2.5-audio-vae-bf16.safetensors",
        "description": "Audio VAE",
    },
    {
        "repo_id": "Lightricks/LTX-2.5",
        "filename": "ltx-2.5-video-vae-conv-bf16.safetensors",
        "subfolder": "vae",
        "local_path": MODELS_DIR / "vae" / "ltx-2.5-video-vae-conv-bf16.safetensors",
        "description": "Video VAE (convolutional decoder, lower VRAM)",
    },
    {
        "repo_id": "Lightricks/LTX-2.5",
        "filename": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "subfolder": "latent_upscale_models",
        "local_path": MODELS_DIR / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "description": "Latent Spatial Upscaler (2x)",
    },
    {
        "repo_id": "Lightricks/LTX-2.5",
        "filename": "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "subfolder": "text_encoders",
        "local_path": MODELS_DIR / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "description": "Text Encoder (Gemma 4 12B, LTX-2.5 finetuned)",
    },
    # --- IC-LoRA for Inpainting (from separate repo) ---
    {
        "repo_id": "Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting",
        "filename": "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
        "subfolder": None,
        "local_path": MODELS_DIR / "loras" / "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
        "description": "IC-LoRA In/Outpainting Adapter",
    },
    # --- Text Enhancer (optional but recommended) ---
    {
        "repo_id": "Comfy-Org/gemma-4",
        "filename": "gemma4_e2b_it_bf16.safetensors",
        "subfolder": "text_encoders",
        "local_path": MODELS_DIR / "prompt_enhancer" / "gemma4_e2b_it_bf16.safetensors",
        "description": "Text Enhancer (Gemma 4 E2B IT) [optional]",
    },
]


# ============================================================================
# Helper functions
# ============================================================================

def run_cmd(cmd: list[str], cwd: str | Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and stream output."""
    print(f"\n{'='*60}")
    print(f"  Running: {' '.join(str(c) for c in cmd)}")
    if cwd:
        print(f"  In: {cwd}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        # Stream output in real-time
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return result


def check_python_version():
    """Ensure Python >= 3.10."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        print(f"ERROR: Python >= 3.10 required. You have {major}.{minor}")
        sys.exit(1)
    print(f"✓ Python version: {major}.{minor}")


def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ CUDA available: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            if vram_gb < 16:
                print(f"  ⚠ WARNING: {vram_gb:.1f} GB VRAM is below recommended 24 GB.")
                print(f"             You may need --skip-stage-2 or --tile flags when running.")
        else:
            print("⚠ WARNING: CUDA not available. GPU is required for inference.")
    except ImportError:
        print("  (torch not yet installed — CUDA check will happen after install)")


def install_pip_dependencies():
    """Install core Python dependencies."""
    print("\n" + "="*60)
    print("  STEP 1: Installing Python dependencies")
    print("="*60)

    deps = [
        "torch>=2.1.0",
        "huggingface_hub",
        "safetensors",
        "tqdm",
        "pillow",
        "av",          # PyAV for video I/O
        "numpy",
    ]

    run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run_cmd([sys.executable, "-m", "pip", "install"] + deps)
    print("\n✓ Core Python dependencies installed.")


def clone_ltx2_repo():
    """Clone the official Lightricks/LTX-2 repository."""
    print("\n" + "="*60)
    print("  STEP 2: Cloning Lightricks/LTX-2 repository")
    print("="*60)

    if LTX2_REPO_DIR.exists():
        print(f"  LTX-2 repo already exists at: {LTX2_REPO_DIR}")
        print("  Pulling latest changes...")
        run_cmd(["git", "pull"], cwd=LTX2_REPO_DIR, check=False)
    else:
        run_cmd(["git", "clone", "https://github.com/Lightricks/LTX-2.git", str(LTX2_REPO_DIR)])

    print(f"\n✓ LTX-2 repo ready at: {LTX2_REPO_DIR}")


def install_ltx_packages():
    """Install ltx-core and ltx-pipelines from the cloned repo."""
    print("\n" + "="*60)
    print("  STEP 3: Installing ltx-core & ltx-pipelines packages")
    print("="*60)

    # Install ltx-core first (dependency of ltx-pipelines)
    ltx_core_dir = LTX2_REPO_DIR / "packages" / "ltx-core"
    if ltx_core_dir.exists():
        run_cmd([sys.executable, "-m", "pip", "install", "-e", str(ltx_core_dir)])
        print("  ✓ ltx-core installed")
    else:
        print(f"  ⚠ ltx-core not found at {ltx_core_dir}")
        print("    Attempting install from repo root...")
        run_cmd([sys.executable, "-m", "pip", "install", "-e", str(LTX2_REPO_DIR)], check=False)

    # Install ltx-pipelines
    ltx_pipelines_dir = LTX2_REPO_DIR / "packages" / "ltx-pipelines"
    if ltx_pipelines_dir.exists():
        run_cmd([sys.executable, "-m", "pip", "install", "-e", str(ltx_pipelines_dir)])
        print("  ✓ ltx-pipelines installed")
    else:
        print(f"  ⚠ ltx-pipelines not found at {ltx_pipelines_dir}")

    # Try installing natten (optional, needed for spatial upsampler)
    print("\n  Attempting to install NATTEN (for spatial upsampler)...")
    result = run_cmd(
        [sys.executable, "-m", "pip", "install", "natten"],
        check=False,
    )
    if result.returncode != 0:
        print("  ⚠ NATTEN install failed (common on Windows).")
        print("    The pipeline will still work with --skip-stage-2 flag.")
    else:
        print("  ✓ NATTEN installed")

    print("\n✓ LTX packages installed.")


def download_models():
    """Download all model weights from HuggingFace."""
    print("\n" + "="*60)
    print("  STEP 4: Downloading model weights from HuggingFace")
    print("="*60)

    try:
        from huggingface_hub import hf_hub_download, login
    except ImportError:
        run_cmd([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download, login

    # Authenticate using HF_TOKEN from environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print(f"\n  ✓ HF_TOKEN found in environment — authenticating automatically.")
        try:
            login(token=hf_token, add_to_git_credential=False)
            print("  ✓ HuggingFace authentication successful.")
        except Exception as e:
            print(f"  ⚠ HuggingFace login warning: {e}")
            print("    Will still attempt downloads with token directly.")
    else:
        print("\n  ⚠ HF_TOKEN not found in environment variables.")
        print("    Set it via: set HF_TOKEN=hf_your_token_here  (Windows)")
        print("    Or run: huggingface-cli login")
        print("    Some gated models may fail to download without authentication.")

    print("\n  Note: Visit each model page and click 'Agree and Access' if prompted.\n")

    total = len(MODEL_DOWNLOADS)
    for i, model in enumerate(MODEL_DOWNLOADS, 1):
        local_path = Path(model["local_path"])

        # Skip regular files, but repair legacy symlinks into HF's hash-only
        # blob cache so model loaders receive the expected filename.
        if local_path.exists() and not local_path.is_symlink():
            size_gb = local_path.stat().st_size / (1024**3)
            print(f"  [{i}/{total}] ✓ SKIP (exists, {size_gb:.2f} GB): {model['description']}")
            continue
        if local_path.is_symlink():
            local_path.unlink()

        print(f"\n  [{i}/{total}] Downloading: {model['description']}")
        print(f"           From: {model['repo_id']}")
        print(f"           File: {model['filename']}")

        # Create target directory
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            downloaded_path = hf_hub_download(
                repo_id=model["repo_id"],
                filename=model["filename"],
                subfolder=model["subfolder"],
                local_dir=None,  # Use HF cache
                token=hf_token,  # Pass token directly for gated repos
            )

            # Give model loaders a real filename ending in .safetensors. A
            # symlink is resolved to HF's hash-only blob name, which Gemma's
            # loader rejects. A hard link avoids duplicating the large weight
            # when the HF cache and models directory share a filesystem.
            downloaded_path = Path(downloaded_path).resolve()
            if local_path.is_symlink():
                local_path.unlink()
            if not local_path.exists():
                try:
                    os.link(downloaded_path, local_path)
                    print(f"           ✓ Hard-linked to: {local_path}")
                except OSError:
                    shutil.copy2(downloaded_path, local_path)
                    print(f"           ✓ Copied to: {local_path}")

        except Exception as e:
            print(f"           ✗ FAILED: {e}")
            print(f"           → Make sure you've accepted terms at:")
            print(f"             https://huggingface.co/{model['repo_id']}")
            if not hf_token:
                print(f"           → Set HF_TOKEN env variable or run: huggingface-cli login")

    print("\n✓ Model download step complete.")


def print_summary():
    """Print a summary of what was set up."""
    print("\n" + "="*60)
    print("  SETUP COMPLETE — Summary")
    print("="*60)

    print(f"\n  Workspace:    {SCRIPT_DIR}")
    print(f"  LTX-2 Repo:   {LTX2_REPO_DIR}")
    print(f"  Models:       {MODELS_DIR}")

    print("\n  Model Files:")
    for model in MODEL_DOWNLOADS:
        local_path = Path(model["local_path"])
        status = "✓" if local_path.exists() else "✗ MISSING"
        size = ""
        if local_path.exists():
            size_gb = local_path.stat().st_size / (1024**3)
            size = f" ({size_gb:.2f} GB)"
        print(f"    {status} {model['description']}{size}")
        print(f"        → {local_path}")

    print(f"\n  Next Steps:")
    print(f"    1. Place your input video in:  {SCRIPT_DIR / 'inputs' / 'source_video.mp4'}")
    print(f"    2. Place your mask video in:   {SCRIPT_DIR / 'inputs' / 'mask_video.mp4'}")
    print(f"    3. Run: python run_inpaint.py --source inputs/source_video.mp4 --mask inputs/mask_video.mp4")

    # CUDA check after everything is installed
    print()
    check_cuda()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Install LTX dependencies and/or download its model weights.")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download model files; do not install Python packages or clone the LTX repository.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the interactive confirmation prompt.")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  LTX-2.5 IC-LoRA Video Inpainting — Setup & Download   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    check_python_version()

    if args.download_only:
        print(f"\nDownloading model weights into: {MODELS_DIR}")
    else:
        print("\nThis script will:")
        print("  1. Install Python dependencies (torch, huggingface_hub, etc.)")
        print("  2. Clone Lightricks/LTX-2 repo")
        print("  3. Install ltx-core & ltx-pipelines packages")
        print("  4. Download model weights from HuggingFace, including the convolutional video VAE")

    if not args.yes and not args.download_only:
        response = input("\nProceed? [Y/n]: ").strip().lower()
        if response and response != "y":
            print("Aborted.")
            sys.exit(0)

    if not args.download_only:
        install_pip_dependencies()
        clone_ltx2_repo()
        install_ltx_packages()
    download_models()
    print_summary()


if __name__ == "__main__":
    main()
