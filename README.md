# LTX-2.5 IC-LoRA Video Inpainting — Python Workflow

Remove or modify objects in video using mask-conditioned inpainting with Lightricks LTX-2.5 + IC-LoRA.

## Folder Structure

```
ltx_inpaint_python_workflow/
├── setup_and_download.py     # Step 1: Install deps + download models
├── run_inpaint.py            # Step 2: Run inpainting
├── handler.py                # RunPod Serverless job handler
├── Dockerfile                # CUDA 13.2 + PyTorch worker image
├── README.md
├── models/                   # Downloaded model weights (~55 GB)
│   ├── diffusion_models/     # Base transformer
│   ├── vae/                  # Convolutional video VAE and audio VAE
│   ├── text_encoders/        # Gemma 4 text encoder
│   ├── latent_upscale_models/# 2x spatial upscaler
│   ├── loras/                # IC-LoRA inpainting adapter
│   └── prompt_enhancer/      # Optional text enhancer
├── inputs/                   # Your input videos & masks
├── outputs/                  # Generated results
└── LTX-2/                    # Cloned official repo (auto-created by setup)
```

## RunPod Serverless

The worker accepts any CUDA-capable GPU and uses the low-VRAM inference preset. Smaller GPUs may still run out of memory depending on video resolution and length. It expects the attached RunPod Network Volume at `/runpod-volume` and reads models from `/runpod-volume/models`. The container image includes Python, CUDA-compatible PyTorch, the LTX pipeline code, and Python dependencies; it does not include model weights.

### Put model weights on the network volume

For an initial pre-download outside the first inference request, attach the volume to a temporary RunPod Pod with its mount path set to `/runpod-volume`, provide `HF_TOKEN` through a private environment variable, and run:

```bash
export LTX_MODELS_DIR=/runpod-volume/models
export HF_HOME=/runpod-volume/huggingface
export HF_HUB_CACHE=/runpod-volume/huggingface/hub
python /app/setup_and_download.py --download-only --yes
```

The download-only mode reuses the worker's Python environment, writes the convolutional video VAE and other configured weights under the network volume, and skips dependency installation and repository cloning. Hugging Face access must already be approved for gated model repositories.

You can also skip this pre-download: the Serverless handler checks the volume before each job. If required weights are missing, the first job downloads them directly to `/runpod-volume/models` before starting inference. Add `HF_TOKEN` as a private endpoint environment secret first. The first request will take longer and incur GPU worker time while the weights download.

### Build and publish the worker image

```bash
docker build -t ghcr.io/YOUR_GITHUB_USER/ltx-inpaint-worker:latest .
docker push ghcr.io/YOUR_GITHUB_USER/ltx-inpaint-worker:latest
```

Create a **queue-based** Serverless endpoint using that image, a CUDA-capable GPU, data center `EUR-NO-1`, and network volume `4bllfzloan` (the recreated 100 GB Standard volume). Mount it at `/runpod-volume`. Add an endpoint environment variable with key `HF_TOKEN` and value `{{ RUNPOD_SECRET_HF_TOKEN }}` so RunPod injects the saved secret when the worker starts.

With the RunPod CLI authenticated, the endpoint creation command will look like this after the image is published and the matching volume exists:

```powershell
runpod endpoint create `
  --name ltx-inpaint `
  --type queue `
  --image YOUR_REGISTRY/ltx-inpaint-worker:latest `
  --gpu "RTX 5090" `
  --data-center EUR-NO-1 `
  --network-volume 4bllfzloan `
  --cuda 13.2 `
  --min-workers 0 `
  --max-workers 1 `
  --idle-timeout-seconds 5
```

### Request format

Input paths must already exist inside the mounted volume. Relative paths are resolved from `/runpod-volume`.

```json
{
  "input": {
    "video_path": "/runpod-volume/inputs/source_video.mp4",
    "mask_path": "/runpod-volume/inputs/mask_video.mp4",
    "output_path": "/runpod-volume/outputs/source_inpainted.mp4",
    "prompt": "Continue the surrounding scene naturally; remove the logo and add no text or watermark.",
    "seed": 1234
  }
}
```

`output_path` is optional; when omitted, a unique result path is created under `/runpod-volume/outputs/`. The handler runs the existing two-stage low-VRAM flow with the convolutional VAE. It returns the output path on the network volume.

## Quick Start

### 1. Setup (one-time)

```bash
python setup_and_download.py
```

This will:
- Install Python dependencies (torch, huggingface_hub, etc.)
- Clone the official [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) repo
- Install `ltx-core` and `ltx-pipelines` packages
- Download model weights from HuggingFace, including the convolutional video VAE

> **Note:** Set your `HF_TOKEN` environment variable before running. The script auto-authenticates with it.
> You also need to visit each model page on HuggingFace and click "Agree and Access" once.

### 2. Run Inpainting

```bash
# Simplest run (white pixels in mask_video.mp4 mark the logo/area to replace):
python run_inpaint.py

# Or specify custom video paths:
python run_inpaint.py --source inputs/my_video.mp4 --mask inputs/my_mask.mp4

# Custom scene description while removing the masked logo:
python run_inpaint.py --source inputs/video.mp4 --mask inputs/mask.mp4 --prompt "continue the brick wall texture seamlessly; no logo, text, or watermark"
```

### 3. Low VRAM Options

```bash
# Option A: Two-stage low-VRAM preset (CPU offload, Conv VAE & tiling)
python run_inpaint.py --low-vram

# Option B: Lower VRAM (with FP8 quantization + CPU offloading; retains both stages)
python run_inpaint.py --low-vram --quantization fp8-scaled-mm --offload cpu

# Individual switches:
# 1. Skip stage 2 explicitly (intermediate generation is half resolution)
python run_inpaint.py --skip-stage-2

# 2. Use Convolutional VAE (much lower memory than Diffusion VAE)
python run_inpaint.py --use-conv-vae

# 3. Enable spatial tiling (prevents OOM on high resolutions)
python run_inpaint.py --tile
```

## Mask Format

- **Grayscale** video (or black and white)
- **Frame count** matches the source, or is one frame for a static mask
- Output uses the LTX canvas resolution
- **White (255)** = area to **INPAINT** (remove/modify), such as a logo
- **Black (0)** = area to **PRESERVE**
- Can be generated using SAM2, After Effects, or any rotoscoping tool

Mask polarity is preserved: white marks the area to remove and black marks the area to preserve. The runner applies the same centered crop to the source and mask, paints the white area LTX green in a temporary reference video, runs IC-LoRA, then Laplacian-pyramid blends through the aligned mask. Pixels outside the mask come from the transformed source clip. The source audio is retained.

Single-frame masks are reused for every source frame during reference generation and final blending.

The script keeps the ComfyUI workflow's two-stage LTX generation and Laplacian blending, while choosing a close-aspect canvas divisible by 64 with neither dimension smaller than the source. For a 720x1280 source, this produces a 768x1344 output. It applies a matching centered aspect crop to the green-mask reference, source, and mask before generation and final blending, keeping them pixel-aligned without stretching. `--skip-stage-2` generates an intermediate at half canvas size. `--low-vram` keeps both stages enabled and reduces memory use with offloading, tiling, and the convolutional VAE.

## Requirements

| Component | Minimum | Recommended |
|:--|:--|:--|
| GPU VRAM | 16 GB | 24 GB+ |
| Disk Space | 55 GB | 60 GB+ |
| Python | 3.10+ | 3.12 |
| CUDA | 11.8+ | 12.7+ |

## Models Used

| Model | Source |
|:--|:--|
| LTX-2.5 22B Distilled Transformer | `Lightricks/LTX-2.5` |
| IC-LoRA In/Outpainting v0.9 | `Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting` |
| Video VAE (convolutional) | `Lightricks/LTX-2.5` |
| Audio VAE | `Lightricks/LTX-2.5` |
| Latent Spatial Upscaler 2x | `Lightricks/LTX-2.5` |
| Gemma 4 12B Text Encoder | `Lightricks/LTX-2.5` |
| Gemma 4 E2B Text Enhancer | `Comfy-Org/gemma-4` |
