# RTX 5090 is available to RunPod Serverless with CUDA 13.2 in the catalog.
# Keep CUDA/PyTorch aligned with that runtime, with no model weights baked in.
FROM nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RUNPOD_VOLUME_ROOT=/runpod-volume \
    LTX_MODELS_DIR=/runpod-volume/models \
    LTX_OUTPUTS_DIR=/runpod-volume/outputs \
    HF_HOME=/runpod-volume/huggingface \
    HF_HUB_CACHE=/runpod-volume/huggingface/hub

RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      python3.12 \
      python3.12-venv \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# Use a matched PyTorch package set. CUDA 13.2 currently publishes torch 2.13
# without a matching torchaudio wheel; CUDA 13.0 has a complete 2.11.0 set.
# Its bundled CUDA runtime is compatible with the CUDA 13.2 container driver.
RUN python -m pip install --upgrade pip \
    && python -m pip install \
      --index-url https://download.pytorch.org/whl/cu130 \
      torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# Install the official LTX pipeline source and the worker's runtime dependencies.
ARG LTX2_GIT_REF=main
RUN git clone --depth 1 --branch "${LTX2_GIT_REF}" \
      https://github.com/Lightricks/LTX-2.git /opt/LTX-2 \
    && python -m pip install \
      -e /opt/LTX-2/packages/ltx-core \
      -e /opt/LTX-2/packages/ltx-pipelines \
      runpod huggingface_hub safetensors tqdm pillow av numpy

ENV LTX2_REPO_DIR=/opt/LTX-2

COPY run_inpaint.py setup_and_download.py handler.py ./

CMD ["python", "-u", "/app/handler.py"]
