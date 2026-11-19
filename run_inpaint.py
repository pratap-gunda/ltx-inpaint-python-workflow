"""
LTX-2.5 IC-LoRA Video Inpainting — Run Script
===============================================
Takes a source video + mask video and inpaints/removes the masked region.

Usage:
    # Run with default inputs (inputs/source_video.mp4 & inputs/mask_video.mp4):
    python run_inpaint.py

    # Or specify custom paths:
    python run_inpaint.py --source inputs/video.mp4 --mask inputs/mask.mp4

    # Custom prompt:
    python run_inpaint.py --source inputs/video.mp4 --mask inputs/mask.mp4 --prompt "a wooden chair" --output outputs/result.mp4

Mask format:
    - Grayscale video, same resolution & frame count as source
    - White (255) = area to INPAINT (remove/modify)
    - Black (0)   = area to PRESERVE

Requirements:
    - Run setup_and_download.py first
    - CUDA GPU with >= 16 GB VRAM (24 GB+ recommended)
"""

import argparse
import math
import os
import sys
import subprocess
import time
import tempfile
from fractions import Fraction
from pathlib import Path


# ============================================================================
# Configuration — Model Paths
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("LTX_MODELS_DIR", SCRIPT_DIR / "models")).expanduser().resolve()
LTX2_REPO_DIR = Path(os.environ.get("LTX2_REPO_DIR", SCRIPT_DIR / "LTX-2")).expanduser().resolve()
OUTPUTS_DIR = Path(os.environ.get("LTX_OUTPUTS_DIR", SCRIPT_DIR / "outputs")).expanduser().resolve()

# Model file paths (relative to models/)
MODEL_PATHS = {
    "transformer": MODELS_DIR / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoder": MODELS_DIR / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae": MODELS_DIR / "vae" / "ltx-2.5-video-vae-bf16.safetensors",
    "video_vae_conv": MODELS_DIR / "vae" / "ltx-2.5-video-vae-conv-bf16.safetensors",
    "audio_vae": MODELS_DIR / "vae" / "ltx-2.5-audio-vae-bf16.safetensors",
    "spatial_upsampler": MODELS_DIR / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "ic_lora": MODELS_DIR / "loras" / "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
    "prompt_enhancer": MODELS_DIR / "prompt_enhancer" / "gemma4_e2b_it_bf16.safetensors",
}


# ============================================================================
# Validation helpers
# ============================================================================

def resolve_input_file(path_str: str, candidates: list[str], is_mask: bool = False) -> Path | None:
    """Find file from specified path, candidate list, or inputs/ directory search."""
    p = Path(path_str)
    if p.exists():
        return p
    if (SCRIPT_DIR / p).exists():
        return SCRIPT_DIR / p

    inputs_dirs = [SCRIPT_DIR / "inputs", Path("inputs")]
    video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

    for idir in inputs_dirs:
        if not idir.exists():
            continue
        # Check specific candidate names
        for cname in candidates:
            cand = idir / cname
            if cand.exists():
                return cand

        # Fallback search by filename pattern
        for f in sorted(idir.iterdir()):
            if f.is_file() and f.suffix.lower() in video_exts:
                name_lower = f.name.lower()
                if is_mask and "mask" in name_lower:
                    return f
                elif not is_mask and "mask" not in name_lower:
                    return f
    return None


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate that all required files exist and inputs are sane."""
    errors = []

    # Resolve source video (supports source_video.mp4, videeo.mp4, video.mp4, etc.)
    source_candidates = ["source_video.mp4", "videeo.mp4", "video.mp4", "input.mp4", "source.mp4"]
    resolved_source = resolve_input_file(args.source, source_candidates, is_mask=False)
    if resolved_source:
        args.source = str(resolved_source.resolve())
    else:
        errors.append(
            f"Source video not found: {args.source}\n"
            f"    Place your video at 'inputs/source_video.mp4' (or 'inputs/videeo.mp4') or pass --source <path>"
        )

    # Resolve mask video (supports mask_video.mp4, mask.mp4, etc.)
    mask_candidates = ["mask_video.mp4", "mask.mp4", "input_mask.mp4"]
    resolved_mask = resolve_input_file(args.mask, mask_candidates, is_mask=True)
    if resolved_mask:
        args.mask = str(resolved_mask.resolve())
    else:
        errors.append(
            f"Mask video not found: {args.mask}\n"
            f"    Place your mask at 'inputs/mask_video.mp4' (or 'inputs/mask.mp4') or pass --mask <path>"
        )

    # Check essential model files
    video_vae_key = "video_vae_conv" if args.use_conv_vae else "video_vae"
    essential_models = [
        "transformer",
        "text_encoder",
        video_vae_key,
        "audio_vae",
        "spatial_upsampler",
        "ic_lora",
    ]
    for key in essential_models:
        path = MODEL_PATHS[key]
        if not path.exists():
            errors.append(f"Model not found: {key} → {path}")

    # Check video VAE (auto-select whichever variant is downloaded)
    prefer_conv = args.use_conv_vae or args.low_vram
    if prefer_conv and MODEL_PATHS["video_vae_conv"].exists():
        args.use_conv_vae = True
    elif MODEL_PATHS["video_vae"].exists():
        args.use_conv_vae = False
    elif MODEL_PATHS["video_vae_conv"].exists():
        args.use_conv_vae = True
    else:
        errors.append(
            f"No video VAE found in {MODELS_DIR / 'vae'}. "
            f"Expected ltx-2.5-video-vae-bf16.safetensors or conv variant."
        )

    # Check LTX-2 repo
    if not LTX2_REPO_DIR.exists():
        errors.append(f"LTX-2 repo not found at: {LTX2_REPO_DIR}\n  Run setup_and_download.py first.")

    if errors:
        print("\n✗ Validation errors:")
        for e in errors:
            print(f"  • {e}")
        print("\n  Run setup_and_download.py to install dependencies and download models.")
        sys.exit(1)

    print("✓ All inputs and models validated.")


def get_video_info(video_path: str) -> dict:
    """Get basic video info using PyAV."""
    try:
        import av
        container = av.open(video_path)
        stream = container.streams.video[0]
        info = {
            "width": stream.width,
            "height": stream.height,
            "frames": stream.frames if stream.frames > 0 else None,
            "fps": float(stream.average_rate) if stream.average_rate else None,
            "duration": float(container.duration / av.time_base) if container.duration else None,
        }
        container.close()
        return info
    except Exception as e:
        print(f"  ⚠ Could not read video info: {e}")
        return {}


def _temporary_mp4(prefix: str) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".mp4", dir=OUTPUTS_DIR, delete=False)
    path = Path(handle.name)
    handle.close()
    return path


def prepare_inpaint_reference(
    source_path: str,
    mask_path: str,
    output_path: Path,
    width: int,
    height: int,
    dilation: int = 5,
) -> None:
    """Resize to the ComfyUI LTX canvas, then paint white-mask pixels green.

    The user's mask remains white=inpaint. No polarity conversion is performed.
    Source, mask, and generated video must share this exact canvas for blending.
    """
    import av
    import numpy as np
    from PIL import Image, ImageFilter, ImageOps

    source = av.open(source_path)
    mask_container = av.open(mask_path)
    source_stream = source.streams.video[0]
    mask_stream = mask_container.streams.video[0]
    fps = source_stream.average_rate or Fraction(24, 1)
    encoder = av.open(str(output_path), mode="w")
    out_stream = encoder.add_stream("libx264", rate=fps)
    out_stream.width = width
    out_stream.height = height
    pix_fmt = "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p"
    out_stream.pix_fmt = pix_fmt

    mask_frames = iter(mask_container.decode(video=0))
    single_mask = mask_stream.frames == 1
    first_mask = next(mask_frames, None)
    if first_mask is None:
        raise ValueError(f"Mask video contains no frames: {mask_path}")
    frame_index = 0
    try:
        for source_frame in source.decode(video=0):
            mask_frame = first_mask if single_mask else (first_mask if frame_index == 0 else next(mask_frames, None))
            if mask_frame is None:
                raise ValueError("Mask video ended before the source video. Match their frame counts.")

            source_image = source_frame.to_image().convert("RGB")
            mask_image = mask_frame.to_image().convert("L")
            if mask_image.size != source_image.size:
                mask_image = mask_image.resize(source_image.size, Image.Resampling.NEAREST)
            source_image = ImageOps.fit(
                source_image, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
            )
            mask_image = ImageOps.fit(
                mask_image, (width, height), method=Image.Resampling.BILINEAR, centering=(0.5, 0.5)
            )
            mask_image = mask_image.point(lambda value: 255 if value >= 128 else 0)
            if dilation > 0:
                mask_image = mask_image.filter(ImageFilter.MaxFilter(dilation * 2 + 1))

            green = Image.new("RGB", source_image.size, (102, 255, 0))
            composite = Image.composite(green, source_image, mask_image)
            video_frame = av.VideoFrame.from_image(composite)
            video_frame = video_frame.reformat(format=pix_fmt)
            video_frame.pts = frame_index
            video_frame.time_base = Fraction(fps.denominator, fps.numerator)
            for packet in out_stream.encode(video_frame):
                encoder.mux(packet)
            frame_index += 1

        for packet in out_stream.encode():
            encoder.mux(packet)
    finally:
        encoder.close()
        source.close()
        mask_container.close()


def _laplacian_blend_frame(original, generated, mask, levels: int = 6, dilation: int = 6):
    """Laplacian pyramid blend; white mask selects generated pixels."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    orig = torch.from_numpy(np.asarray(original, dtype=np.float32).copy() / 255.0).permute(2, 0, 1).unsqueeze(0)
    gen = torch.from_numpy(np.asarray(generated, dtype=np.float32).copy() / 255.0).permute(2, 0, 1).unsqueeze(0)
    m = torch.from_numpy(np.asarray(mask, dtype=np.float32).copy() / 255.0).unsqueeze(0).unsqueeze(0)

    long_side = max(m.shape[-2:])
    low_h = max(1, round(m.shape[-2] * 64 / long_side))
    low_w = max(1, round(m.shape[-1] * 64 / long_side))
    m_low = F.interpolate(m, size=(low_h, low_w), mode="bilinear", align_corners=False)
    if dilation > 0:
        m_low = F.max_pool2d(m_low, kernel_size=dilation * 2 + 1, stride=1, padding=dilation)
    m = F.interpolate(m_low, size=m.shape[-2:], mode="bilinear", align_corners=False).clamp_(0, 1)

    orig_pyr, gen_pyr, mask_pyr = [orig], [gen], [m]
    for _ in range(levels):
        h, w = orig_pyr[-1].shape[-2:]
        if min(h, w) < 2:
            break
        next_size = (max(1, h // 2), max(1, w // 2))
        orig_pyr.append(F.interpolate(orig_pyr[-1], size=next_size, mode="bilinear", align_corners=False))
        gen_pyr.append(F.interpolate(gen_pyr[-1], size=next_size, mode="bilinear", align_corners=False))
        mask_pyr.append(F.interpolate(mask_pyr[-1], size=next_size, mode="bilinear", align_corners=False))

    orig_lap, gen_lap = [], []
    for index in range(len(orig_pyr) - 1):
        orig_lap.append(orig_pyr[index] - F.interpolate(orig_pyr[index + 1], size=orig_pyr[index].shape[-2:], mode="bilinear", align_corners=False))
        gen_lap.append(gen_pyr[index] - F.interpolate(gen_pyr[index + 1], size=gen_pyr[index].shape[-2:], mode="bilinear", align_corners=False))
    orig_lap.append(orig_pyr[-1])
    gen_lap.append(gen_pyr[-1])

    result = gen_lap[-1] * mask_pyr[-1] + orig_lap[-1] * (1 - mask_pyr[-1])
    for index in range(len(orig_lap) - 2, -1, -1):
        residual = gen_lap[index] * mask_pyr[index] + orig_lap[index] * (1 - mask_pyr[index])
        result = F.interpolate(result, size=residual.shape[-2:], mode="bilinear", align_corners=False) + residual

    image = (result[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def blend_generated_video(source_path: str, mask_path: str, generated_path: str, output_path: str, dilation: int = 6) -> None:
    """Composite generated pixels only inside the white mask and retain audio."""
    import av
    from PIL import Image, ImageOps

    original = av.open(source_path)
    generated = av.open(generated_path)
    mask_container = av.open(mask_path)
    original_stream = original.streams.video[0]
    generated_stream = generated.streams.video[0]
    mask_stream = mask_container.streams.video[0]
    fps = generated_stream.average_rate or original_stream.average_rate or Fraction(24, 1)

    output = av.open(output_path, mode="w")
    video_out = output.add_stream("libx264", rate=fps)
    # The ComfyUI workflow saves at its generated canvas resolution. Resize the
    # source and mask to the same canvas so all three inputs stay pixel-aligned.
    output_width = generated_stream.width or original_stream.width
    output_height = generated_stream.height or original_stream.height
    print(f"  Blend canvas: {output_width}x{output_height} (generated output resolution)")
    video_out.width = output_width
    video_out.height = output_height
    pix_fmt = "yuv420p" if output_width % 2 == 0 and output_height % 2 == 0 else "yuv444p"
    video_out.pix_fmt = pix_fmt
    video_out.options = {"bf": "0"}

    source_audio = next((stream for stream in original.streams if stream.type == "audio"), None)
    generated_audio = next((stream for stream in generated.streams if stream.type == "audio"), None)
    audio_input = source_audio if source_audio is not None else generated_audio
    audio_out = output.add_stream_from_template(audio_input) if audio_input else None
    audio_packets = []
    if audio_input and audio_out:
        audio_container = original if source_audio else generated
        for packet in audio_container.demux(audio_input):
            if packet.dts is not None:
                copied = av.Packet(bytes(packet))
                copied.pts, copied.dts, copied.duration = packet.pts, packet.dts, packet.duration
                copied.time_base = packet.time_base
                copied.stream = audio_out
                timestamp = float(packet.dts * packet.time_base) if packet.time_base else 0.0
                audio_packets.append((timestamp, copied))
        audio_container.seek(0)

    original_frames = iter(original.decode(video=0))
    mask_frames = iter(mask_container.decode(video=0))
    single_mask = mask_stream.frames == 1
    first_mask = next(mask_frames, None)
    if first_mask is None:
        raise ValueError(f"Mask video contains no frames: {mask_path}")
    frame_index = 0
    audio_index = 0
    audio_packets.sort(key=lambda item: item[0])
    try:
        for generated_frame in generated.decode(video=0):
            original_frame = next(original_frames, None)
            mask_frame = first_mask if single_mask else (first_mask if frame_index == 0 else next(mask_frames, None))
            if original_frame is None or mask_frame is None:
                break

            original_image = original_frame.to_image().convert("RGB")
            source_size = original_image.size
            if original_image.size != (output_width, output_height):
                original_image = ImageOps.fit(
                    original_image,
                    (output_width, output_height),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )
            generated_image = generated_frame.to_image().convert("RGB").resize(
                (output_width, output_height), Image.Resampling.LANCZOS
            )
            mask_image = mask_frame.to_image().convert("L")
            if mask_image.size != source_size:
                mask_image = mask_image.resize(source_size, Image.Resampling.NEAREST)
            mask_image = ImageOps.fit(
                mask_image,
                (output_width, output_height),
                method=Image.Resampling.BILINEAR,
                centering=(0.5, 0.5),
            )
            blended = _laplacian_blend_frame(original_image, generated_image, mask_image, dilation=dilation)

            frame_time = frame_index / float(fps)
            while audio_index < len(audio_packets) and audio_packets[audio_index][0] <= frame_time:
                output.mux(audio_packets[audio_index][1])
                audio_index += 1

            video_frame = av.VideoFrame.from_image(blended).reformat(format=pix_fmt)
            video_frame.pts = frame_index
            video_frame.time_base = Fraction(fps.denominator, fps.numerator)
            for packet in video_out.encode(video_frame):
                output.mux(packet)
            frame_index += 1

        for packet in video_out.encode():
            output.mux(packet)
        final_video_time = frame_index / float(fps)
        while audio_index < len(audio_packets) and audio_packets[audio_index][0] <= final_video_time:
            output.mux(audio_packets[audio_index][1])
            audio_index += 1
    finally:
        output.close()
        original.close()
        generated.close()
        mask_container.close()


def nearest_higher_ltx_canvas(width: int, height: int) -> tuple[int, int]:
    """Choose a close-aspect /64 canvas with neither axis below the source."""
    min_width_units = max(1, math.ceil(width / 64))
    min_height_units = max(1, math.ceil(height / 64))
    source_ratio = width / height
    candidates = [
        (w_units * 64, h_units * 64)
        for w_units in range(min_width_units, min_width_units + 9)
        for h_units in range(min_height_units, min_height_units + 9)
    ]

    def score(canvas: tuple[int, int]) -> float:
        canvas_width, canvas_height = canvas
        ratio_error = abs(math.log((canvas_width / canvas_height) / source_ratio))
        size_error = ((canvas_width - width) / width) ** 2 + ((canvas_height - height) / height) ** 2
        return ratio_error * 2 + size_error

    return min(candidates, key=score)


def validate_video_pair(args: argparse.Namespace) -> None:
    """Read media metadata and choose the nearest LTX-compatible canvas."""
    src_info = get_video_info(args.source)
    mask_info = get_video_info(args.mask)

    if src_info:
        if src_info.get("width") and src_info.get("height"):
            args.source_width = src_info["width"]
            args.source_height = src_info["height"]
            args.output_width, args.output_height = nearest_higher_ltx_canvas(
                args.source_width, args.source_height
            )
            print(f"  LTX canvas:   {args.output_width}x{args.output_height} "
                  f"(closest higher /64 canvas; centered aspect crop for alignment; "
                  f"source {args.source_width}x{args.source_height})")
        args.source_fps = src_info.get("fps")
        args.source_frames = src_info.get("frames")
        if not args.source_frames and src_info.get("duration") and args.source_fps:
            args.source_frames = round(src_info["duration"] * args.source_fps)
        if args.source_frames:
            # LTX video lengths use the 8k+1 frame grid.
            args.source_frames = max(1, ((args.source_frames - 1) // 8) * 8 + 1)

    if src_info and mask_info:
        if src_info.get("width") and mask_info.get("width"):
            if src_info["width"] != mask_info["width"] or src_info["height"] != mask_info["height"]:
                print(f"  ⚠ WARNING: Resolution mismatch!")
                print(f"    Source: {src_info['width']}x{src_info['height']}")
                print(f"    Mask:   {mask_info['width']}x{mask_info['height']}")
                print(f"    The mask should match the source video resolution.")

        print(f"  Source video: {src_info.get('width', '?')}x{src_info.get('height', '?')}, "
              f"fps={src_info.get('fps', '?')}")
        print(f"  Mask video:   {mask_info.get('width', '?')}x{mask_info.get('height', '?')}, "
              f"fps={mask_info.get('fps', '?')}")


# ============================================================================
# Pipeline runner
# ============================================================================

def build_command(args: argparse.Namespace) -> list[str]:
    """Build the ltx_pipelines.ic_lora command-line arguments."""
    # Select video VAE variant (uses validated args.use_conv_vae)
    vae_key = "video_vae_conv" if args.use_conv_vae else "video_vae"

    cmd = [
        sys.executable, "-m", "ltx_pipelines.ic_lora",
        # --- Model paths ---
        "--transformer-path", str(MODEL_PATHS["transformer"]),
        "--text-encoder-path", str(MODEL_PATHS["text_encoder"]),
        "--video-vae-path", str(MODEL_PATHS[vae_key]),
        "--audio-vae-path", str(MODEL_PATHS["audio_vae"]),
        "--spatial-upsampler-path", str(MODEL_PATHS["spatial_upsampler"]),
        # --- IC-LoRA adapter ---
        "--lora", str(MODEL_PATHS["ic_lora"]), str(args.lora_strength),
        # --- Conditioning inputs ---
        "--video-conditioning", str(Path(args.conditioning_reference).resolve()), str(args.conditioning_strength),
        # --- Generation settings ---
        "--prompt", args.prompt,
        "--output-path", str(Path(args.pipeline_output).resolve()),
    ]

    if getattr(args, "output_width", None) and getattr(args, "output_height", None):
        cmd.extend(["--width", str(args.output_width), "--height", str(args.output_height)])
    if getattr(args, "source_frames", None):
        cmd.extend(["--num-frames", str(args.source_frames)])

    # Optional: seed for reproducibility
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    # Optional: quantization (cuts model VRAM in half)
    if args.quantization:
        cmd.extend(["--quantization", args.quantization])

    # Optional: CPU offloading (saves ~14 GB VRAM)
    offload_mode = args.offload
    if not offload_mode and args.low_vram:
        offload_mode = "cpu"
    if offload_mode:
        if offload_mode.lower() in ("cpu", "offloadmode.cpu"):
            offload_mode = "cpu"
        elif offload_mode.lower() in ("none", "offloadmode.none"):
            offload_mode = "none"
        elif offload_mode.lower() in ("disk", "offloadmode.disk"):
            offload_mode = "disk"
        cmd.extend(["--offload", offload_mode])

    # Explicit option only: low-VRAM mode keeps both generation stages enabled.
    if args.skip_stage_2:
        cmd.append("--skip-stage-2")

    # Optional: tiling (saves VRAM for high-res inputs)
    if args.tile or args.low_vram:
        cmd.append("--tile")

    # Tiled full-resolution stage 2 requires IC-LoRA guidance in current LTX pipeline.
    if args.stage_2_ic_lora or args.low_vram:
        cmd.append("--stage-2-ic-lora")

    return cmd


def run_inpainting(args: argparse.Namespace) -> None:
    """Execute the LTX IC-LoRA inpainting pipeline."""

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path = _temporary_mp4("ltx_green_inpaint_reference_")
    pipeline_output_path = _temporary_mp4("ltx_generated_unblended_")
    args.conditioning_reference = str(reference_path)
    args.pipeline_output = str(pipeline_output_path)
    print("  Preparing green-mask IC-LoRA reference (white mask = replace)...")
    try:
        prepare_inpaint_reference(
            args.source,
            args.mask,
            reference_path,
            width=args.output_width,
            height=args.output_height,
            dilation=args.mask_dilation,
        )
    except Exception:
        reference_path.unlink(missing_ok=True)
        pipeline_output_path.unlink(missing_ok=True)
        raise
    # Build and display command
    cmd = build_command(args)

    print("\n" + "="*60)
    print("  LTX-2.5 IC-LoRA Inpainting — Starting")
    print("="*60)
    print(f"\n  Source:    {args.source}")
    print(f"  Mask:      {args.mask}")
    print(f"  Prompt:    \"{args.prompt}\"")
    print(f"  Output:    {args.output}")
    print(f"  LoRA:      strength={args.lora_strength}")
    print("  Mask:      white=replace, black=preserve; source mask passed unchanged")
    print(f"  Blending:  Laplacian pyramid, mask dilation={args.blend_dilation}")
    print(f"  VAE:       {'conv (fast)' if args.use_conv_vae else 'diffusion (quality)'}")
    if args.low_vram:
        print("  Mode:      LOW VRAM TWO-STAGE (CPU offload + tile + conv VAE)")
    if args.quantization:
        print(f"  Quantize:  {args.quantization}")
    if args.offload:
        print(f"  Offload:   {args.offload}")
    if args.skip_stage_2:
        if getattr(args, "output_width", None) and getattr(args, "output_height", None):
            print(f"  Resolution: {args.output_width // 2}x{args.output_height // 2} "
                  f"(half-resolution because stage 2 is skipped)")
        print(f"  Stage 2:   SKIPPED (half-res output, saves VRAM)")
    if args.tile or args.low_vram:
        print(f"  Tiling:    ENABLED (lower VRAM)")

    print(f"\n  Full command:")
    # Print command in a readable way
    cmd_str = cmd[0]
    for part in cmd[1:]:
        if part.startswith("--"):
            cmd_str += f" \\\n    {part}"
        else:
            cmd_str += f" {part}"
    print(f"  {cmd_str}")

    print(f"\n{'='*60}")

    # Run the pipeline
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(LTX2_REPO_DIR),  # Run from LTX-2 repo root
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=True,
        )
        print("  Blending generated frames into the original through the white mask...")
        blend_generated_video(args.source, args.mask, str(pipeline_output_path), str(output_path), dilation=args.blend_dilation)
        elapsed = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"  ✓ INPAINTING COMPLETE")
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Output saved to: {args.output}")
        print(f"{'='*60}")

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  ✗ INPAINTING FAILED (exit code {e.returncode})")
        print(f"  Time: {elapsed:.1f}s")
        print(f"{'='*60}")
        print(f"\n  Troubleshooting:")
        print(f"    • Out of VRAM? Try: --skip-stage-2 or --tile or --use-conv-vae")
        print(f"    • Missing models? Run: python setup_and_download.py")
        print(f"    • NATTEN error? Try: --skip-stage-2")
        sys.exit(1)

    except FileNotFoundError:
        print(f"\n  ✗ ERROR: Could not find ltx_pipelines module.")
        print(f"    Make sure you ran setup_and_download.py first.")
        sys.exit(1)
    finally:
        reference_path.unlink(missing_ok=True)
        pipeline_output_path.unlink(missing_ok=True)


# ============================================================================
# CLI Argument Parser
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LTX-2.5 IC-LoRA Video Inpainting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run directly with default inputs (inputs/source_video.mp4 & inputs/mask_video.mp4):
  python run_inpaint.py

  # Run with custom video/mask:
  python run_inpaint.py --source video.mp4 --mask mask.mp4

  # Replace masked area with new content:
  python run_inpaint.py --source video.mp4 --mask mask.mp4 --prompt "continue the wall texture; no logo, text, or watermark"

  # One-click low-VRAM mode (skips upsampling, uses conv VAE & tiling):
  python run_inpaint.py --low-vram

  # Ultra low-VRAM mode (with FP8 quantization + CPU offload):
  python run_inpaint.py --low-vram --quantization fp8-scaled-mm --offload OffloadMode.CPU
        """,
    )

    # Prompts and inputs
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="inputs/source_video.mp4",
        help="Path to the source/input video to inpaint (default: 'inputs/source_video.mp4')",
    )
    parser.add_argument(
        "--mask", "-m",
        type=str,
        default="inputs/mask_video.mp4",
        help="Path to the grayscale mask video (default: 'inputs/mask_video.mp4')",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=("Continue the surrounding scene naturally and seamlessly. Preserve the original camera, "
                 "lighting, colors, motion, and background. Fill the masked area with matching scene texture. "
                 "Remove the logo completely; do not add any logo, brand mark, text, lettering, watermark, "
                 "or graphic. Keep all unmasked content unchanged."),
        help="Text prompt describing what should appear in the masked area. "
             "(default: seamless scene continuation with the logo removed and no text or branding)",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output video path (default: outputs/<source_name>_inpainted.mp4)",
    )

    # IC-LoRA strength
    parser.add_argument(
        "--lora-strength",
        type=float,
        default=1.0,
        help="IC-LoRA adapter weight (0.0-1.0, default: 1.0)",
    )

    # Conditioning strengths
    parser.add_argument(
        "--conditioning-strength",
        type=float,
        default=1.0,
        help="Video conditioning strength (0.0-1.0, default: 1.0)",
    )
    parser.add_argument(
        "--mask-dilation",
        type=int,
        default=5,
        help="Expand the white inpaint region before creating the green IC-LoRA cue (default: 5 pixels)",
    )
    parser.add_argument(
        "--blend-dilation",
        type=int,
        default=6,
        help="Expand the white region at low resolution for Laplacian blending (default: 6)",
    )

    # Generation settings
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of inference steps (default: pipeline default, typically 8 for distilled)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Guidance scale (default: pipeline default)",
    )

    # VRAM / Performance flags
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Two-stage low-VRAM mode (CPU offload, conv VAE, tiled stages; keeps full-resolution stage 2)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["fp8-cast", "fp8-scaled-mm", "nvfp4-cast", "nvfp4-prequant"],
        help="Quantization mode (fp8-scaled-mm or fp8-cast drastically cuts VRAM)",
    )
    parser.add_argument(
        "--offload",
        type=str,
        default=None,
        help="CPU offloading mode to save VRAM (e.g. 'cpu')",
    )
    parser.add_argument(
        "--skip-stage-2",
        action="store_true",
        help="Skip stage 2 upsampling (outputs at half resolution, saves ~8-10 GB VRAM)",
    )
    parser.add_argument(
        "--tile",
        action="store_true",
        help="Enable tiling for transformer steps (saves VRAM for large inputs)",
    )
    parser.add_argument(
        "--stage-2-ic-lora",
        action="store_true",
        help="Keep IC-LoRA active during stage 2 (needed if tiling exceeds window)",
    )
    parser.add_argument(
        "--use-conv-vae",
        action="store_true",
        help="Use convolutional VAE decoder (faster, less VRAM) instead of diffusion decoder",
    )

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        source_stem = Path(args.source).stem
        args.output = str(OUTPUTS_DIR / f"{source_stem}_inpainted.mp4")

    return args


# ============================================================================
# Main
# ============================================================================

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  LTX-2.5 IC-LoRA Video Inpainting — Run                ║")
    print("╚══════════════════════════════════════════════════════════╝")

    args = parse_args()

    # Validate and resolve video inputs
    validate_inputs(args)

    # Ensure output name matches resolved source video
    if args.output is None or args.output.endswith("source_video_inpainted.mp4"):
        source_stem = Path(args.source).stem
        args.output = str(OUTPUTS_DIR / f"{source_stem}_inpainted.mp4")

    validate_video_pair(args)

    # Run
    run_inpainting(args)


if __name__ == "__main__":
    main()
