# Modal worker (separate from RunPod)

This is a parallel Modal deployment. It does not replace or modify the existing
RunPod Dockerfile, handler, endpoint, or upload_job.ps1 workflow.

## Resources

- Modal App name: ltx-inpaint-modal
- Modal image definition: Dockerfile.modal
- Model weights and Hugging Face cache: Modal Volume ltx-model-cache, mounted at /models
- Test inputs and outputs: Modal Volume v2 ltx-job-files-v2, mounted at /job-files
- Hugging Face token: existing Modal Secret huggingface-secret, with a key named HF_TOKEN
- GPU default: RTX-PRO-6000; set MODAL_GPU to override it

The worker uses the convolutional VAE. Its default mode is standard (no
low-VRAM offloading or tiling); pass `--mode low-vram` when you want that path.
The first job downloads missing models into ltx-model-cache and commits that
volume before it starts inference. Future jobs reuse those weights.

## Create a test job and upload its files

Create the job-files v2 volume once:

    modal volume create ltx-job-files-v2 --version=2

The script creates a unique job ID and uploads the local test video and mask
and a required user prompt to separate paths under that job in the Modal volume:

    $job = .\upload_modal_job.ps1 -Prompt "Continue the background texture naturally and remove the small corner logo." | ConvertFrom-Json
    $job | Format-List

By default, it uploads inputs/source_video.mp4 and inputs/mask_video.mp4 to
jobs/<job-id>/input/source_video.mp4 and jobs/<job-id>/input/mask_video.mp4.
You can pass different local files with -VideoPath and -MaskPath. This Modal
test uses the ltx-job-files-v2 Volume and does not modify the RunPod workflow.

## Run the inpainting job

Run the job using the paths returned by the upload script. The worker reads the
prompt from the job manifest created during upload:

    modal run modal_app.py --job-id "$($job.job_id)" --video-path "$($job.video_path)" --mask-path "$($job.mask_path)"

The function validates that both inputs are inside that job's input folder.
It writes the result to /job-files/jobs/<job-id>/output/inpainted.mp4 in the
ltx-job-files-v2 volume.

## Download the result locally

Use the generated job ID:

    modal volume get ltx-job-files-v2 "jobs/$($job.job_id)/output/inpainted.mp4" "outputs/$($job.job_id)_inpainted.mp4"

The local output will be saved under the project outputs directory.
To override the saved prompt for a manual run, add `--prompt "your replacement description"`.

The Modal function can run up to five containers concurrently, with one GPU
and one video job per container. Additional jobs sent to the same deployed
function wait in Modal's input queue until a container is free. This allows up
to five users' jobs to generate at the same time when five GPUs are available;
GPU availability or workspace limits can mean fewer than five run at once.
The `modal run` command above is a one-job development/test flow. For user
traffic, deploy the app and route all submissions to that same deployed
function; separate ephemeral `modal run` invocations do not share one global
five-worker queue.

## Job-file retention

The deployed app includes a lightweight scheduled cleanup function. It runs
daily at 04:00 UTC and removes a job directory, including its inputs, output,
and manifest, once its `created_at` is more than 30 days old. It only scans
`/job-files/jobs`; model weights are in a separate Volume and are not touched.
Recent queued/running jobs are retained; a running job's retention age starts
when processing begins. Stale queued/running jobs older than 30 days are also
removed. Jobs with malformed or unknown manifests are retained. An incomplete
upload folder without a manifest expires based on its last modification time.

The cleanup schedule becomes active when this updated app is deployed. To run
the cleanup once manually after deployment:

    modal run modal_app.py::cleanup_expired_jobs

By default, standard mode is enabled. To run low-VRAM mode, add
`--mode low-vram` to the run command above.

Standard mode disables CPU offload and tiling. In the pipeline's standard
defaults, Stage 2 IC-LoRA is also off; low-VRAM mode enables it automatically.

## Website upload API

Deploy `modal_app.py` after creating a Modal Secret named
`ltx-web-upload-auth` with a `WEB_UPLOAD_TOKEN` key. Store the same token in
your website's server-side secrets; never put it in browser JavaScript. Use a
random token with at least 32 characters.

The web job files use a Modal Volume v2 for concurrent uploads and job outputs.
The older `ltx-job-files` test Volume is left untouched.

The deployed API accepts a multipart `POST /api/jobs` with fields named `video`,
`mask`, and `prompt`, plus `Authorization: Bearer <WEB_UPLOAD_TOKEN>`. The user
prompt is required and limited to 4096 characters. It is stored with the job
manifest and used for generation unless processing explicitly overrides it.
The API creates a unique job ID and stores the files under that job in the
Modal Volume. Original filenames are kept as metadata; internal paths are
role-based. The response includes the job ID, both input paths, the future
output path, and status `uploaded`. Uploading does not start GPU inference.

The existing `process_job` function accepts that same job ID and the returned
input paths, then writes the painted video to the returned output path. The
worker reloads the Volume before reading, so it sees files uploaded after its
container started.

### Direct browser uploads to Modal

The Viresto Object Remover now requests a short-lived, single-job upload ticket
from its authenticated Firebase callable. The browser posts the video and PNG
mask directly to `POST /api/jobs/{job_id}/upload` with that ticket. Modal
validates the ticket and limits, then writes both files to the job Volume. The
website calls its authenticated `startObjectRemoverJob` callable afterward;
that function charges the wallet/bundle and asks Modal to start the job. The
permanent Modal service token stays server-side, and the uploaded media never
passes through Firebase Storage.

For static PNG masks, the Modal worker writes a one-frame MP4 at the source
video's resolution and frame rate. The inpainting preprocessor and final blend
reuse that frame for every source frame, avoiding a full-length repeated mask
video.

For this route, Modal's `WEB_UPLOAD_TOKEN` and Firebase's
`MODAL_OBJECT_REMOVER_TOKEN` must have the same value; the existing backend
Modal API calls already depend on that shared credential. The
`/api/jobs/from-urls` route remains available for existing callers.
