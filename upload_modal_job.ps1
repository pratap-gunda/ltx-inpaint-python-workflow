param(
    [string]$VideoPath = "inputs\source_video.mp4",
    [string]$MaskPath = "inputs\mask_video.mp4",
    [string]$Prompt,
    [string]$JobId
)

$ErrorActionPreference = "Stop"
# PowerShell 7.3+ can turn any native stderr output (including Modal's
# harmless Python deprecation warnings) into a terminating error. Capture it
# so the script can decide success from modal.exe's exit code instead.
$PSNativeCommandUseErrorActionPreference = $false
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not [System.IO.Path]::IsPathRooted($VideoPath)) { $VideoPath = Join-Path $projectRoot $VideoPath }
if (-not [System.IO.Path]::IsPathRooted($MaskPath)) { $MaskPath = Join-Path $projectRoot $MaskPath }
if (-not (Test-Path -LiteralPath $VideoPath -PathType Leaf)) { throw "Video file not found: $VideoPath" }
if (-not (Test-Path -LiteralPath $MaskPath -PathType Leaf)) { throw "Mask file not found: $MaskPath" }
if ([string]::IsNullOrWhiteSpace($Prompt)) { throw "Prompt is required." }
if ($Prompt.Length -gt 4096) { throw "Prompt must be 4096 characters or fewer." }

if ([string]::IsNullOrWhiteSpace($JobId)) { $JobId = [guid]::NewGuid().ToString('N') }
if ($JobId -notmatch '^[A-Za-z0-9_-]{1,80}$') { throw "JobId may contain only letters, numbers, underscores, and hyphens." }

$videoRelativePath = "jobs/$JobId/input/source_video.mp4"
$maskRelativePath = "jobs/$JobId/input/mask_video.mp4"

function Invoke-ModalVolumePut([string]$LocalPath, [string]$VolumePath, [string]$Label) {
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        # Modal writes harmless Python deprecation warnings to stderr. With
        # Stop, PowerShell can throw on that stream before checking exit code.
        $ErrorActionPreference = "Continue"
        $output = & modal volume put ltx-job-files-v2 $LocalPath $VolumePath --force 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    foreach ($line in $output) { Write-Host $line }
    if ($exitCode -ne 0) { throw "$Label upload failed (exit code $exitCode)." }
}

Invoke-ModalVolumePut $VideoPath $videoRelativePath "Video"
Invoke-ModalVolumePut $MaskPath $maskRelativePath "Mask"

$manifestPath = Join-Path $env:TEMP "ltx-job-$JobId.json"
$manifest = [ordered]@{
    job_id = $JobId
    status = "uploaded"
    created_at = [datetime]::UtcNow.ToString("o")
    prompt = $Prompt.Trim()
    original_filenames = @{
        video = [System.IO.Path]::GetFileName($VideoPath)
        mask = [System.IO.Path]::GetFileName($MaskPath)
    }
    input_paths = @{
        video = "/job-files/$videoRelativePath"
        mask = "/job-files/$maskRelativePath"
    }
    output_path = "/job-files/jobs/$JobId/output/inpainted.mp4"
}
$manifestJson = $manifest | ConvertTo-Json -Depth 5
[System.IO.File]::WriteAllText($manifestPath, $manifestJson, [System.Text.UTF8Encoding]::new($false))
try {
    Invoke-ModalVolumePut $manifestPath "jobs/$JobId/job.json" "Job manifest"
}
finally {
    Remove-Item -LiteralPath $manifestPath -Force -ErrorAction SilentlyContinue
}

[pscustomobject]@{
    job_id = $JobId
    video_path = "/job-files/$videoRelativePath"
    mask_path = "/job-files/$maskRelativePath"
    output_path = "/job-files/jobs/$JobId/output/inpainted.mp4"
    volume = "ltx-job-files-v2"
    prompt = $Prompt.Trim()
} | ConvertTo-Json -Compress
