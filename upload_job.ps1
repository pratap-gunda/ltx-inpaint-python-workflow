param(
    [string]$VideoPath = "inputs\source_video.mp4",
    [string]$MaskPath = "inputs\mask_video.mp4",
    [string]$JobId
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$envFile = Join-Path $projectRoot ".env"

if (-not (Test-Path -LiteralPath $envFile -PathType Leaf)) {
    throw "Missing .env file: $envFile"
}

function Read-DotEnv([string]$Path) {
    $values = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
            $name = $matches[1]
            $value = $matches[2].Trim()
            if ($value.Length -ge 2 -and $value.StartsWith('"') -and $value.EndsWith('"')) {
                $value = $value.Substring(1, $value.Length - 2)
            } elseif ($value.Length -ge 2 -and $value.StartsWith("'") -and $value.EndsWith("'")) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            $values[$name] = $value
        }
    }
    return $values
}

$config = Read-DotEnv $envFile
$required = @('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'S3_BUCKET', 'S3_REGION', 'S3_ENDPOINT_URL')
foreach ($name in $required) {
    if (-not $config.ContainsKey($name) -or [string]::IsNullOrWhiteSpace($config[$name])) {
        throw "Set $name in $envFile before uploading."
    }
}

if (-not [System.IO.Path]::IsPathRooted($VideoPath)) { $VideoPath = Join-Path $projectRoot $VideoPath }
if (-not [System.IO.Path]::IsPathRooted($MaskPath)) { $MaskPath = Join-Path $projectRoot $MaskPath }
if (-not (Test-Path -LiteralPath $VideoPath -PathType Leaf)) { throw "Video file not found: $VideoPath" }
if (-not (Test-Path -LiteralPath $MaskPath -PathType Leaf)) { throw "Mask file not found: $MaskPath" }

if ([string]::IsNullOrWhiteSpace($JobId)) { $JobId = [guid]::NewGuid().ToString('N') }
if ($JobId -notmatch '^[A-Za-z0-9_-]{1,80}$') { throw "JobId may contain only letters, numbers, underscores, and hyphens." }

$oldAccessKey = $env:AWS_ACCESS_KEY_ID
$oldSecretKey = $env:AWS_SECRET_ACCESS_KEY
$oldDefaultRegion = $env:AWS_DEFAULT_REGION
$oldRegion = $env:AWS_REGION
try {
    $env:AWS_ACCESS_KEY_ID = $config['AWS_ACCESS_KEY_ID']
    $env:AWS_SECRET_ACCESS_KEY = $config['AWS_SECRET_ACCESS_KEY']
    $env:AWS_DEFAULT_REGION = $config['S3_REGION']
    $env:AWS_REGION = $config['S3_REGION']

    $prefix = "s3://$($config['S3_BUCKET'])/jobs/$JobId/input"
    aws s3 cp $VideoPath "$prefix/video.mp4" --region $config['S3_REGION'] --endpoint-url $config['S3_ENDPOINT_URL'] --no-progress
    if ($LASTEXITCODE -ne 0) { throw "Video upload failed (aws exit code $LASTEXITCODE)." }

    aws s3 cp $MaskPath "$prefix/mask.mp4" --region $config['S3_REGION'] --endpoint-url $config['S3_ENDPOINT_URL'] --no-progress
    if ($LASTEXITCODE -ne 0) { throw "Mask upload failed (aws exit code $LASTEXITCODE)." }

    [pscustomobject]@{
        job_id = $JobId
        video_path = "/runpod-volume/jobs/$JobId/input/video.mp4"
        mask_path = "/runpod-volume/jobs/$JobId/input/mask.mp4"
        s3_prefix = $prefix
    } | ConvertTo-Json
} finally {
    $env:AWS_ACCESS_KEY_ID = $oldAccessKey
    $env:AWS_SECRET_ACCESS_KEY = $oldSecretKey
    $env:AWS_DEFAULT_REGION = $oldDefaultRegion
    $env:AWS_REGION = $oldRegion
}
