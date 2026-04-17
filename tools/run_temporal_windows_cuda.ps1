param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("fresh", "resume")]
    [string]$Mode,

    [Parameter(Mandatory = $true)]
    [string]$InputWav,

    [Parameter(Mandatory = $true)]
    [string]$OutputWav,

    [Parameter(Mandatory = $true)]
    [string]$OutDir,

    [string]$RepoRoot = ".",
    [string]$PythonExe = "python",
    [int]$Steps = 50000,
    [int]$BatchSize = 8,
    [int]$Context = 8192,
    [int]$Target = 4096,
    [int]$EpochSteps = 2000,
    [int]$ValCheckInterval = 250,
    [int]$CheckpointEvery = 500,
    [int]$PreviewEvery = 2000,
    [int]$LogEvery = 50,
    [double]$LearningRate = 0.0003,
    [double]$EsrWeight = 0.25,
    [double]$MrstftWeight = 0.0002,
    [double]$EsrDenominatorFloor = 1e-8,
    [int]$HiddenSize = 64,
    [int]$TrainBurnIn = 1024,
    [int]$TrainTruncate = 4096,
    [int]$NumWorkers = 4,
    [int]$PrefetchFactor = 2,
    [string]$Precision = "16-mixed",
    [double]$ActiveSamplingRatio = 0.7,
    [double]$ActiveRmsQuantile = 0.8,
    [switch]$ValidationRequireActive,
    [switch]$NoLogger,

    # Alignment / normalization defaults for robust training
    [ValidateSet("none", "global", "piecewise")]
    [string]$AlignmentMode = "global",
    [int]$PiecewiseBlockSamples = 65536,
    [int]$PiecewiseHopSamples = 0,
    [int]$PiecewiseSmoothBlocks = 3,
    [int]$PiecewiseMaxResidualDelaySamples = 512,
    [double]$PiecewiseMinPeakRatio = 1.02,
    [ValidateSet("none", "rms_match", "affine")]
    [string]$NormalizationMode = "none",
    [switch]$RemoveDC,
    [double]$MinAlignmentPeakRatio = 1.25,
    [double]$MaxResidualDelayStdSamples = 4.0,
    [double]$ClipThreshold = 0.999,
    [double]$MaxClipFraction = 0.02,
    [switch]$FailOnQualityGates
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path $RepoRoot).Path
$trainScript = Join-Path $repo "train.py"
$checkpointsDir = Join-Path $OutDir "checkpoints"
$resumeCkpt = Join-Path $checkpointsDir "last.ckpt"

if (-not (Test-Path $trainScript)) {
    throw "Could not find train.py at: $trainScript"
}

if (-not (Test-Path $InputWav)) {
    throw "Input WAV not found: $InputWav"
}

if (-not (Test-Path $OutputWav)) {
    throw "Output WAV not found: $OutputWav"
}

if ($Mode -eq "fresh") {
    if (Test-Path $OutDir) {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $archiveDir = "${OutDir}_archived_$timestamp"
        Write-Host "[info] Archiving existing outdir to: $archiveDir"
        Move-Item -Path $OutDir -Destination $archiveDir
    }
}

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}

$args = @(
    $trainScript
    "--input", $InputWav
    "--output", $OutputWav
    "--outdir", $OutDir
    "--steps", "$Steps"
    "--batch-size", "$BatchSize"
    "--context", "$Context"
    "--target", "$Target"
    "--epoch-steps", "$EpochSteps"
    "--val-check-interval", "$ValCheckInterval"
    "--checkpoint-every", "$CheckpointEvery"
    "--preview-every", "$PreviewEvery"
    "--log-every", "$LogEvery"
    "--learning-rate", "$LearningRate"
    "--esr-weight", "$EsrWeight"
    "--mrstft-weight", "$MrstftWeight"
    "--esr-denominator-floor", "$EsrDenominatorFloor"
    "--hidden-size", "$HiddenSize"
    "--train-burn-in", "$TrainBurnIn"
    "--train-truncate", "$TrainTruncate"
    "--num-workers", "$NumWorkers"
    "--prefetch-factor", "$PrefetchFactor"
    "--precision", $Precision
    "--device", "gpu"
    "--active-sampling-ratio", "$ActiveSamplingRatio"
    "--active-rms-quantile", "$ActiveRmsQuantile"
    "--alignment-mode", $AlignmentMode
    "--piecewise-block-samples", "$PiecewiseBlockSamples"
    "--piecewise-smooth-blocks", "$PiecewiseSmoothBlocks"
    "--piecewise-max-residual-delay-samples", "$PiecewiseMaxResidualDelaySamples"
    "--piecewise-min-peak-ratio", "$PiecewiseMinPeakRatio"
    "--normalization-mode", $NormalizationMode
    "--min-alignment-peak-ratio", "$MinAlignmentPeakRatio"
    "--max-residual-delay-std-samples", "$MaxResidualDelayStdSamples"
    "--clip-threshold", "$ClipThreshold"
    "--max-clip-fraction", "$MaxClipFraction"
)

if ($PiecewiseHopSamples -gt 0) {
    $args += @("--piecewise-hop-samples", "$PiecewiseHopSamples")
}

if ($ValidationRequireActive) {
    $args += "--validation-require-active"
}

if ($NoLogger) {
    $args += "--no-logger"
}

if ($RemoveDC) {
    $args += "--remove-dc"
}

if ($FailOnQualityGates) {
    $args += "--fail-on-quality-gates"
}

if ($Mode -eq "resume") {
    if (-not (Test-Path $resumeCkpt)) {
        throw "Resume checkpoint not found: $resumeCkpt"
    }
    $args += @("--resume", $resumeCkpt)
    Write-Host "[info] Resuming from checkpoint: $resumeCkpt"
} else {
    Write-Host "[info] Starting fresh run in: $OutDir"
}

Write-Host "[cmd] $PythonExe $($args -join ' ')"
Push-Location $repo
try {
    & $PythonExe @args
}
finally {
    Pop-Location
}

