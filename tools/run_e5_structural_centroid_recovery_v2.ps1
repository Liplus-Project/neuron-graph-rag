param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("prepare", "claim", "finalize", "export", "audit")]
    [string]$Phase,
    [string]$SourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [Parameter(Mandatory = $true)]
    [string]$RegisteredRoot
)

$ErrorActionPreference = "Stop"
$source = [System.IO.Path]::GetFullPath($SourceRoot)
$registered = [System.IO.Path]::GetFullPath($RegisteredRoot)
$claimRelative = "tests/evidence/e5_structural_centroid_finalizer_recovery_v2/development.claim.json"
$resultRelative = "tests/evidence/e5_structural_centroid_finalizer_recovery_v2/development.recovered.json"
$goldRelative = "tests/fixtures/full_corpus_rerank_oracle_v2.gold.json"
$runnerRelative = "src/neuron_graph_rag/e5_structural_centroid_recovery_v2.py"
$claimFiles = @(
    $runnerRelative,
    "src/neuron_graph_rag/e5_structural_centroid_recovery.py",
    "tools/run_e5_structural_centroid_recovery_v2.ps1",
    "tests/evidence/e5_structural_centroid_ablation_v1/development.claim.json",
    "tests/evidence/e5_structural_centroid_ablation_v1/development.error.json",
    "tests/evidence/e5_structural_centroid_ablation_v1/development.preflight.json",
    "tests/evidence/e5_structural_centroid_ablation_v1/development.worker.json",
    "tests/evidence/e5_structural_centroid_finalizer_recovery_v1/development.claim.json",
    "tests/evidence/e5_structural_centroid_finalizer_recovery_v1/development.error.json",
    "tests/fixtures/e5_structural_centroid_ablation_v1.manifest.json",
    "tests/fixtures/e5_structural_centroid_ablation_v1.model.json",
    "tests/fixtures/e5_structural_centroid_ablation_v1.schema.json",
    "tests/fixtures/e5_structural_centroid_recovery_v1.manifest.json",
    "tests/fixtures/e5_structural_centroid_recovery_v1.schema.json",
    "tests/fixtures/e5_structural_centroid_recovery_v2.manifest.json",
    "tests/fixtures/e5_structural_centroid_recovery_v2.schema.json"
)

function Copy-ExclusiveFile([string]$Relative) {
    $from = Join-Path $source $Relative
    $to = Join-Path $registered $Relative
    if (-not (Test-Path -LiteralPath $from -PathType Leaf)) {
        throw "required source file missing: $Relative"
    }
    if (Test-Path -LiteralPath $to) {
        throw "registered file already exists: $Relative"
    }
    $parent = Split-Path -Parent $to
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Copy-Item -LiteralPath $from -Destination $to
}

if ($Phase -eq "prepare") {
    if (Test-Path -LiteralPath $registered) {
        throw "registered root must not exist"
    }
    New-Item -ItemType Directory -Path $registered | Out-Null
    foreach ($relative in $claimFiles) {
        Copy-ExclusiveFile $relative
    }
    Write-Output "prepared gold-absent recovery root: $registered"
    exit 0
}

if (-not (Test-Path -LiteralPath $registered -PathType Container)) {
    throw "registered root missing; run prepare first"
}

$runner = Join-Path $registered $runnerRelative
if ($Phase -eq "claim") {
    if (Test-Path -LiteralPath (Join-Path $registered $goldRelative)) {
        throw "gold must be absent before recovery claim"
    }
    & python $runner claim --root $registered
    if ($LASTEXITCODE -ne 0) { throw "recovery claim failed" }
    exit 0
}

if ($Phase -eq "finalize") {
    if (-not (Test-Path -LiteralPath (Join-Path $registered $claimRelative) -PathType Leaf)) {
        throw "successful recovery claim is required before gold mount"
    }
    Copy-ExclusiveFile $goldRelative
    & python $runner finalize --root $registered
    if ($LASTEXITCODE -ne 0) { throw "recovery finalizer failed" }
    exit 0
}

if ($Phase -eq "audit") {
    & python $runner audit --root $registered
    if ($LASTEXITCODE -ne 0) { throw "recovery audit failed" }
    exit 0
}

foreach ($relative in @($claimRelative, $resultRelative)) {
    $from = Join-Path $registered $relative
    $to = Join-Path $source $relative
    if (-not (Test-Path -LiteralPath $from -PathType Leaf)) {
        throw "registered evidence missing: $relative"
    }
    if (Test-Path -LiteralPath $to) {
        throw "repository evidence already exists: $relative"
    }
    $parent = Split-Path -Parent $to
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Copy-Item -LiteralPath $from -Destination $to
}
Write-Output "exported append-only recovery evidence"
