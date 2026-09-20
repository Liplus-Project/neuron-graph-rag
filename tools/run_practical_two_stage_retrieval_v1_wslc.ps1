param(
    [Parameter(Position = 0)]
    [ValidateSet("audit", "probe", "preflight", "run")]
    [string]$Action = "audit"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$workspace = Split-Path -Parent $root
$experimentRoot = Join-Path $workspace "workspace\experiments\practical_two_stage_retrieval_v1"
$buildContext = Join-Path $experimentRoot "bundle-image-context"
$ceHostCache = Join-Path $experimentRoot "model-cache"
$e5HostCache = Join-Path $workspace ".semantic230-model"
$volume = "github-practical-two-stage-retrieval-v1-runtime"
$containerRoot = "/opt/ngr-practical-two-stage-v1"
$containerSource = "$containerRoot/source"
$containerE5Cache = "$containerRoot/e5-cache"
$containerCECache = "$containerRoot/ce-cache"
$containerRun = "$containerRoot/run"
$e5Image = "ngr-e5-structural-centroid-ablation-v1:freeze"
$ceImage = "ngr-cross-encoder-precision-v8:freeze"
$module = "neuron_graph_rag.practical_two_stage_retrieval"

function Invoke-Checked {
    param([string[]]$Command)
    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "command failed with exit code ${LASTEXITCODE}: $($Command -join ' ')"
    }
}

function Base-Arguments {
    param([string]$Image, [string]$Memory)
    return @(
        "run", "--rm", "--network", "none", "--cpus", "4", "--memory", $Memory,
        "--volume", "${volume}:${containerRoot}",
        "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONPATH=${containerSource}/src",
        "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
        "--env", "NO_PROXY=*", "--env", "OMP_NUM_THREADS=4", "--env", "MKL_NUM_THREADS=4",
        "--workdir", $containerSource, "--entrypoint", "python", $Image
    )
}

function Export-Artifact {
    param([string]$ContainerPath, [string]$HostPath)
    $encoded = (& wslc run --rm --network none --volume "${volume}:${containerRoot}:ro" --entrypoint /usr/bin/base64 $ceImage -w 0 $ContainerPath) -join ""
    if ($LASTEXITCODE -ne 0) { throw "could not export artifact: $ContainerPath" }
    $directory = Split-Path -Parent $HostPath
    New-Item -ItemType Directory -Force $directory | Out-Null
    if (Test-Path -LiteralPath $HostPath) { throw "append-only host artifact exists: $HostPath" }
    [System.IO.File]::WriteAllBytes($HostPath, [System.Convert]::FromBase64String($encoded))
}

if ($Action -in @("audit", "probe")) {
    $env:PYTHONPATH = Join-Path $root "src"
    $env:PYTHONUTF8 = "1"
    Invoke-Checked -Command @($python, "-m", $module, $Action, "--root", $root)
    exit 0
}

$status = (& git -C $root status --porcelain=v1)
if ($LASTEXITCODE -ne 0 -or $status) { throw "registered protocol requires a clean committed source tree" }
$sourceCommit = (& git -C $root rev-parse HEAD).Trim()
if ($sourceCommit -notmatch "^[0-9a-f]{40}$") { throw "cannot resolve source identity" }

if ($Action -eq "preflight") {
    foreach ($required in @($e5HostCache, $ceHostCache)) {
        if (-not (Test-Path -LiteralPath $required -PathType Container)) { throw "pinned model cache unavailable: $required" }
    }
    & wslc volume inspect $volume *> $null
    if ($LASTEXITCODE -eq 0) { throw "fresh runtime volume already exists: $volume" }

    $resolvedExperiment = [System.IO.Path]::GetFullPath($experimentRoot)
    $resolvedContext = [System.IO.Path]::GetFullPath($buildContext)
    if (-not $resolvedContext.StartsWith($resolvedExperiment + [System.IO.Path]::DirectorySeparatorChar)) {
        throw "unsafe build-context path: $resolvedContext"
    }
    if (Test-Path -LiteralPath $resolvedContext) { Remove-Item -LiteralPath $resolvedContext -Recurse -Force }
    New-Item -ItemType Directory -Force (Join-Path $resolvedContext "source") | Out-Null

    $sourceMap = [ordered]@{
        "src\neuron_graph_rag\__init__.py" = "src/neuron_graph_rag/__init__.py"
        "src\neuron_graph_rag\cross_encoder_precision_v2_evaluation.py" = "src/neuron_graph_rag/cross_encoder_precision_v2_evaluation.py"
        "src\neuron_graph_rag\e5_structural_centroid_ablation.py" = "src/neuron_graph_rag/e5_structural_centroid_ablation.py"
        "src\neuron_graph_rag\full_corpus_rerank_oracle.py" = "src/neuron_graph_rag/full_corpus_rerank_oracle.py"
        "src\neuron_graph_rag\full_corpus_rerank_oracle_v2.py" = "src/neuron_graph_rag/full_corpus_rerank_oracle_v2.py"
        "src\neuron_graph_rag\practical_two_stage_retrieval.py" = "src/neuron_graph_rag/practical_two_stage_retrieval.py"
        "src\neuron_graph_rag\structural_representation_length_bias_ablation.py" = "src/neuron_graph_rag/structural_representation_length_bias_ablation.py"
        "src\neuron_graph_rag\structural_representation_length_bias_diagnostic.py" = "src/neuron_graph_rag/structural_representation_length_bias_diagnostic.py"
        "tests\evidence\practical_two_stage_retrieval_v1\result_free_minilm_parity.json" = "tests/evidence/practical_two_stage_retrieval_v1/result_free_minilm_parity.json"
        "tests\fixtures\e5_structural_centroid_ablation_v1.model.json" = "tests/fixtures/e5_structural_centroid_ablation_v1.model.json"
        "tests\fixtures\full_corpus_rerank_oracle_v2.query.json" = "tests/fixtures/full_corpus_rerank_oracle_v2.query.json"
        "tests\fixtures\github_retrieval_parity_v4.corpus.json" = "tests/fixtures/github_retrieval_parity_v4.corpus.json"
        "tests\fixtures\practical_two_stage_retrieval_v1.manifest.json" = "tests/fixtures/practical_two_stage_retrieval_v1.manifest.json"
        "tests\fixtures\practical_two_stage_retrieval_v1.models.json" = "tests/fixtures/practical_two_stage_retrieval_v1.models.json"
        "tests\fixtures\practical_two_stage_retrieval_v1.schema.json" = "tests/fixtures/practical_two_stage_retrieval_v1.schema.json"
    }
    foreach ($entry in $sourceMap.GetEnumerator()) {
        $from = Join-Path $root $entry.Key
        $to = Join-Path (Join-Path $resolvedContext "source") $entry.Value
        New-Item -ItemType Directory -Force (Split-Path -Parent $to) | Out-Null
        Copy-Item -LiteralPath $from -Destination $to
    }
    $e5Snapshot = "models--intfloat--multilingual-e5-small\snapshots\614241f622f53c4eeff9890bdc4f31cfecc418b3"
    foreach ($relative in @("config.json", "onnx\model.onnx", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json")) {
        $from = Join-Path $e5HostCache "$e5Snapshot\$relative"
        $to = Join-Path $resolvedContext "e5-cache\$e5Snapshot\$relative"
        New-Item -ItemType Directory -Force (Split-Path -Parent $to) | Out-Null
        Copy-Item -LiteralPath $from -Destination $to
    }
    foreach ($snapshot in @(
        "models--cross-encoder--ms-marco-MiniLM-L6-v2\snapshots\233902d25c440f23af6f7d6e94d2946bac0bee0a",
        "models--BAAI--bge-reranker-v2-m3\snapshots\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    )) {
        Get-ChildItem -LiteralPath (Join-Path $ceHostCache $snapshot) -File | ForEach-Object {
            $to = Join-Path $resolvedContext "ce-cache\$snapshot\$($_.Name)"
            New-Item -ItemType Directory -Force (Split-Path -Parent $to) | Out-Null
            Copy-Item -LiteralPath $_.FullName -Destination $to
        }
    }
    Invoke-Checked -Command @("wslc", "volume", "create", $volume)
    $stream = "tar -cf - -C $resolvedContext source e5-cache ce-cache | wslc run --rm --interactive --network none --volume ${volume}:${containerRoot} --entrypoint /bin/tar $ceImage -xf - -C $containerRoot"
    & cmd.exe /d /c $stream
    if ($LASTEXITCODE -ne 0) { throw "could not stream the allowlisted bundle into the fresh runtime volume" }
    Invoke-Checked -Command (@("wslc") + (Base-Arguments -Image $e5Image -Memory "6G") + @("-m", $module, "preflight-e5", "--root", $containerSource, "--cache", $containerE5Cache, "--runtime-root", $containerRun, "--source-commit", $sourceCommit))
    Invoke-Checked -Command (@("wslc") + (Base-Arguments -Image $ceImage -Memory "8G") + @("-m", $module, "preflight-ce", "--root", $containerSource, "--cache", $containerCECache, "--runtime-root", $containerRun, "--source-commit", $sourceCommit))
    Invoke-Checked -Command (@("wslc") + (Base-Arguments -Image $ceImage -Memory "4G") + @("-m", $module, "preflight-bind", "--root", $containerSource, "--runtime-root", $containerRun, "--source-commit", $sourceCommit))
    exit 0
}

& wslc volume inspect $volume *> $null
if ($LASTEXITCODE -ne 0) { throw "preflight runtime volume unavailable: $volume" }
$ceArguments = Base-Arguments -Image $ceImage -Memory "8G"
try {
    Invoke-Checked -Command (@("wslc") + $ceArguments + @("-m", $module, "claim", "--root", $containerSource, "--runtime-root", $containerRun, "--source-commit", $sourceCommit))
    Invoke-Checked -Command (@("wslc") + (Base-Arguments -Image $e5Image -Memory "6G") + @("-m", $module, "stage1", "--root", $containerSource, "--cache", $containerE5Cache, "--runtime-root", $containerRun))
    foreach ($kind in @("minilm", "v2-m3")) {
        Invoke-Checked -Command (@("wslc") + $ceArguments + @("-m", $module, "stage2", "--root", $containerSource, "--cache", $containerCECache, "--runtime-root", $containerRun, "--kind", $kind))
    }
    $goldDirectory = Join-Path $root "tests\fixtures"
    $copyGold = "tar -cf - -C $goldDirectory full_corpus_rerank_oracle_v2.gold.json | wslc run --rm --interactive --network none --volume ${volume}:${containerSource}/tests/fixtures --entrypoint /bin/tar $ceImage -xf - -C ${containerSource}/tests/fixtures"
    & cmd.exe /d /c $copyGold
    if ($LASTEXITCODE -ne 0) { throw "could not stream development-only gold after worker completion" }
    Invoke-Checked -Command (@("wslc") + $ceArguments + @("-m", $module, "finalize", "--root", $containerSource, "--runtime-root", $containerRun, "--source-commit", $sourceCommit))
}
catch {
    try {
        Invoke-Checked -Command (@("wslc") + $ceArguments + @("-m", $module, "record-error", "--root", $containerSource, "--source-commit", $sourceCommit, "--message", $_.Exception.Message))
    }
    catch { Write-Warning "could not append failure evidence: $($_.Exception.Message)" }
    throw
}
finally {
    $evidenceRoot = Join-Path $root "tests\evidence\practical_two_stage_retrieval_v1"
    $exports = [ordered]@{
        "${containerRun}/preflight.attestation.json" = (Join-Path $evidenceRoot "development.preflight.json")
        "${containerSource}/tests/evidence/practical_two_stage_retrieval_v1/development.claim.json" = (Join-Path $evidenceRoot "development.claim.json")
        "${containerSource}/tests/evidence/practical_two_stage_retrieval_v1/development.observed.json" = (Join-Path $evidenceRoot "development.observed.json")
        "${containerSource}/tests/evidence/practical_two_stage_retrieval_v1/development.error.json" = (Join-Path $evidenceRoot "development.error.json")
    }
    foreach ($entry in $exports.GetEnumerator()) {
        & wslc run --rm --network none --volume "${volume}:${containerRoot}:ro" --entrypoint /bin/test $ceImage -f $entry.Key
        if ($LASTEXITCODE -eq 0) { Export-Artifact -ContainerPath $entry.Key -HostPath $entry.Value }
    }
}
