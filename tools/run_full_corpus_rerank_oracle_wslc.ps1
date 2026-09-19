param(
    [Parameter(Position = 0)]
    [ValidateSet("audit", "probe", "preflight", "run")]
    [string]$Action = "audit"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$workspace = Split-Path -Parent $root
$modelCache = Join-Path $workspace "workspace\experiments\github_cross_encoder_precision_v1\model-cache"
$volume = "github-full-corpus-rerank-oracle-v1-runtime"
$containerRoot = "/opt/ngr-oracle-v1"
$containerSource = "$containerRoot/source"
$containerCache = "$containerRoot/model-cache"
$containerRun = "$containerRoot/run"
$image = "ngr-cross-encoder-precision-v8:freeze"

function Invoke-Checked {
    param([string[]]$Command)
    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "command failed with exit code ${LASTEXITCODE}: $($Command -join ' ')"
    }
}

function Base-Container-Arguments {
    return @(
        "run", "--rm", "--network", "none", "--cpus", "4", "--memory", "8G",
        "--volume", "${volume}:${containerRoot}",
        "--env", "PYTHONPATH=${containerSource}/src",
        "--env", "HF_HUB_OFFLINE=1",
        "--env", "TRANSFORMERS_OFFLINE=1",
        "--env", "HF_HOME=${containerCache}",
        "--env", "HF_HUB_CACHE=${containerCache}",
        "--env", "NO_PROXY=*",
        "--env", "OMP_NUM_THREADS=4",
        "--env", "MKL_NUM_THREADS=4",
        "--workdir", $containerSource,
        "--entrypoint", "python",
        $image
    )
}

if ($Action -in @("audit", "probe")) {
    if (-not (Test-Path -LiteralPath $python)) {
        throw "project verification environment is unavailable"
    }
    $env:PYTHONPATH = Join-Path $root "src"
    $env:PYTHONUTF8 = "1"
    Invoke-Checked -Command @(
        $python, "-m", "neuron_graph_rag.full_corpus_rerank_oracle", $Action,
        "--root", $root
    )
    exit 0
}

if ($Action -eq "preflight") {
    if (-not (Test-Path -LiteralPath $modelCache)) {
        throw "pinned model cache is unavailable: $modelCache"
    }
    & git -C $root diff --quiet HEAD --
    if ($LASTEXITCODE -ne 0) {
        throw "preflight requires a clean committed source tree"
    }
    & wslc volume inspect $volume *> $null
    if ($LASTEXITCODE -eq 0) {
        throw "fresh runtime volume already exists: $volume"
    }
    Invoke-Checked -Command @("wslc", "volume", "create", $volume)
    $prepareScript = @(
        "set -eu; "
        "test ! -e '${containerSource}'; test ! -e '${containerCache}'; "
        "mkdir '${containerSource}' '${containerCache}'; "
        "cp -a /input/source/. '${containerSource}/'; "
        "rm -rf '${containerSource}/.git' '${containerSource}/.venv' "
        "'${containerSource}/.ruff_cache' '${containerSource}/dist'; "
        "cp -a /input/model-cache/. '${containerCache}/'; "
        "test -f '${containerSource}/src/neuron_graph_rag/full_corpus_rerank_oracle.py'; "
        "test -d '${containerCache}/models--BAAI--bge-reranker-base'; "
        "test -d '${containerCache}/models--BAAI--bge-reranker-v2-m3'"
    ) -join ""
    $prepare = @(
        "wslc", "run", "--rm", "--network", "none",
        "--volume", "${volume}:${containerRoot}",
        "--volume", "${root}:/input/source:ro",
        "--volume", "${modelCache}:/input/model-cache:ro",
        "--entrypoint", "/bin/sh", $image, "-c",
        $prepareScript
    )
    Invoke-Checked -Command $prepare
    $arguments = Base-Container-Arguments
    Invoke-Checked -Command (@("wslc") + $arguments + @(
        "-m", "neuron_graph_rag.full_corpus_rerank_oracle", "preflight",
        "--root", $containerSource, "--cache", $containerCache
    ))
    exit 0
}

& wslc volume inspect $volume *> $null
if ($LASTEXITCODE -ne 0) {
    throw "preflight runtime volume is unavailable: $volume"
}
$sourceCommit = (& git -C $root rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $sourceCommit -notmatch "^[0-9a-f]{40}$") {
    throw "cannot resolve the committed oracle source identity"
}
& git -C $root diff --quiet HEAD --
if ($LASTEXITCODE -ne 0) {
    throw "registered run requires the same clean committed source tree as preflight"
}
$arguments = Base-Container-Arguments
Invoke-Checked -Command (@("wslc") + $arguments + @(
    "-m", "neuron_graph_rag.full_corpus_rerank_oracle", "run",
    "--root", $containerSource,
    "--cache", $containerCache,
    "--runtime-root", $containerRun,
    "--source-commit", $sourceCommit
))

$exportScript = @(
    "set -eu; "
    "test -f '${containerSource}/tests/evidence/full_corpus_rerank_oracle_v1/development.claim.json'; "
    "test -f '${containerSource}/tests/evidence/full_corpus_rerank_oracle_v1/development.observed.json'; "
    "test ! -e '/output/tests/evidence/full_corpus_rerank_oracle_v1/development.claim.json'; "
    "test ! -e '/output/tests/evidence/full_corpus_rerank_oracle_v1/development.observed.json'; "
    "mkdir -p '/output/tests/evidence/full_corpus_rerank_oracle_v1'; "
    "cp '${containerSource}/tests/evidence/full_corpus_rerank_oracle_v1/development.claim.json' "
    "'/output/tests/evidence/full_corpus_rerank_oracle_v1/development.claim.json'; "
    "cp '${containerSource}/tests/evidence/full_corpus_rerank_oracle_v1/development.observed.json' "
    "'/output/tests/evidence/full_corpus_rerank_oracle_v1/development.observed.json'"
) -join ""
$export = @(
    "wslc", "run", "--rm", "--network", "none",
    "--volume", "${volume}:${containerRoot}:ro",
    "--volume", "${root}:/output",
    "--entrypoint", "/bin/sh", $image, "-c",
    $exportScript
)
Invoke-Checked -Command $export
