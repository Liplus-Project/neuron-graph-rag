param(
    [Parameter(Position = 0)]
    [ValidateSet("audit", "probe", "preflight", "run")]
    [string]$Action = "audit"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$workspace = Split-Path -Parent $root
$modelCache = Join-Path $workspace ".semantic230-model"
$volume = "github-e5-structural-centroid-ablation-v1-runtime"
$containerRoot = "/opt/ngr-e5-centroid-v1"
$containerSource = "$containerRoot/source"
$containerCache = "$containerRoot/model-cache"
$containerRun = "$containerRoot/run"
$image = "ngr-e5-structural-centroid-ablation-v1:freeze"

function Invoke-Checked {
    param([string[]]$Command)
    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) { throw "command failed with exit code ${LASTEXITCODE}: $($Command -join ' ')" }
}

function Base-Arguments {
    return @(
        "run", "--rm", "--network", "none", "--cpus", "4", "--memory", "6G",
        "--volume", "${volume}:${containerRoot}",
        "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONPATH=${containerSource}/src",
        "--env", "NO_PROXY=*", "--env", "OMP_NUM_THREADS=4", "--env", "MKL_NUM_THREADS=4",
        "--workdir", $containerSource, "--entrypoint", "python", $image
    )
}

if ($Action -in @("audit", "probe")) {
    $env:PYTHONPATH = Join-Path $root "src"
    $env:PYTHONUTF8 = "1"
    Invoke-Checked -Command @($python, "-m", "neuron_graph_rag.e5_structural_centroid_ablation", $Action, "--root", $root)
    exit 0
}

$status = (& git -C $root status --porcelain=v1)
if ($LASTEXITCODE -ne 0 -or $status) { throw "registered protocol requires a clean committed source tree" }
$sourceCommit = (& git -C $root rev-parse HEAD).Trim()
if ($sourceCommit -notmatch "^[0-9a-f]{40}$") { throw "cannot resolve source identity" }

if ($Action -eq "preflight") {
    if (-not (Test-Path -LiteralPath $modelCache)) { throw "pinned model cache unavailable: $modelCache" }
    & wslc volume inspect $volume *> $null
    if ($LASTEXITCODE -eq 0) { throw "fresh runtime volume already exists: $volume" }
    & wslc image inspect $image *> $null
    if ($LASTEXITCODE -ne 0) {
        Invoke-Checked -Command @("wslc", "build", "--tag", $image, (Join-Path $root "containers\e5_structural_centroid_ablation_v1"))
    }
    Invoke-Checked -Command @("wslc", "volume", "create", $volume)
    $sourceMap = [ordered]@{
        "package" = @("tests\fixtures\full_corpus_rerank_oracle_v2.package_init.py", "src/neuron_graph_rag/__init__.py")
        "package-fixture" = @("tests\fixtures\full_corpus_rerank_oracle_v2.package_init.py", "tests/fixtures/full_corpus_rerank_oracle_v2.package_init.py")
        "projection" = @("src\neuron_graph_rag\cross_encoder_precision_v2_evaluation.py", "src/neuron_graph_rag/cross_encoder_precision_v2_evaluation.py")
        "v1" = @("src\neuron_graph_rag\full_corpus_rerank_oracle.py", "src/neuron_graph_rag/full_corpus_rerank_oracle.py")
        "v2" = @("src\neuron_graph_rag\full_corpus_rerank_oracle_v2.py", "src/neuron_graph_rag/full_corpus_rerank_oracle_v2.py")
        "structural" = @("src\neuron_graph_rag\structural_representation_length_bias_diagnostic.py", "src/neuron_graph_rag/structural_representation_length_bias_diagnostic.py")
        "runner" = @("src\neuron_graph_rag\e5_structural_centroid_ablation.py", "src/neuron_graph_rag/e5_structural_centroid_ablation.py")
        "manifest" = @("tests\fixtures\e5_structural_centroid_ablation_v1.manifest.json", "tests/fixtures/e5_structural_centroid_ablation_v1.manifest.json")
        "schema" = @("tests\fixtures\e5_structural_centroid_ablation_v1.schema.json", "tests/fixtures/e5_structural_centroid_ablation_v1.schema.json")
        "model" = @("tests\fixtures\e5_structural_centroid_ablation_v1.model.json", "tests/fixtures/e5_structural_centroid_ablation_v1.model.json")
        "query" = @("tests\fixtures\full_corpus_rerank_oracle_v2.query.json", "tests/fixtures/full_corpus_rerank_oracle_v2.query.json")
        "corpus" = @("tests\fixtures\github_retrieval_parity_v4.corpus.json", "tests/fixtures/github_retrieval_parity_v4.corpus.json")
        "parity" = @("tests\evidence\e5_structural_centroid_ablation_v1\result_free_parity.json", "tests/evidence/e5_structural_centroid_ablation_v1/result_free_parity.json")
    }
    $snapshot = "models--intfloat--multilingual-e5-small\snapshots\614241f622f53c4eeff9890bdc4f31cfecc418b3"
    $modelRows = @("config.json", "onnx\model.onnx", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json")
    $prepare = @("run", "--rm", "--network", "none", "--volume", "${volume}:${containerRoot}")
    $commands = @("set -eu", "test ! -e '${containerSource}'", "test ! -e '${containerCache}'", "mkdir -p '${containerSource}' '${containerCache}'")
    foreach ($entry in $sourceMap.GetEnumerator()) {
        $hostPath = Join-Path $root $entry.Value[0]
        if (-not (Test-Path -LiteralPath $hostPath -PathType Leaf)) { throw "allowlisted source unavailable: $hostPath" }
        $input = "/input/source-$($entry.Key)"; $prepare += @("--volume", "${hostPath}:${input}:ro")
        $destination = "$containerSource/$($entry.Value[1])"; $directory = $destination.Substring(0, $destination.LastIndexOf('/'))
        $commands += "mkdir -p '$directory'; cp '$input' '$destination'"
    }
    foreach ($relative in $modelRows) {
        $hostPath = Join-Path $modelCache "$snapshot\$relative"
        if (-not (Test-Path -LiteralPath $hostPath -PathType Leaf)) { throw "model file unavailable: $hostPath" }
        $input = "/input/model-$($relative.Replace('\', '-'))"; $prepare += @("--volume", "${hostPath}:${input}:ro")
        $destination = "$containerCache/$($snapshot.Replace('\', '/'))/$($relative.Replace('\', '/'))"; $directory = $destination.Substring(0, $destination.LastIndexOf('/'))
        $commands += "mkdir -p '$directory'; cp -L '$input' '$destination'"
    }
    $prepare += @("--entrypoint", "/bin/sh", $image, "-c", ($commands -join "; "))
    Invoke-Checked -Command (@("wslc") + $prepare)
    Invoke-Checked -Command (@("wslc") + (Base-Arguments) + @("-m", "neuron_graph_rag.e5_structural_centroid_ablation", "preflight", "--root", $containerSource, "--cache", $containerCache, "--runtime-root", $containerRun, "--source-commit", $sourceCommit))
    exit 0
}

& wslc volume inspect $volume *> $null
if ($LASTEXITCODE -ne 0) { throw "preflight runtime volume unavailable: $volume" }
$arguments = Base-Arguments
try {
    Invoke-Checked -Command (@("wslc") + $arguments + @("-m", "neuron_graph_rag.e5_structural_centroid_ablation", "claim", "--root", $containerSource, "--runtime-root", $containerRun, "--source-commit", $sourceCommit))
    Invoke-Checked -Command (@("wslc") + $arguments + @("-m", "neuron_graph_rag.e5_structural_centroid_ablation", "worker", "--root", $containerSource, "--cache", $containerCache, "--runtime-root", $containerRun))
    $goldPath = Join-Path $root "tests\fixtures\full_corpus_rerank_oracle_v2.gold.json"
    $final = $arguments[0..9] + @("--volume", "${goldPath}:${containerSource}/tests/fixtures/full_corpus_rerank_oracle_v2.gold.json:ro") + $arguments[10..($arguments.Length - 1)]
    Invoke-Checked -Command (@("wslc") + $final + @("-m", "neuron_graph_rag.e5_structural_centroid_ablation", "finalize", "--root", $containerSource, "--runtime-root", $containerRun, "--source-commit", $sourceCommit))
} catch {
    try { Invoke-Checked -Command (@("wslc") + $arguments + @("-m", "neuron_graph_rag.e5_structural_centroid_ablation", "record-error", "--root", $containerSource, "--source-commit", $sourceCommit, "--message", $_.Exception.Message)) } catch { Write-Warning "could not append failure evidence" }
    throw
} finally {
    $destination = "/output/tests/evidence/e5_structural_centroid_ablation_v1"
    $export = "set -eu; mkdir -p '$destination'; " +
        "if test -f '${containerRun}/preflight.attestation.json'; then test ! -e '${destination}/development.preflight.json'; cp '${containerRun}/preflight.attestation.json' '${destination}/development.preflight.json'; fi; " +
        "if test -f '${containerSource}/tests/evidence/e5_structural_centroid_ablation_v1/development.claim.json'; then test ! -e '${destination}/development.claim.json'; cp '${containerSource}/tests/evidence/e5_structural_centroid_ablation_v1/development.claim.json' '${destination}/development.claim.json'; fi; " +
        "if test -f '${containerSource}/tests/evidence/e5_structural_centroid_ablation_v1/development.observed.json'; then test ! -e '${destination}/development.observed.json'; cp '${containerSource}/tests/evidence/e5_structural_centroid_ablation_v1/development.observed.json' '${destination}/development.observed.json'; fi; " +
        "if test -f '${containerSource}/tests/evidence/e5_structural_centroid_ablation_v1/development.error.json'; then test ! -e '${destination}/development.error.json'; cp '${containerSource}/tests/evidence/e5_structural_centroid_ablation_v1/development.error.json' '${destination}/development.error.json'; fi"
    Invoke-Checked -Command @("wslc", "run", "--rm", "--network", "none", "--volume", "${volume}:${containerRoot}:ro", "--volume", "${root}:/output", "--entrypoint", "/bin/sh", $image, "-c", $export)
}
