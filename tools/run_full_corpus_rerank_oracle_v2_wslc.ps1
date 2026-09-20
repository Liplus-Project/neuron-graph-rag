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
$volume = "github-full-corpus-rerank-oracle-v2-runtime"
$containerRoot = "/opt/ngr-oracle-v2"
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

function Resolve-SourceCommit {
    $status = (& git -C $root status --porcelain=v1)
    if ($LASTEXITCODE -ne 0 -or $status) {
        throw "registered protocol requires a clean committed source tree"
    }
    $commit = (& git -C $root rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $commit -notmatch "^[0-9a-f]{40}$") {
        throw "cannot resolve the committed oracle source identity"
    }
    return $commit
}

function Base-Container-Arguments {
    param([switch]$ReadOnly)
    $mount = "${volume}:${containerRoot}"
    if ($ReadOnly) {
        $mount = "${mount}:ro"
    }
    return @(
        "run", "--rm", "--network", "none", "--cpus", "4", "--memory", "8G",
        "--volume", $mount,
        "--env", "PYTHONDONTWRITEBYTECODE=1",
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
        $python, "-m", "neuron_graph_rag.full_corpus_rerank_oracle_v2", $Action,
        "--root", $root
    )
    exit 0
}

$sourceCommit = Resolve-SourceCommit

if ($Action -eq "preflight") {
    if (-not (Test-Path -LiteralPath $modelCache)) {
        throw "pinned model cache is unavailable: $modelCache"
    }
    & wslc volume inspect $volume *> $null
    if ($LASTEXITCODE -eq 0) {
        throw "fresh runtime volume already exists: $volume"
    }
    Invoke-Checked -Command @("wslc", "volume", "create", $volume)

    $sourceMap = [ordered]@{
        "package-init" = @("tests\fixtures\full_corpus_rerank_oracle_v2.package_init.py", "src/neuron_graph_rag/__init__.py")
        "projection" = @("src\neuron_graph_rag\cross_encoder_precision_v2_evaluation.py", "src/neuron_graph_rag/cross_encoder_precision_v2_evaluation.py")
        "v1-helpers" = @("src\neuron_graph_rag\full_corpus_rerank_oracle.py", "src/neuron_graph_rag/full_corpus_rerank_oracle.py")
        "v2-runner" = @("src\neuron_graph_rag\full_corpus_rerank_oracle_v2.py", "src/neuron_graph_rag/full_corpus_rerank_oracle_v2.py")
        "manifest" = @("tests\fixtures\full_corpus_rerank_oracle_v2.manifest.json", "tests/fixtures/full_corpus_rerank_oracle_v2.manifest.json")
        "query" = @("tests\fixtures\full_corpus_rerank_oracle_v2.query.json", "tests/fixtures/full_corpus_rerank_oracle_v2.query.json")
        "schema" = @("tests\fixtures\full_corpus_rerank_oracle_v2.schema.json", "tests/fixtures/full_corpus_rerank_oracle_v2.schema.json")
        "corpus" = @("tests\fixtures\github_retrieval_parity_v4.corpus.json", "tests/fixtures/github_retrieval_parity_v4.corpus.json")
        "models" = @("tests\fixtures\github_cross_encoder_precision_v8.models.json", "tests/fixtures/github_cross_encoder_precision_v8.models.json")
    }
    $modelRows = @(
        @("base-config", "models--BAAI--bge-reranker-base\snapshots\2cfc18c9415c912f9d8155881c133215df768a70\config.json"),
        @("base-model", "models--BAAI--bge-reranker-base\snapshots\2cfc18c9415c912f9d8155881c133215df768a70\model.safetensors"),
        @("base-sp", "models--BAAI--bge-reranker-base\snapshots\2cfc18c9415c912f9d8155881c133215df768a70\sentencepiece.bpe.model"),
        @("base-special", "models--BAAI--bge-reranker-base\snapshots\2cfc18c9415c912f9d8155881c133215df768a70\special_tokens_map.json"),
        @("base-tokenizer", "models--BAAI--bge-reranker-base\snapshots\2cfc18c9415c912f9d8155881c133215df768a70\tokenizer.json"),
        @("base-tokenizer-config", "models--BAAI--bge-reranker-base\snapshots\2cfc18c9415c912f9d8155881c133215df768a70\tokenizer_config.json"),
        @("m3-config", "models--BAAI--bge-reranker-v2-m3\snapshots\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e\config.json"),
        @("m3-model", "models--BAAI--bge-reranker-v2-m3\snapshots\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e\model.safetensors"),
        @("m3-sp", "models--BAAI--bge-reranker-v2-m3\snapshots\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e\sentencepiece.bpe.model"),
        @("m3-special", "models--BAAI--bge-reranker-v2-m3\snapshots\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e\special_tokens_map.json"),
        @("m3-tokenizer", "models--BAAI--bge-reranker-v2-m3\snapshots\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e\tokenizer.json"),
        @("m3-tokenizer-config", "models--BAAI--bge-reranker-v2-m3\snapshots\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e\tokenizer_config.json")
    )
    $prepare = @("run", "--rm", "--network", "none", "--volume", "${volume}:${containerRoot}")
    $copyCommands = @("set -eu", "test ! -e '${containerSource}'", "test ! -e '${containerCache}'", "mkdir -p '${containerSource}' '${containerCache}'")
    foreach ($entry in $sourceMap.GetEnumerator()) {
        $hostPath = Join-Path $root $entry.Value[0]
        if (-not (Test-Path -LiteralPath $hostPath -PathType Leaf)) {
            throw "allowlisted source is unavailable: $hostPath"
        }
        $inputPath = "/input/source-$($entry.Key)"
        $prepare += @("--volume", "${hostPath}:${inputPath}:ro")
        $destination = "$containerSource/$($entry.Value[1])"
        $destinationDirectory = $destination.Substring(0, $destination.LastIndexOf('/'))
        $copyCommands += "mkdir -p '$destinationDirectory'; cp '$inputPath' '$destination'"
    }
    foreach ($row in $modelRows) {
        $hostPath = Join-Path $modelCache $row[1]
        if (-not (Test-Path -LiteralPath $hostPath -PathType Leaf)) {
            throw "allowlisted model file is unavailable: $hostPath"
        }
        $inputPath = "/input/model-$($row[0])"
        $prepare += @("--volume", "${hostPath}:${inputPath}:ro")
        $destination = "$containerCache/$($row[1].Replace('\', '/'))"
        $destinationDirectory = $destination.Substring(0, $destination.LastIndexOf('/'))
        $copyCommands += "mkdir -p '$destinationDirectory'; cp -L '$inputPath' '$destination'"
    }
    $prepare += @("--entrypoint", "/bin/sh", $image, "-c", ($copyCommands -join "; "))
    Invoke-Checked -Command (@("wslc") + $prepare)
    $arguments = Base-Container-Arguments
    Invoke-Checked -Command (@("wslc") + $arguments + @(
        "-m", "neuron_graph_rag.full_corpus_rerank_oracle_v2", "preflight",
        "--root", $containerSource, "--cache", $containerCache,
        "--runtime-root", $containerRun, "--source-commit", $sourceCommit
    ))
    exit 0
}

& wslc volume inspect $volume *> $null
if ($LASTEXITCODE -ne 0) {
    throw "preflight runtime volume is unavailable: $volume"
}

$arguments = Base-Container-Arguments
try {
    Invoke-Checked -Command (@("wslc") + $arguments + @(
        "-m", "neuron_graph_rag.full_corpus_rerank_oracle_v2", "claim",
        "--root", $containerSource, "--runtime-root", $containerRun,
        "--source-commit", $sourceCommit
    ))
    foreach ($kind in @("base", "v2-m3")) {
        Invoke-Checked -Command (@("wslc") + $arguments + @(
            "-m", "neuron_graph_rag.full_corpus_rerank_oracle_v2", "worker",
            "--root", $containerSource, "--cache", $containerCache,
            "--runtime-root", $containerRun, "--kind", $kind
        ))
    }
    $goldPath = Join-Path $root "tests\fixtures\full_corpus_rerank_oracle_v2.gold.json"
    $finalizeArguments = Base-Container-Arguments
    $finalizeArguments = $finalizeArguments[0..9] + @(
        "--volume", "${goldPath}:${containerSource}/tests/fixtures/full_corpus_rerank_oracle_v2.gold.json:ro"
    ) + $finalizeArguments[10..($finalizeArguments.Length - 1)]
    Invoke-Checked -Command (@("wslc") + $finalizeArguments + @(
        "-m", "neuron_graph_rag.full_corpus_rerank_oracle_v2", "finalize",
        "--root", $containerSource, "--runtime-root", $containerRun,
        "--source-commit", $sourceCommit
    ))
}
catch {
    $message = $_.Exception.Message
    try {
        Invoke-Checked -Command (@("wslc") + $arguments + @(
            "-m", "neuron_graph_rag.full_corpus_rerank_oracle_v2", "record-error",
            "--root", $containerSource, "--source-commit", $sourceCommit,
            "--message", $message
        ))
    }
    catch {
        Write-Warning "could not append failure evidence: $($_.Exception.Message)"
    }
    throw
}
finally {
    $export = @(
        "wslc", "run", "--rm", "--network", "none",
        "--volume", "${volume}:${containerRoot}:ro", "--volume", "${root}:/output",
        "--entrypoint", "/bin/sh", $image, "-c",
        "set -eu; mkdir -p '/output/tests/evidence/full_corpus_rerank_oracle_v2'; " +
        "if test -f '${containerRun}/preflight.attestation.json'; then " +
        "test ! -e '/output/tests/evidence/full_corpus_rerank_oracle_v2/development.preflight.json'; " +
        "cp '${containerRun}/preflight.attestation.json' " +
        "'/output/tests/evidence/full_corpus_rerank_oracle_v2/development.preflight.json'; fi; " +
        "if test -f '${containerSource}/tests/evidence/full_corpus_rerank_oracle_v2/development.claim.json'; then " +
        "test ! -e '/output/tests/evidence/full_corpus_rerank_oracle_v2/development.claim.json'; " +
        "cp '${containerSource}/tests/evidence/full_corpus_rerank_oracle_v2/development.claim.json' " +
        "'/output/tests/evidence/full_corpus_rerank_oracle_v2/development.claim.json'; fi; " +
        "if test -f '${containerSource}/tests/evidence/full_corpus_rerank_oracle_v2/development.observed.json'; then " +
        "test ! -e '/output/tests/evidence/full_corpus_rerank_oracle_v2/development.observed.json'; " +
        "cp '${containerSource}/tests/evidence/full_corpus_rerank_oracle_v2/development.observed.json' " +
        "'/output/tests/evidence/full_corpus_rerank_oracle_v2/development.observed.json'; fi; " +
        "if test -f '${containerSource}/tests/evidence/full_corpus_rerank_oracle_v2/development.error.json'; then " +
        "test ! -e '/output/tests/evidence/full_corpus_rerank_oracle_v2/development.error.json'; " +
        "cp '${containerSource}/tests/evidence/full_corpus_rerank_oracle_v2/development.error.json' " +
        "'/output/tests/evidence/full_corpus_rerank_oracle_v2/development.error.json'; fi"
    )
    Invoke-Checked -Command $export
}
