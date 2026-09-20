param(
    [Parameter(Position = 0)]
    [ValidateSet("audit", "probe")]
    [string]$Action = "audit"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "project verification environment is unavailable"
}

$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONUTF8 = "1"
& $python -m neuron_graph_rag.full_corpus_rerank_oracle $Action --root $root
exit $LASTEXITCODE
