param(
    [Parameter(Position = 0)]
    [ValidateSet("prebuild", "smoke", "audit")]
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
& $python -m neuron_graph_rag.cross_encoder_precision_v9_observation $Action
exit $LASTEXITCODE
