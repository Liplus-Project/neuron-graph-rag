param(
    [Parameter(Position = 0)]
    [ValidateSet("prebuild", "freeze", "audit")]
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
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
& $python -m neuron_graph_rag.cross_encoder_precision_v15_observation $Action
exit $LASTEXITCODE
