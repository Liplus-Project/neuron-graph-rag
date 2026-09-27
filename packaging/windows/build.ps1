param([ValidateSet('cpu','cuda')][string]$Flavor)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$version = (Select-String -Path "$root\pyproject.toml" -Pattern '^version = "([0-9]+\.[0-9]+\.[0-9]+)"$').Matches.Groups[1].Value
if (-not $version) { throw 'Cannot read package version' }
$outputRoot = Join-Path $root 'package-output'
$bundleRoot = Join-Path $root 'package-build'
New-Item -ItemType Directory -Force $outputRoot, $bundleRoot | Out-Null

# Each job starts on a clean runner. PyPI packages and the CUDA wheel source are pinned.
python -m pip install --disable-pip-version-check 'pyinstaller==6.22.3' 'mcp==2.2.0' 'httpx2==2.13.1' 'uvicorn==0.54.0' 'starlette==1.7.0' 'numpy==2.4.6' 'onnxruntime==1.30.0' 'tokenizers==0.22.1'
if ($LASTEXITCODE) { throw 'Package dependency installation failed' }
if ($Flavor -eq 'cuda') {
  python -m pip install --disable-pip-version-check 'torch==2.9.1+cu128' --index-url 'https://download.pytorch.org/whl/cu128'
  if ($LASTEXITCODE) { throw 'CUDA PyTorch installation failed' }
  python -m pip install --disable-pip-version-check 'transformers==4.57.6' 'safetensors==0.6.2'
  if ($LASTEXITCODE) { throw 'CUDA runtime installation failed' }
}
python -m pip install --disable-pip-version-check --no-deps "$root"
if ($LASTEXITCODE) { throw 'NGR installation failed' }

$arguments = @('--noconfirm','--clean','--onedir','--name','NGR','--distpath',$bundleRoot,'--workpath',(Join-Path $bundleRoot 'work'),'--specpath',$bundleRoot,'--copy-metadata','neuron-graph-rag','--collect-submodules','neuron_graph_rag')
# MCP's optional CLI imports typer and exits when that unrelated extra is absent.
# The application imports the MCP client/server modules statically; collecting
# every mcp.* module would include its unrelated CLI and fail the build.
if ($Flavor -eq 'cuda') { $arguments += @('--hidden-import','transformers.models.xlm_roberta.modeling_xlm_roberta','--hidden-import','transformers.models.xlm_roberta.tokenization_xlm_roberta_fast','--hidden-import','safetensors.torch','--collect-data','transformers') }
$arguments += (Join-Path $root 'packaging\windows\entry.py')
python -m PyInstaller @arguments
if ($LASTEXITCODE) { throw 'PyInstaller build failed' }
& "$bundleRoot\NGR\NGR.exe" --version
if ($LASTEXITCODE) { throw 'Frozen executable version probe failed' }
@{ schema = 'ngr.windows-package/v1'; version = $version; flavor = $Flavor } | ConvertTo-Json | Set-Content -Path (Join-Path $bundleRoot 'NGR\package-manifest.json') -Encoding utf8
$bundleBytes = (Get-ChildItem -LiteralPath (Join-Path $bundleRoot 'NGR') -File -Recurse | Measure-Object -Property Length -Sum).Sum
Write-Output "BUNDLE_SIZE_BYTES=$bundleBytes"

$innoVersion = '6.7.3'
$innoUrl = "https://github.com/jrsoftware/issrc/releases/download/is-6_7_3/innosetup-$innoVersion.exe"
$innoHash = '9c73c3bae7ed48d44112a0f48e66742c00090bdb5bef71d9d3c056c66e97b732'
$innoInstaller = Join-Path $env:RUNNER_TEMP 'innosetup-6.7.3.exe'
$innoDir = Join-Path $env:RUNNER_TEMP 'inno-setup-6.7.3'
if (Test-Path -LiteralPath $innoDir) { throw "Inno Setup target is not clean: $innoDir" }
Invoke-WebRequest -Uri $innoUrl -OutFile $innoInstaller
if ((Get-FileHash $innoInstaller -Algorithm SHA256).Hash.ToLowerInvariant() -ne $innoHash) { throw 'Inno Setup download hash mismatch' }
$innoProcess = Start-Process -FilePath $innoInstaller -ArgumentList @('/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART',"/DIR=$innoDir") -Wait -PassThru -WindowStyle Hidden
if ($innoProcess.ExitCode) { throw "Inno Setup installation failed with code $($innoProcess.ExitCode)" }
$compiler = Join-Path $innoDir 'ISCC.exe'
if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) { throw "Verified Inno Setup compiler not found: $compiler" }
& $compiler "/DAppVersion=$version" "/DFlavor=$Flavor" "/DBundleDir=$bundleRoot\NGR" "/DOutputRoot=$outputRoot" (Join-Path $root 'packaging\windows\installer.iss')
if ($LASTEXITCODE) { throw 'Inno Setup compile failed' }

$fileName = "NGR-$version-windows-x64-$Flavor-setup.exe"
$setupFile = Join-Path $outputRoot $fileName
$digest = (Get-FileHash $setupFile -Algorithm SHA256).Hash.ToLowerInvariant()
$setupBytes = (Get-Item $setupFile).Length
Write-Output "SETUP_SIZE_BYTES=$setupBytes"
"$digest *$fileName" | Set-Content -Path "$setupFile.sha256" -Encoding ascii
@{ schema = 'ngr.windows-package/v1'; version = $version; flavor = $Flavor; setup_file = $fileName; sha256 = $digest; size = $setupBytes; bundle_size_bytes = $bundleBytes } | ConvertTo-Json | Set-Content -Path (Join-Path $outputRoot 'package-manifest.json') -Encoding utf8
