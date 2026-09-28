param([ValidateSet('cpu','cuda')][string]$Flavor)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$version = (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'release-version.txt') -Raw).Trim()
if ($version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+$') { throw 'Invalid release version' }
$outputRoot = Join-Path $root 'package-output'
$bundleRoot = Join-Path $root 'package-build'
New-Item -ItemType Directory -Force $outputRoot, $bundleRoot | Out-Null
$packageSource = Join-Path $bundleRoot 'release-source'
New-Item -ItemType Directory -Force $packageSource | Out-Null
Copy-Item -LiteralPath (Join-Path $root 'pyproject.toml'), (Join-Path $root 'README.md'), (Join-Path $root 'LICENSE'), (Join-Path $root 'NOTICE') -Destination $packageSource
$releaseSrc = Join-Path $packageSource 'src'
New-Item -ItemType Directory -Force $releaseSrc | Out-Null
Copy-Item -LiteralPath (Join-Path $root 'src/neuron_graph_rag'), (Join-Path $root 'src/neuron_graph_rag_mcp') -Destination $releaseSrc -Recurse
$releaseProject = Join-Path $packageSource 'pyproject.toml'
$projectContent = [System.IO.File]::ReadAllText($releaseProject)
$releaseContent = [regex]::Replace($projectContent, '(?m)^version = "[0-9]+\.[0-9]+\.[0-9]+"$', "version = `"$version`"")
if ($releaseContent -eq $projectContent) { throw 'Release version did not replace source metadata' }
[System.IO.File]::WriteAllText($releaseProject, $releaseContent, [System.Text.UTF8Encoding]::new($false))

# Each job starts on a clean runner. PyPI packages and the CUDA wheel source are pinned.
python -m pip install --disable-pip-version-check 'pyinstaller==6.22.3' 'mcp==2.2.0' 'httpx2==2.13.1' 'uvicorn==0.54.0' 'starlette==1.7.0' 'numpy==1.26.4' 'onnxruntime==1.30.0' 'tokenizers==0.22.1'
if ($LASTEXITCODE) { throw 'Package dependency installation failed' }
if ($Flavor -eq 'cuda') {
  python -m pip install --disable-pip-version-check 'torch==2.9.1+cu128' --index-url 'https://download.pytorch.org/whl/cu128'
  if ($LASTEXITCODE) { throw 'CUDA PyTorch installation failed' }
  python -m pip install --disable-pip-version-check 'transformers==4.57.6' 'safetensors==0.6.2'
  if ($LASTEXITCODE) { throw 'CUDA runtime installation failed' }
}
python -m pip install --disable-pip-version-check --no-deps "$packageSource"
if ($LASTEXITCODE) { throw 'NGR installation failed' }
# Keep unrelated runner tools' DLLs out of PyInstaller's dependency search.
$pythonHome = python -c 'import sys; print(sys.base_prefix)'
if ($LASTEXITCODE) { throw 'Python runtime lookup failed' }
$pythonExe = python -c 'import sys; print(sys.executable)'
if ($LASTEXITCODE) { throw 'Python executable lookup failed' }
$pythonScripts = Split-Path -Parent $pythonExe
$env:PATH = "$pythonScripts;$pythonHome;$env:SystemRoot\System32;$env:SystemRoot"

$arguments = @('--noconfirm','--clean','--onedir','--name','NGR','--distpath',$bundleRoot,'--workpath',(Join-Path $bundleRoot 'work'),'--specpath',$bundleRoot,'--copy-metadata','neuron-graph-rag','--collect-submodules','neuron_graph_rag')
# MCP's optional CLI imports typer and exits when that unrelated extra is absent.
# The application imports the MCP client/server modules statically; collecting
# every mcp.* module would include its unrelated CLI and fail the build.
if ($Flavor -eq 'cuda') {
  # AutoTokenizer's lazy mapping imports model packages at runtime, including
  # models unrelated to the selected XLM-RoBERTa snapshot. Collect the whole
  # model namespace so a missing dynamic import cannot move to the next model.
  $arguments += @('--collect-submodules','transformers.models','--hidden-import','safetensors.torch','--collect-data','transformers')
}
$arguments += (Join-Path $root 'packaging\windows\entry.py')
python -m PyInstaller @arguments
if ($LASTEXITCODE) { throw 'PyInstaller build failed' }
# Replace any PyInstaller-collected copies with unmodified x64 files from a
# licensed Visual Studio installation's VC\Redist tree. Fail closed when that
# source is unavailable or ambiguous; System32 and wheel copies are not sources.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) { throw 'Visual Studio locator not found' }
$installations = @(& $vswhere -all -products '*' -property installationPath)
if ($LASTEXITCODE) { throw 'Visual Studio installation lookup failed' }
$runtimeNames = @('msvcp140.dll','msvcp140_1.dll','vcruntime140.dll','vcruntime140_1.dll')
$candidates = @()
foreach ($installation in $installations) {
  if (-not $installation -or $installation -match '(?i)preview' -or $installation -notmatch '[\\/]2022[\\/]') { continue }
  $edition = Split-Path -Leaf $installation
  if ($edition -notin @('Community','Professional','Enterprise')) { continue }
  $redistRoot = Join-Path $installation 'VC\Redist\MSVC'
  if (-not (Test-Path -LiteralPath $redistRoot -PathType Container)) { continue }
  foreach ($versionDir in (Get-ChildItem -LiteralPath $redistRoot -Directory)) {
    if ($versionDir.Name -notmatch '^14\.\d+\.\d+$') { continue }
    $crt = Join-Path $versionDir.FullName 'x64\Microsoft.VC143.CRT'
    if ($runtimeNames | Where-Object { -not (Test-Path -LiteralPath (Join-Path $crt $_) -PathType Leaf) }) { continue }
    $candidates += [pscustomobject]@{ Installation = $installation; Edition = $edition; Crt = $crt; Version = [version]$versionDir.Name }
  }
}
if (-not $candidates) { throw 'No complete release x64 Visual Studio VC\Redist CRT found' }
$selected = $candidates | Sort-Object Version -Descending | Select-Object -First 1
$runtimeFiles = @()
foreach ($dllName in $runtimeNames) {
  $source = Join-Path $selected.Crt $dllName
  $data = [System.IO.File]::ReadAllBytes($source)
  $peOffset = [BitConverter]::ToInt32($data, 0x3c)
  if ($data.Length -lt ($peOffset + 6) -or [BitConverter]::ToUInt32($data, $peOffset) -ne 0x4550 -or [BitConverter]::ToUInt16($data, $peOffset + 4) -ne 0x8664) {
    throw "VC runtime is not x64 PE: $source"
  }
  $destination = Join-Path $bundleRoot "NGR\_internal\$dllName"
  Copy-Item -LiteralPath $source -Destination $destination -Force
  $sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
  if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sourceHash) { throw "VC runtime copy changed: $dllName" }
  $runtimeFiles += @{ name = $dllName; source = $source; sha256 = $sourceHash; file_version = (Get-Item -LiteralPath $source).VersionInfo.FileVersion }
}
$runtimeProvenance = Join-Path $bundleRoot 'vc-runtime-source.json'
$editionTerms = if ($selected.Edition -eq 'Community') { 'https://visualstudio.microsoft.com/license-terms/vs2022-ga-community/' } else { 'https://visualstudio.microsoft.com/license-terms/vs2022-ga-proenterprise/' }
@{ schema = 'ngr.vc-redist-source/v1'; installation = $selected.Installation; edition = $selected.Edition; edition_terms = $editionTerms; redist_directory = $selected.Crt; files = $runtimeFiles } |
  ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $runtimeProvenance -Encoding utf8
& "$bundleRoot\NGR\NGR.exe" --version
if ($LASTEXITCODE) { throw 'Frozen executable version probe failed' }
python (Join-Path $root 'packaging\windows\collect_notices.py') --analysis (Join-Path $bundleRoot 'work\NGR\Analysis-00.toc') --bundle (Join-Path $bundleRoot 'NGR') --repository $root --flavor $Flavor --vc-source-manifest $runtimeProvenance
if ($LASTEXITCODE) { throw 'Bundled license collection failed' }
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
