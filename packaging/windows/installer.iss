#ifndef AppVersion
  #error AppVersion must be supplied by the build
#endif
#ifndef Flavor
  #error Flavor must be supplied by the build
#endif
#ifndef BundleDir
  #error BundleDir must be supplied by the build
#endif
#ifndef OutputRoot
  #error OutputRoot must be supplied by the build
#endif

[Setup]
AppId={{CC547682-8A6D-4762-A489-EF63CBCA6F40}
AppName=Neuron Graph RAG
AppVersion={#AppVersion}
AppVerName=Neuron Graph RAG {#AppVersion} ({#Flavor})
AppPublisher=Liplus Project
DefaultDirName={localappdata}\Programs\Neuron Graph RAG
DefaultGroupName=Neuron Graph RAG
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputRoot}
OutputBaseFilename=NGR-{#AppVersion}-windows-x64-{#Flavor}-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\NGR.exe
CloseApplications=yes
RestartApplications=no
DisableProgramGroupPage=yes
LicenseFile={#BundleDir}\licenses\Setup-LICENSE.txt

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Configure MCP clients"; Filename: "{app}\NGR.exe"; Parameters: "--configure-clients"
Name: "{group}\Licenses and notices"; Filename: "{app}\licenses\README.txt"
Name: "{group}\Uninstall Neuron Graph RAG"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\NGR.exe"; Parameters: "--configure-clients"; Description: "Configure Codex or Claude Code MCP connection"; Flags: postinstall skipifsilent unchecked nowait

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
  if WizardSilent and (CompareText(ExpandConstant('{param:ACCEPTVCRUNTIME|}'), 'yes') <> 0) then
  begin
    Log('Silent installation refused: /ACCEPTVCRUNTIME=yes is required.');
    Result := False;
  end;
end;
