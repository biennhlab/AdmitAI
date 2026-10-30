param(
    [ValidateSet("Check", "Start", "Stop")]
    [string]$Action = "Start",
    [switch]$SecurityGateConfirmed
)
$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $projectRoot
try {
    $runtimeArguments = @("scripts/local_runtime.py", $Action.ToLowerInvariant())
    if ($SecurityGateConfirmed) { $runtimeArguments += "--security-gate-confirmed" }
    & python @runtimeArguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally { Pop-Location }
