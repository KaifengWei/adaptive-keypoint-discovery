param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ArisArgs
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$arisExecutable = Join-Path $projectRoot '.aris\bin\aris.exe'
if (-not (Test-Path -LiteralPath $arisExecutable -PathType Leaf)) {
    throw "ARIS-Code is missing. Extract aris.exe from aris-code-windows-x64.zip to .aris\bin\aris.exe."
}

Push-Location $projectRoot
try {
    & $arisExecutable @ArisArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
