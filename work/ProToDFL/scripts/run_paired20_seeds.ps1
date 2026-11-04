param(
    [string]$Python = "c:/Users/Administrator/Desktop/bishe/.venv/Scripts/python.exe",
    [string]$DataDir = "c:/Users/Administrator/Desktop/bishe/datasets",
    [string]$OutputDir = "c:/Users/Administrator/Desktop/bishe/work/ProToDFL/outputs"
)

$ErrorActionPreference = "Stop"
$Runner = Join-Path $PSScriptRoot "run_paired20.ps1"

& $Runner -Python $Python -DataDir $DataDir -OutputDir $OutputDir -Seed 1
if ($LASTEXITCODE -ne 0) { throw "seed 1 failed" }
& $Runner -Python $Python -DataDir $DataDir -OutputDir $OutputDir -Seed 2
if ($LASTEXITCODE -ne 0) { throw "seed 2 failed" }
