param(
    [string]$Python = (Join-Path $PSScriptRoot '../../../.venv/Scripts/python.exe'),
    [string]$DataDir = (Join-Path $PSScriptRoot '../../../datasets'),
    [string]$OutputDir = (Join-Path $PSScriptRoot '../outputs'),
    [int]$Seed = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$Entry = Join-Path $ProjectRoot "fedml_experiments/standalone/ProToDFL/main_ProToDFL.py"
$Watchdog = Join-Path $ProjectRoot "scripts/run_with_watchdog.py"
$WatchdogDir = Join-Path $OutputDir "watchdog"
$LogDir = Join-Path $OutputDir "logs/cifar10"
New-Item -ItemType Directory -Force -Path $WatchdogDir | Out-Null

$Common = @(
    "--dataset", "cifar10",
    "--model", "resnet18",
    "--data_dir", $DataDir,
    "--output_dir", $OutputDir,
    "--partition_method", "dir",
    "--partition_alpha", "0.3",
    "--client_num_in_total", "20",
    "--frac", "1.0",
    "--comm_round", "2",
    "--epochs", "1",
    "--batch_size", "256",
    "--cs", "ring",
    "--gpu", "0",
    "--dynamic", "0",
    "--k", "2",
    "--t_max", "100",
    "--momentums", "0.5",
    "--seed", "$Seed",
    "--times", "1",
    "--step", "0",
    "--max_neighbors", "2",
    "--validation_ratio", "0.2",
    "--save_models_round", "1"
)

function Run-Guarded {
    param([string]$Tag, [string[]]$Extra)
    $Cmd = @(
        $Watchdog,
        "--idle-timeout", "900",
        "--hard-timeout", "10800",
        "--watch-dir", $LogDir,
        "--status-file", (Join-Path $WatchdogDir "$Tag.json"),
        "--stdout", (Join-Path $WatchdogDir "$Tag.out"),
        "--stderr", (Join-Path $WatchdogDir "$Tag.err"),
        "--", $Python, $Entry
    ) + $Common + $Extra + @("--tag", $Tag)
    & $Python @Cmd
    if ($LASTEXITCODE -ne 0) { throw "Run failed or watchdog timeout: $Tag" }
}

Run-Guarded -Tag "paired20-all-s$Seed" -Extra @("--aggregate_scope", "all")
Run-Guarded -Tag "paired20-base-s$Seed" -Extra @("--aggregate_scope", "base")
