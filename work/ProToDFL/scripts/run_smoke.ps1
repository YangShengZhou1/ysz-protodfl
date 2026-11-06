param(
    [string]$Python = (Join-Path $PSScriptRoot '../../../.venv/Scripts/python.exe'),
    [string]$DataDir = (Join-Path $PSScriptRoot '../../../datasets'),
    [string]$OutputDir = (Join-Path $PSScriptRoot '../outputs')
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
    "--client_num_in_total", "5",
    "--frac", "1.0",
    "--comm_round", "0",
    "--epochs", "1",
    "--batch_size", "256",
    "--cs", "ring",
    "--gpu", "0",
    "--dynamic", "0",
    "--t_max", "100",
    "--momentums", "0.5",
    "--seed", "0",
    "--times", "1"
)

function Run-Guarded {
    param([int]$Step, [int]$K, [string]$Tag)
    $Cmd = @(
        $Watchdog,
        "--idle-timeout", "600",
        "--hard-timeout", "7200",
        "--watch-dir", $LogDir,
        "--status-file", (Join-Path $WatchdogDir "$Tag.json"),
        "--stdout", (Join-Path $WatchdogDir "$Tag.out"),
        "--stderr", (Join-Path $WatchdogDir "$Tag.err"),
        "--", $Python, $Entry
    ) + $Common + @("--step", "$Step", "--k", "$K", "--tag", $Tag)
    & $Python @Cmd
    if ($LASTEXITCODE -ne 0) { throw "Run failed or watchdog timeout: $Tag" }
}

Run-Guarded -Step 1 -K 2 -Tag "smoke-fixed"
Run-Guarded -Step 0 -K 2 -Tag "smoke-reference"
