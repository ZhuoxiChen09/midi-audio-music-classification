$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir "unattended_$Stamp.log"
$CandidateTask3 = Join-Path $Root "predictions3_candidate_long.json"

Set-Location $Root

$env:PYTHONUNBUFFERED = "1"
$env:TASK3_SKIP_EXISTING = "0"
$env:TASK3_BACKBONES = "resnet34"
$env:TASK3_SEEDS = "0,17,42"
$env:TASK3_TTA_SHIFTS = "5"
$env:TASK3_CHECKPOINT = "task3_checkpoint_valrun.pt"

Start-Transcript -Path $LogPath -Append

try {
    Write-Host "Working directory: $Root"
    Write-Host "Log: $LogPath"

    python -u -c "import pipeline as a; a.run3(fit_epochs=45, finetune_epochs=0, finetune_low_epochs=0)"

    if (Test-Path (Join-Path $Root "predictions3.json")) {
        Copy-Item -Force (Join-Path $Root "predictions3.json") $CandidateTask3
        Write-Host "Saved long-run Task 3 candidate to $CandidateTask3"
    }

    python -m py_compile pipeline.py
    Write-Host "Unattended run finished."
} catch {
    Write-Error $_
    throw
} finally {
    Stop-Transcript
}
