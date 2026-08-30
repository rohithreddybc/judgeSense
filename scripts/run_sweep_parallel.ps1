# Launch the v2 multi-vendor sweep as genuinely detached Windows processes.
#
# The Bash equivalent (run_sweep_parallel.sh) uses nohup, which does not detach
# properly under Git Bash on Windows: the workers are children of the bash
# session and die with it. Every process below is its own Windows process and
# outlives the shell that started it.
#
# Splitting by PROVIDER avoids rate-limit contention. The larger providers are
# split further for throughput; run_cell takes a lock per cell, so an accidental
# overlap is refused rather than silently duplicating paid work.
#
# Every process is resumable: raw output is append-only and completed rows are
# skipped on restart, so an interruption costs time and nothing else.
#
# python -u so the logs are written as work happens rather than buffered until
# exit -- a worker that dies mid-run left an empty log file otherwise.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force (Join-Path $root 'logs\sweep') | Out-Null

$common = @('--budget-policy', 'matched', '--repeat-baseline', '--skip-preflight', '--yes')

$groups = [ordered]@{
    google  = @('gemini-flash', 'gemini-3.7-flash', 'gemini-3.1-pro')
    hf1     = @('llama3-8b', 'llama-3.3-70b')
    hf2     = @('llama-4-scout', 'llama-4-maverick')
    hf3     = @('gemma-4-31b', 'qwen3-8b')
    hf4     = @('qwen3-14b', 'qwen3-32b')
    hf5     = @('qwen3.8-27b-hf')
    mistral = @('mistral-small', 'magistral-small', 'mistral-medium')
    novita  = @('qwen', 'deepseek-v4-flash')
    ds1     = @('qwen-3.6-flash', 'qwen3.7-flash')
    ds2     = @('deepseek-v4-flash-ds', 'glm-5.2')
    ds3     = @('kimi-k3', 'deepseek-v4-pro')
}

Write-Output ("launching {0} judges across 5 providers:" -f ($groups.Values | ForEach-Object { $_.Count } | Measure-Object -Sum).Sum)

foreach ($tag in $groups.Keys) {
    $judges = $groups[$tag]
    $out = Join-Path $root "logs\sweep\$tag.log"
    $err = Join-Path $root "logs\sweep\$tag.err.log"
    $argv = @('-u', '-m', 'src.run_v2', '--judges') + $judges + $common

    $p = Start-Process -FilePath 'python' -ArgumentList $argv `
        -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err

    Set-Content -Path (Join-Path $root "logs\sweep\$tag.pid") -Value $p.Id -Encoding utf8
    Write-Output ("  [{0}] pid {1} : {2}" -f $tag, $p.Id, ($judges -join ' '))
}

Write-Output ''
Write-Output 'monitor: bash scripts/sweep_status.sh'
