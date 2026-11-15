<#
.SYNOPSIS
    Start the sandwm lab: one terminal tab per agent (eval, data, train, research), each an
    interactive Claude Code session. eval, data and train keep working until you stop the lab;
    research sleeps until one of them, or you, writes to it.

.DESCRIPTION
    Run it from anywhere; it finds the repository from its own location.
      .\lab\launch.ps1 -DryRun                  print what would be started, start nothing
      .\lab\launch.ps1                          start all four agents
      .\lab\launch.ps1 -Roles train             start (or restart) one agent
      .\lab\launch.ps1 -Roles eval -Continue    restart one agent in its last conversation
    To stop everything:   uv run python -m lab stop
#>
[CmdletBinding()]
param(
    [ValidateSet('eval', 'data', 'train', 'research')][string[]]$Roles = @('eval', 'data', 'train', 'research'),
    [string]$Model = 'claude-opus-5-5',
    [ValidateSet('low', 'medium', 'high', 'xhigh', 'max')][string]$Effort = 'xhigh',
    [ValidateSet('auto', 'acceptEdits', 'manual', 'dontAsk', 'bypassPermissions')][string]$PermissionMode = 'auto',
    [switch]$Continue,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'

$repo  = Split-Path -Parent $PSScriptRoot
$agent = Join-Path $PSScriptRoot 'agent.ps1'
$board = Join-Path $repo '.lab\board'

if (-not (Get-Command claude -ErrorAction SilentlyContinue)) { throw 'claude is not on the PATH.' }
if (-not (Test-Path (Join-Path $board 'limits.toml'))) { throw "The lab is not set up yet. In $repo run:  uv run python -m lab setup" }
if (Test-Path (Join-Path $board 'STOP')) { throw "The lab is stopped. To lift the stop, in $repo run:  uv run python -m lab go" }
if (-not $DryRun -and ($env:CLAUDECODE -or $env:CLAUDE_CODE_CHILD_SESSION)) {
    # Agents started from inside a Claude Code session become its children and cannot be messaged by the others.
    throw 'Run this from your own terminal, not from inside a Claude Code session: agents started there cannot be woken by the other agents.'
}

Push-Location $repo
try { & uv run python -m lab budget } finally { Pop-Location }
Write-Output ''

foreach ($role in $Roles) {
    if ($DryRun) {
        Write-Output "--- lab-$role ($Model, effort $Effort, permission mode $PermissionMode)"
        & $agent -Role $role -Model $Model -Effort $Effort -PermissionMode $PermissionMode -Continue:$Continue -DryRun
        Write-Output ''
        continue
    }
    $tree  = Join-Path $repo ".lab\$role"
    $inner = @('-NoExit', '-ExecutionPolicy', 'Bypass', '-File', $agent, '-Role', $role, '-Model', $Model,
               '-Effort', $Effort, '-PermissionMode', $PermissionMode)
    if ($Continue) { $inner += '-Continue' }
    if (Get-Command wt.exe -ErrorAction SilentlyContinue) {
        & wt.exe -w sandwm-lab new-tab --title "lab-$role" -d $tree powershell.exe @inner
    } else {
        Start-Process powershell.exe -ArgumentList $inner -WorkingDirectory $tree
    }
    Write-Output "started lab-$role"
    Start-Sleep -Seconds 5   # let each session get going before the next one starts
}

if ($DryRun) {
    Write-Output 'Dry run: nothing was started.'
} else {
    Write-Output ''
    Write-Output 'The agents are running in the Windows Terminal window "sandwm-lab", one tab each.'
    Write-Output 'Watch:  uv run python -m lab status        Stop:  uv run python -m lab stop'
}
