<#
.SYNOPSIS
    Start one agent of the sandwm lab in this terminal: an interactive Claude Code session in
    the agent's own git worktree, looping on its own until the lab is stopped.

.EXAMPLE
    .\lab\agent.ps1 -Role eval
    .\lab\agent.ps1 -Role train -DryRun       # print the command, start nothing
    .\lab\agent.ps1 -Role data -Continue      # pick up the agent's last conversation
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateSet('eval', 'data', 'train')][string]$Role,
    [string]$Model = 'claude-opus-5-5',
    [ValidateSet('low', 'medium', 'high', 'xhigh', 'max')][string]$Effort = 'xhigh',
    [ValidateSet('auto', 'acceptEdits', 'manual', 'dontAsk', 'bypassPermissions')][string]$PermissionMode = 'auto',
    [switch]$Continue,
    [switch]$DryRun,
    # For testing this script: the program to start in place of claude, followed by arguments to put first.
    [string[]]$Program = @('claude')
)
$ErrorActionPreference = 'Stop'

$repo  = Split-Path -Parent $PSScriptRoot
$lab   = Join-Path $repo '.lab'
$board = Join-Path $lab 'board'
$tree  = Join-Path $lab $Role

if (-not (Test-Path (Join-Path $tree '.claude\settings.local.json'))) {
    throw "The lab is not set up yet. In $repo run:  uv run python -m lab setup"
}
if (Test-Path (Join-Path $board 'STOP')) {
    throw "The lab is stopped (STOP is on the board). To lift it, in $repo run:  uv run python -m lab go"
}

# Both texts are passed to claude.exe as single arguments: keep them on one line, without double quotes.
$identity = "You are the $Role agent of the sandwm lab: one of three autonomous Claude Code agents (eval, data, train) " +
    "that together improve a world model of a 2D falling-sand world and make that world richer. " +
    "Your standing instructions are two files on the lab board: $board\PROGRAM.md and $board\roles\$Role.md. " +
    "Read both at the start of each tick, because the human edits them, and follow them where they differ from your defaults. " +
    "You work in your own git worktree, $tree, and in the shared folders under $lab. " +
    "The rest of this machine, including the main checkout $repo, is not yours to change. " +
    "Nobody is watching this terminal: do not ask questions here and do not wait for answers. " +
    "Whatever the human should know or decide goes to their inbox with: uv run python -m lab msg human. " +
    "The three agents run as the sessions lab-eval, lab-data and lab-train; yours is lab-$Role."

$tick = "/loop Lab tick ($Role agent). " +
    "First read $board\PROGRAM.md and $board\roles\$Role.md. " +
    "Then run: uv run python -m lab status. " +
    "Then read and handle your inbox. " +
    "Then do the most valuable piece of work your role file describes, up to a point where it is recorded and shared. " +
    "Before the turn ends, record what you did and what comes next with: uv run python -m lab note. " +
    "Finally schedule your next wake-up as PROGRAM.md says."

$claudeArgs = @(
    '--model', $Model,
    '--effort', $Effort,
    '--permission-mode', $PermissionMode,
    '--name', "lab-$Role",
    '--add-dir', $board, (Join-Path $lab 'runs'), (Join-Path $lab 'datasets'),
    '--append-system-prompt', $identity
)
if ($Continue) { $claudeArgs += '--continue' }
$claudeArgs += $tick

if ($DryRun) {
    Write-Output "cd $tree"
    Write-Output ('claude ' + (($claudeArgs | ForEach-Object { if ($_ -match '\s') { "'$_'" } else { $_ } }) -join ' '))
    return
}

$env:LAB_ROLE = $Role
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location $tree
$Host.UI.RawUI.WindowTitle = "lab-$Role"
$exe = $Program[0]
$all = @($Program | Select-Object -Skip 1) + $claudeArgs
& $exe @all
