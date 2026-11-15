<#
.SYNOPSIS
    Start one agent of the sandwm lab in this terminal: an interactive Claude Code session in
    the agent's own git worktree, looping on its own until the lab is stopped.

.DESCRIPTION
    The script stays in the terminal for as long as the agent runs. From time to time it ends
    the agent's conversation and starts an empty one in its place (a "fresh start": see
    lab\sessions.py and "Fresh starts" in PROGRAM.md), because Claude Code cannot clear a
    running conversation by itself. It does so when the agent asks for it, or when the
    conversation is due and sits idle.

.EXAMPLE
    .\lab\agent.ps1 -Role eval
    .\lab\agent.ps1 -Role train -DryRun       # print the command, start nothing
    .\lab\agent.ps1 -Role data -Continue      # pick up the agent's last conversation
    .\lab\agent.ps1 -Role research            # the research agent: no loop, woken only by messages
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateSet('eval', 'data', 'train', 'research')][string]$Role,
    [string]$Model = 'claude-opus-5-5',
    [ValidateSet('low', 'medium', 'high', 'xhigh', 'max')][string]$Effort = 'xhigh',
    [ValidateSet('auto', 'acceptEdits', 'manual', 'dontAsk', 'bypassPermissions')][string]$PermissionMode = 'auto',
    [switch]$Continue,
    [switch]$DryRun,
    # For testing this script: the program to start in place of claude, followed by arguments to put first;
    # and how many seconds pass between two looks for a fresh-start request.
    [string[]]$Program = @('claude'),
    [int]$Poll = 5
)
$ErrorActionPreference = 'Stop'

$repo    = Split-Path -Parent $PSScriptRoot
$lab     = Join-Path $repo '.lab'
$board   = Join-Path $lab 'board'
$tree    = Join-Path $lab $Role
$seen    = Join-Path $board "status\$Role.json"    # the guard hook's record of the role's conversation
$request = Join-Path $board "status\$Role.fresh"   # written by `python -m lab fresh`: replace the conversation now
$python  = Join-Path $repo '.venv\Scripts\python.exe'

if (-not (Test-Path (Join-Path $tree '.claude\settings.local.json'))) {
    throw "The lab is not set up yet. In $repo run:  uv run python -m lab setup"
}
if (Test-Path (Join-Path $board 'STOP')) {
    throw "The lab is stopped (STOP is on the board). To lift it, in $repo run:  uv run python -m lab go"
}
if (-not $DryRun -and $Program[0] -eq 'claude') {
    if ($env:CLAUDECODE -or $env:CLAUDE_CODE_CHILD_SESSION) {
        # A session started from inside another Claude Code session is that session's child: it gets no
        # address of its own, so the other agents could never message or wake it.
        throw "Start lab agents from your own terminal, not from inside a Claude Code session: an agent started there cannot be messaged by the others."
    }
    # One conversation per role: a second one would work in the same worktree and answer to the same name.
    $twin = Get-CimInstance Win32_Process -Filter "Name = 'claude.exe' OR Name = 'node.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match ('--name lab-' + $Role + '(\s|$)') } | Select-Object -First 1
    if ($twin) {
        throw "lab-$Role is already running (process $($twin.ProcessId)). Close its tab first. If it has a cloud job running (uv run python -m lab status), wait for the job: it ends with the tab."
    }
}

# Both texts are passed to claude.exe as single arguments: keep them on one line, without double quotes.
$identity = "You are the $Role agent of the sandwm lab: one of four autonomous Claude Code agents (eval, data, train, research) " +
    "that together improve a world model of a 2D falling-sand world and make that world richer. " +
    "Your standing instructions are two files on the lab board: $board\PROGRAM.md and $board\roles\$Role.md. " +
    "Read both every time you are woken, because the human edits them, and follow them where they differ from your defaults. " +
    "You work in your own git worktree, $tree, and in the shared folders under $lab. " +
    "The rest of this machine, including the main checkout $repo, is not yours to change. " +
    "Nobody is watching this terminal: do not ask questions here and do not wait for answers. " +
    "Whatever the human should know or decide goes to their inbox with: uv run python -m lab msg human. " +
    "The agents run as the sessions lab-eval, lab-data, lab-train and lab-research; yours is lab-$Role."

$tick = "/loop Lab tick ($Role agent). " +
    "First read $board\PROGRAM.md and $board\roles\$Role.md. " +
    "Then run: uv run python -m lab status. " +
    "Then read and handle your inbox. " +
    "If this conversation is new, take the fresh look that PROGRAM.md describes under Fresh starts. " +
    "Then do the most valuable piece of work your role file describes, up to a point where it is recorded and shared. " +
    "Before the turn ends, record what you did and what comes next with: uv run python -m lab note. " +
    "Finally schedule your next wake-up as PROGRAM.md says."
if ($Role -eq 'research') {
    # No loop and no timer: this agent sleeps until another agent or the human writes to it.
    $tick = "Lab research agent: you have just been started. " +
        "Read $board\PROGRAM.md and $board\roles\research.md. " +
        "Then run: uv run python -m lab status. " +
        "Then read tracks.md and your inbox on the board and deal with whatever is there. " +
        "Record what you did with: uv run python -m lab note. Then end your turn. " +
        "Start no loop and no timer: you are woken only by messages."
}

function Get-CompactWindow {
    # compact_tokens in limits.toml: Claude Code summarises (compacts) the conversation once it holds that
    # many tokens. Returns the value for `claude --autocompact`, or '' to leave Claude Code's own default.
    try {
        $hit = Select-String -Path (Join-Path $board 'limits.toml') -Pattern '^\s*compact_tokens\s*=\s*"?([0-9_]+)\s*([kK]?)' |
            Select-Object -First 1
    } catch { return '' }
    if (-not $hit) { return '' }
    $n = [long]($hit.Matches[0].Groups[1].Value -replace '_', '')
    if ($hit.Matches[0].Groups[2].Value) { $n *= 1000 }
    if ($n -le 0) { return '' }
    [string][Math]::Min([Math]::Max($n, 100000), 1000000)   # the range claude accepts
}

function Get-ClaudeArgs([bool]$Resume) {
    $list = @(
        '--model', $Model,
        '--effort', $Effort,
        '--permission-mode', $PermissionMode,
        '--name', "lab-$Role",
        '--add-dir', $board, (Join-Path $lab 'runs'), (Join-Path $lab 'datasets'),
        '--append-system-prompt', $identity
    )
    $window = Get-CompactWindow   # read again at every fresh start, so a changed limit applies to the next conversation
    if ($window) { $list += @('--autocompact', $window) }
    if ($Resume) {
        # By its id, as the guard hook recorded it: `claude --continue` skips conversations that began with /loop.
        $id = ''
        try { $id = [string](Get-Content $seen -Raw -ErrorAction Stop | ConvertFrom-Json).session } catch { }
        if ($id -match '^[0-9a-fA-F-]{36}$') { $list += @('--resume', $id) } else { $list += '--continue' }
    }
    $list + $tick
}

if ($DryRun) {
    Write-Output "cd $tree"
    Write-Output ('claude ' + ((Get-ClaudeArgs ([bool]$Continue) | ForEach-Object { if ($_ -match '\s') { "'$_'" } else { $_ } }) -join ' '))
    return
}

# Runs beside the conversation, as a background job: ends the conversation's process when a fresh start
# is requested, and every 24 looks asks whether a conversation that is due and idle should get one.
$watcher = {
    param($shell, $born, $request, $python, $repo, $poll)
    function Alive {
        # false once the launcher is gone: its process id may then belong to something else, so do nothing more
        $p = Get-Process -Id $shell -ErrorAction SilentlyContinue
        $p -and $p.StartTime.Ticks -eq $born
    }
    function Kids($id) {
        # the processes started by process $id, without this job and without console hosts
        @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $id" |
            Where-Object { $_.ProcessId -ne $PID -and $_.Name -notin 'conhost.exe', 'OpenConsole.exe' })
    }
    function Busy {
        # true while the conversation has anything running: a tool call, a background command
        @(Kids $shell | ForEach-Object { Kids $_.ProcessId }).Count -gt 0
    }
    Set-Location $repo
    $since = $null
    for ($n = 1; ; $n++) {
        Start-Sleep -Seconds $poll
        if (-not (Alive)) { return }
        if (-not (Test-Path $request)) {
            $since = $null
            if (($n % 24) -eq 0 -and (Test-Path $python) -and -not (Busy)) {
                & $python -m lab fresh --idle *> $null   # writes the request if the conversation is due and idle
            }
            continue
        }
        if (-not $since) { $since = Get-Date }
        # let a tool call that is still running finish, but do not wait long for it
        if ((Busy) -and ((Get-Date) - $since).TotalSeconds -lt 60) { continue }
        foreach ($p in (Kids $shell)) { & taskkill.exe /T /F /PID $p.ProcessId *> $null }
        return
    }
}

function Reset-Terminal {
    # The conversation's process had no chance to tidy up: switch off the screen modes it may have left on.
    $e = [char]27
    if ($Host.UI.SupportsVirtualTerminal) {
        [Console]::Write("$e[?1049l$e[?25h$e[?2004l$e[?1000l$e[?1002l$e[?1003l$e[?1004l$e[?1006l$e[<u$e[0m")
    }
    try { [Console]::TreatControlCAsInput = $false; [Console]::CursorVisible = $true } catch { }
}

$env:LAB_ROLE = $Role
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location $tree
$Host.UI.RawUI.WindowTitle = "lab-$Role"
$exe = $Program[0]
$resume = [bool]$Continue
while ($true) {
    $all = @($Program | Select-Object -Skip 1) + @(Get-ClaudeArgs $resume)
    # The key of this launch. The guard hook and `python -m lab fresh` see it in the conversation's environment
    # and know from it that this script is there to replace the conversation.
    $env:LAB_SESSION = '{0:yyyyMMddTHHmmss}-{1}' -f (Get-Date).ToUniversalTime(), $PID
    Remove-Item $request -ErrorAction SilentlyContinue
    $watch = $null
    try {
        $watch = Start-Job -ScriptBlock $watcher -ArgumentList $PID, (Get-Process -Id $PID).StartTime.Ticks, $request, $python, $repo, $Poll
    } catch {
        $env:LAB_SESSION = ''
        Write-Warning "lab-${Role}: no background job could be started, so this conversation gets no fresh starts ($_)."
    }
    try {
        & $exe @all
    } finally {
        if ($watch) {
            Stop-Job $watch -ErrorAction SilentlyContinue
            Remove-Job $watch -Force -ErrorAction SilentlyContinue
        }
    }
    if (-not (Test-Path $request)) { break }   # it ended by itself, or the human ended it: so does this script
    Remove-Item $request -ErrorAction SilentlyContinue
    Reset-Terminal
    Write-Host ''
    if (Test-Path (Join-Path $board 'STOP')) {
        Write-Host "lab-${Role}: the lab is stopped, so no new conversation is started."
        break
    }
    Write-Host ("lab-${Role}: fresh start at {0:HH:mm}. The last conversation was ended on purpose; a new, empty one begins." -f (Get-Date))
    Start-Sleep -Seconds 3
    $resume = $false
}
