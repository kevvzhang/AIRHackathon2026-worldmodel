# The agent lab: commands

Run these from `C:\UoE\Research\WorldModel`, in your own terminal (not inside Claude Code).
How it all works is in [MANUAL.md](MANUAL.md).

## Start and stop

```powershell
.\lab\launch.ps1                          # start all four agents, one tab each
.\lab\launch.ps1 -Roles research          # start or restart one: eval, data, train or research
.\lab\launch.ps1 -Roles train -Continue   # restart one and keep its conversation
.\lab\launch.ps1 -DryRun                  # show what would start, start nothing
uv run python -m lab stop                 # stop all agents and the lab's cloud jobs
uv run python -m lab go                   # lift the stop; then launch again
```

To restart an agent by hand, close its tab first, then start it again. A cloud job it has
running ends with the tab.

The GPU budget is counted per start of the lab. When `launch.ps1` runs while no lab agent is
running, the count goes back to zero and the lab may spend `gpu_minutes_total` again. Starting
one agent beside others that are still running keeps the count, and so do the fresh starts.

```powershell
.\lab\launch.ps1 -KeepBudget              # start the lab but let the old count go on
uv run python -m lab budget --new         # begin a new count now, without restarting anything
```

## Fresh starts

About every 6 hours an agent's conversation is replaced by an empty one, in the same tab, so
that it does not get stuck polishing one idea. The agent leaves a handover in its journal
first, and the new conversation begins by looking at the evidence afresh. Nothing to do:
`lab status` shows each conversation's age and when its turn comes. The period is
`session_hours` in `limits.toml`; 0 switches it off.

In between, Claude Code summarises (compacts) a conversation once it holds `compact_tokens`
tokens (500,000). An agent re-reads its whole conversation at every step, so this is what
keeps the token use down; a lower number saves more and forgets more.

## Watch

```powershell
uv run python -m lab status               # everything on one screen
uv run python -m lab runs                 # every cloud run: cost, state, gates passed
uv run python -m lab budget               # GPU minutes spent and left
uv run python -m lab inbox                # messages the agents sent you
uv run python -m lab ack                  # mark them as read
git -C .lab\board log --stat              # who changed what on the board
```

A run's pictures and scores are in `.lab\runs\<run>\eval\`. To compare two runs:

```powershell
cd .lab\eval
uv run python -m sandwm.scorecard --diff ..\runs\<old> ..\runs\<new>
```

## Steer

```powershell
uv run python -m lab msg train "subject" --body "..."         # tell one agent something
uv run python -m lab msg research "idea: ..." --body "..."    # propose a large new idea
```

After a message to research, also say "you have a message" in its tab: it has no timer.

The agents reread these files in `.lab\board\` at every tick. Edit one, then run
`uv run python -m lab save -m "what changed"`.

| file | what it controls |
|------|------------------|
| `PROGRAM.md` | the rules for all agents |
| `roles\eval.md`, `data.md`, `train.md`, `research.md` | what each agent does |
| `limits.toml` | the budget (below) |

| in `limits.toml` | meaning |
|------------------|---------|
| `gpu_minutes_total` | the GPU minutes the lab may spend per start (see "Start and stop") |
| `gpu_minutes_per_run` | the longest one job may run |
| `max_concurrent_gpu_runs` | GPU jobs running at once; a job started beyond that waits in line and starts by itself. Keep it at or below what Modal allows your workspace (10 GPUs on the Starter plan, shared with your other projects) |
| `gpu_minutes_tracks` | the part of the total that research tracks may use |
| `cpu_minutes_per_job` | the longest one data-generation job may run |
| `session_hours` | hours before an agent's conversation is replaced by an empty one (0 = never) |
| `compact_tokens` | tokens a conversation may hold before it is summarised (0 = Claude Code's default, about 967,000) |

## Go back

```powershell
uv run python -m lab snapshot <name>      # name the lab as it is now
git tag --list "snapshot/*"               # list snapshots (so far: prelaunch, before-research)
uv run python -m lab stop
uv run python -m lab restore <name>       # go back; what you leave is kept under kept/<time>/... tags
uv run python -m lab go
```

## Take the agents' work

```powershell
git merge lab-trunk                       # their code into main; they never touch main themselves
```

## Rarely needed

```powershell
uv run python -m lab setup                # recreate the board and worktrees; safe to repeat
uv run python -m lab fetch <run>          # download a run from the cloud into .lab\runs
```
