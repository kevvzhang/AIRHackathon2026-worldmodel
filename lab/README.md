# The agent lab

Three Claude Code agents improve the world model in a loop, in the spirit of Karpathy's
autoresearch: the human writes the program, the agents do the research.

| agent | looks after | hands over |
|-------|-------------|------------|
| **eval** | what the model actually does: scores every run, looks at the rollout pictures, keeps the list of gaps, decides when the world is "roughly right" and what feature comes next | gaps, verdicts, new checks |
| **data** | the simulator and the datasets: data aimed at the gaps, new world features | datasets, benchmarks |
| **train** | the model and the recipe: architecture, objective, launching and reading runs | runs |

Each agent is an ordinary interactive Claude Code session (Opus 5.5, effort xhigh by default)
in its own terminal tab, working in its own git worktree and waking itself with `/loop`.

## Start, watch, stop

    uv run python -m lab setup          # once: board, trunk, worktrees, benchmark (safe to repeat)
    .\lab\launch.ps1 -DryRun            # print exactly what would be started
    .\lab\launch.ps1                    # start the three agents (one Windows Terminal tab each)

    uv run python -m lab status         # one screen: budget, runs, inboxes, who is alive, open gaps
    uv run python -m lab runs           # every cloud job, its cost and its gates
    git -C .lab\board log --stat        # who changed what on the board, and when

    uv run python -m lab stop           # stop: agents are refused at their next tool call, cloud jobs are stopped
    uv run python -m lab go             # lift the stop; then launch again
    .\lab\launch.ps1 -Roles train       # start or restart a single agent (add -Continue to keep its conversation)

The first time an agent starts in its worktree, Claude Code may ask whether you trust the
folder. The laptop has to stay awake: a sleeping machine runs nothing, and a cloud job whose
terminal went away is stopped.

## How you steer it

Everything the agents are told lives on the board, `.lab\board`, and they reread it every tick:

- `PROGRAM.md` is the program: the goal, the rules, how to work. Edit it and the next tick follows it.
- `roles\eval.md`, `roles\data.md`, `roles\train.md` are the three job descriptions.
- `limits.toml` is the money: total GPU minutes, the longest one job may run, how many at once.
- To tell one agent something: `uv run python -m lab msg train "subject" --body "..."`, or type
  into its terminal tab.
- Agents write to you with `lab msg human`; `lab status` shows how many messages are waiting, and
  they are files in `.lab\board\inbox\human`.

## Where things are

    lab\                 this tooling (tracked in git): launch.ps1, agent.ps1, the `python -m lab` commands, templates
    .lab\                the running lab (ignored by git)
      board\             shared state, on its own branch `lab-board`: the program, gaps, gates, ledger, messages, journals
      eval\ data\ train\ one git worktree per agent, on branches lab-eval, lab-data, lab-train
      runs\              runs fetched from the cloud volume (scorecards, pictures, checkpoints)
      datasets\          local copies of datasets and the benchmark

Agents never touch `main` or this checkout. They publish to the branch `lab` (the trunk) with
`python -m lab publish`, which merges the trunk, runs the tests and only then moves it. To take
their work: `git merge lab` here. To give them a change you made on `main`: `git merge lab`, then
`git branch -f lab main` (never check `lab` out directly; nothing should hold it).

## What a run goes through

`python -m lab train <name> --data <dataset> -- <train args>` requires committed, published code,
books GPU minutes in the ledger, tags the commit as `run/<name>`, trains on Modal, evaluates,
scores the run on the frozen benchmark (`sandwm.scorecard`: about thirty named checks such as
`walls/kept@48` or `mass/sand@48`, on the same windows for every run), fetches the results into
`.lab\runs\<name>`, and prints the scorecard next to the champion's. The eval agent then looks at
the pictures in `eval\bench\`, updates `gaps.md`, and adds a check for every new way of being
wrong, so a later run that breaks it again is caught by the numbers.

## What is enforced, and by what

| rule | enforced by |
|------|-------------|
| total GPU minutes, minutes per job, jobs at once | `lab/budget.py`, called by every entry point of `cloud.py` before anything starts; a job that never reports back is charged its full booking |
| one job cannot run longer than the per-job limit | Modal's function timeout, set from `limits.toml` |
| stop | the guard hook refuses every tool call but reads while `STOP` is on the board; `lab stop` also stops the `sandwm` app on Modal |
| agents write only in their worktree, their own board files and the shared folders | the guard hook (`lab/guard.py`, a PreToolUse hook in each worktree's `.claude\settings.local.json`), for the file-editing tools |
| no direct `modal run`, no touching the limits, the ledger, credentials, other Modal projects, `git push`, branch surgery | the guard hook, by matching shell commands |
| tests pass before code reaches the trunk; `lab/` and `cloud.py` are not changed by agents | `python -m lab publish` |
| everything else | the session's permission mode (`auto` by default: Claude Code's safety classifier) and the instructions |

The guard catches slips; it is not a sandbox, and a shell command can get around it. The limits
an agent cannot reach are on Modal's side: a workspace budget or spend limit under Settings >
Usage and billing in the Modal dashboard.

## What it costs

- **GPU**: at most `gpu_minutes_total` in `limits.toml` (300 minutes by default, roughly $6-8
  on L4/A10G including the container's CPU and memory). Dataset generation on cloud CPUs adds cents.
- **Claude usage**: three Opus sessions at effort xhigh that never stop. An idle agent still
  wakes at least every 30 minutes and rereads its instructions. This is by far the larger cost,
  and it comes out of your Claude plan or API budget, not out of `limits.toml`.

## Known limits of this setup

- The loop is Claude Code's self-paced `/loop`. It ends if a wake-up fails twice in a row (for
  example while a usage limit is in force), it is not restored by `--continue`, and it expires
  after seven days. `lab status` shows each agent's last tool call; restart a silent one with
  `.\lab\launch.ps1 -Roles <role>`. Agents are told to wake a peer that has gone quiet, which
  does not help when all three hit the same limit.
- One poke between agents is a file in the inbox plus a cross-session message that wakes the
  receiving session. If the message is not delivered, the file is found at the next tick.
- All three agents share this laptop's CPU and one Modal GPU quota (also used by your other
  projects), so jobs can queue.
- The agents themselves have not been run yet. The tooling is tested (`uv run pytest`), but the
  first launch is the first test of the prompts and of the `/loop` and wake-up behaviour: watch
  the first hour.
