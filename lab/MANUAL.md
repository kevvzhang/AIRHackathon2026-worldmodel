# The agent lab: manual

The commands alone are in [README.md](README.md). This file explains how the lab works.

Three Claude Code agents improve the world model in a loop, in the spirit of Karpathy's
autoresearch: the human writes the program, the agents do the research. A fourth plans the
ideas that are too different to fit in that loop.

| agent | looks after | hands over |
|-------|-------------|------------|
| **eval** | what the model actually does: scores every run, looks at the rollout pictures, keeps the list of gaps, decides when the world is "roughly right" and what feature comes next | gaps, verdicts, new checks |
| **data** | the simulator and the datasets: data aimed at the gaps, new world features | datasets, benchmarks |
| **train** | the model and the recipe: architecture, objective, launching and reading runs | runs |
| **research** | large ideas of a different kind (an agent acting in the world with RL, another family of model): plans each as a staged side line, a *track*, and coordinates the other three through it. Owns no code, starts no jobs | plans, go/stop calls |

Each agent is an ordinary interactive Claude Code session (Opus 5.5, effort xhigh by default)
in its own terminal tab, working in its own git worktree. It is woken by its own finished jobs
and by messages from the other agents, with a fallback check every 2 hours. The research agent
has no fallback: it sleeps until someone writes to it. About every 6 hours an agent's
conversation is replaced by an empty one (see "Fresh starts").

## Start, watch, stop

    uv run python -m lab setup          # once: board, trunk, worktrees, benchmark (safe to repeat)
    .\lab\launch.ps1 -DryRun            # print exactly what would be started
    .\lab\launch.ps1                    # start the four agents (one Windows Terminal tab each)

    uv run python -m lab status         # one screen: budget, runs, inboxes, who is alive, open gaps
    uv run python -m lab runs           # every cloud job, its cost and its gates
    git -C .lab\board log --stat        # who changed what on the board, and when

    uv run python -m lab stop           # stop: agents are refused at their next tool call, cloud jobs are stopped
    uv run python -m lab go             # lift the stop; then launch again
    .\lab\launch.ps1 -Roles train       # start or restart a single agent (add -Continue to keep its conversation)

    uv run python -m lab snapshot <name>   # name the lab as it is now (the trunk's code and the board)
    uv run python -m lab restore <name>    # go back to a snapshot (stop the lab first)

Run `launch.ps1` from your own terminal, not from inside a Claude Code session: an agent started
there is that session's child, has no address of its own, and cannot be woken by the others
(the launcher refuses). The first time an agent starts in its worktree, Claude Code may ask
whether you trust the folder. The laptop has to stay awake: a sleeping machine runs nothing, and a cloud job whose
terminal went away is stopped.

## How you steer it

Everything the agents are told lives on the board, `.lab\board`, and they reread it every tick:

- `PROGRAM.md` is the program: the goal, the rules, how to work. Edit it and the next tick follows it.
- `roles\eval.md`, `roles\data.md`, `roles\train.md` are the three job descriptions.
- `limits.toml` is the money: total GPU minutes, the longest one job may run, how many at once.
- To tell one agent something: `uv run python -m lab msg train "subject" --body "..."`, or type
  into its terminal tab.
- To put a large new idea into the lab: `uv run python -m lab msg research "idea: ..." --body "..."`
  and then say "you have a message" in the research agent's tab (it has no timer). See "Tracks".
- Agents write to you with `lab msg human`; `lab status` shows how many messages are waiting, and
  they are files in `.lab\board\inbox\human`.

## Where things are

    lab\                 this tooling (tracked in git): launch.ps1, agent.ps1, the `python -m lab` commands, templates
    .lab\                the running lab (ignored by git)
      board\             shared state, on its own branch `lab-board`: the program, gaps, gates, ledger, messages, journals
      eval\ data\ train\ one git worktree per agent, on branches lab-eval, lab-data, lab-train
      runs\              runs fetched from the cloud volume (scorecards, pictures, checkpoints)
      datasets\          local copies of datasets and the benchmark

Agents never touch `main` or this checkout. They publish to the branch `lab-trunk` with
`python -m lab publish`, which merges the trunk, runs the tests and only then moves it. To take
their work: `git merge lab-trunk` here. To give them a change you made on `main`:
`git merge lab-trunk`, then `git branch -f lab-trunk main` (never check `lab-trunk` out; nothing
should hold it).

## Tracks: large ideas beside the main line

Some ideas are not a setting of what exists but a different kind of thing: an agent that learns
to act in the world, a different family of model. Trying one must not stop the main work, may
turn out to damage everything, and is often too big for one step. So such an idea becomes a
*track*, run by the research agent (`PROGRAM.md`, "Tracks", and `roles\research.md`):

- anyone, you included, proposes it to research; research answers "just try it", "later", or
  writes a plan in `.lab\board\tracks\` and lists it in `tracks.md`;
- the plan is a sequence of small stages, each with a GPU allowance and a number that says go on
  or stop; only the next stage is planned in detail;
- the main line keeps at least two thirds of every agent's work, and track runs (named
  `t001-...`) can use at most `gpu_minutes_tracks` from `limits.toml`: the ledger refuses more;
- track code is off by default, track runs are compared with a control and scored on the main
  benchmarks too, and a track run never becomes champion while its track is open;
- a track ends adopted (eval decides; it joins the main recipe), parked, or abandoned.

`uv run python -m lab status` lists the tracks and what they have spent.

## Fresh starts: clearing an agent's context

An agent that works in one conversation for many hours narrows: each step builds on the one
before, and it ends up polishing the idea it already has. So each agent's conversation is
replaced by an empty one about every `session_hours` hours (`limits.toml`, 6 by default, 0 =
never; data's limit is 10% longer and train's 20%, so that the three do not start afresh at
the same moment). Claude Code has no way to clear a running conversation without someone
typing `/clear`, so the launcher ends the session's process and starts a new one in the same
tab:

- The guard hook keeps a clock for each conversation in `.lab\board\status\<role>.json`: when it
  began, how many tool calls it has made, when the last one was. `lab status` shows the age.
- A conversation is due once it is older than the limit and has made at least 50 tool calls
  (one that has hardly worked is left alone, which keeps the sleeping research agent asleep).
- `lab status` then tells the agent. It starts no new cloud job (the ledger refuses), lets its
  running jobs finish, commits its work and runs `python -m lab fresh "<handover>"`. The
  handover goes into its journal.
- `lab\agent.ps1`, which stays in the tab for as long as the agent runs, sees the request,
  ends the session and starts a new one. It also asks on the agent's behalf when a due
  conversation has made no tool call for 20 minutes and has nothing running.
- The new conversation reads the evidence before the old plans, writes down what it would do
  if it started today ("FRESH LOOK" in its journal), and only then decides whether to carry on
  as before (`PROGRAM.md`, "Fresh starts").

What it costs: the first tick of a new conversation is longer, and an agent that is due stops
launching until its last job is in, so a few GPU slots can stand empty for a while. A
conversation started by an older version of the launcher cannot be replaced: `lab status` says
so, and restarting that agent once (close its tab, `.\lab\launch.ps1 -Roles <role>`) fixes it.

## Going back

A snapshot is two git tags, `snapshot/<name>/code` on the trunk and `snapshot/<name>/board` on
the board. The state before the first launch is the snapshot `prelaunch`.

    uv run python -m lab stop
    uv run python -m lab restore prelaunch
    uv run python -m lab go

`restore` puts the trunk, the three agent branches and the board back to the snapshot. Nothing
is thrown away: whatever you are leaving, uncommitted work in the agents' worktrees included, is
first tagged `kept/<time>/trunk`, `/eval`, `/data`, `/train` and `/board`. Three things are never
rewound:

- the ledger: GPU minutes already spent stay spent;
- `main`: agents never change it, so it is the pre-launch code for as long as you do not merge
  `lab-trunk` into it (if you did, `git reset --hard snapshot/prelaunch/code` on `main` undoes it);
- the cloud volume: runs and datasets made since stay there. They are only ever added: a
  training run refuses a name that already holds a model, and a dataset is never regenerated
  under an existing name.

## What a run goes through

`python -m lab train <name> --data <dataset> -- <train args>` requires committed, published code,
books GPU minutes in the ledger, tags the commit as `run/<name>`, trains on Modal, evaluates,
scores the run on the frozen benchmark (`sandwm.scorecard`: about thirty named checks such as
`walls/kept@48` or `mass/sand@48`, on the same windows for every run), fetches the results into
`.lab\runs\<name>`, and prints the scorecard next to the champion's. The eval agent then looks at
the pictures in `eval\bench\`, updates `gaps.md`, and adds a check for every new way of being
wrong, so a later run that breaks it again is caught by the numbers.

Passing every gate only says the model is right where the benchmarks look. So eval keeps a
second list, `coverage.md`: situations the benchmarks do not contain (very long rollouts,
intermittent actions, messy scenes), properties no check measures, numbers that cannot be
trusted yet. It turns each entry into a small probe benchmark and checks, and what the champion
fails there becomes an ordinary gap. `lab status` shows how many entries are still uncovered.

## What is enforced, and by what

| rule | enforced by |
|------|-------------|
| total GPU minutes, minutes per job, jobs at once | `lab/budget.py`, called by every entry point of `cloud.py` before anything starts; a job that never reports back is charged its full booking, unless its launcher's heartbeat stopped: then it is "cut" and charged up to the last heartbeat plus 5 minutes |
| one job cannot run longer than the per-job limit | Modal's function timeout, set from `limits.toml` |
| stop | the guard hook refuses every tool call but reads while `STOP` is on the board; `lab stop` also stops the `sandwm` app on Modal |
| agents write only in their worktree, their own board files and the shared folders | the guard hook (`lab/guard.py`, a PreToolUse hook in each worktree's `.claude\settings.local.json`), for the file-editing tools |
| no direct `modal run`, no touching the limits, the ledger, credentials, other Modal projects, `git push`, branch surgery | the guard hook, by matching shell commands |
| tests pass before code reaches the trunk; `lab/` and `cloud.py` are not changed by agents | `python -m lab publish` |
| research tracks use at most `gpu_minutes_tracks` in total | `lab/budget.py`, by the run name (`t001-...`) |
| a conversation is replaced by an empty one after `session_hours` | `lab/agent.ps1` ends the session and starts a new one, when the agent asks (`lab fresh`) or when it is due and idle; `lab/jobs.py` refuses new cloud jobs from a conversation that is due |
| the handover, and the fresh look a new conversation begins with | the instructions only |
| the share of work tracks get, when a track is adopted or dropped | the instructions only |
| everything else | the session's permission mode (`auto` by default: Claude Code's safety classifier) and the instructions |

The guard catches slips; it is not a sandbox, and a shell command can get around it. The limits
an agent cannot reach are on Modal's side: a workspace budget or spend limit under Settings >
Usage and billing in the Modal dashboard.

## What it costs

- **GPU**: at most `gpu_minutes_total` in `limits.toml` (300 minutes by default, roughly $6-8
  on L4/A10G including the container's CPU and memory) per start of the lab. Dataset generation
  on cloud CPUs adds cents. Of that total, research tracks may use at most `gpu_minutes_tracks`.
  The total is counted per budget period. `launch.ps1` begins a new period, with the count back
  at zero, whenever it starts the lab while no lab agent is running; `-KeepBudget` prevents
  that, and `uv run python -m lab budget --new` does it by hand. Starting one agent beside
  others that are still running does not begin a new period, and neither do the fresh starts.
  So the total caps one run of the lab, not all of them together: every start from a standstill
  allows that many minutes again. Earlier periods stay in the ledger (`lab budget` shows what
  they used).
- **Claude usage**: three Opus sessions at effort xhigh. An agent with nothing to do sleeps
  until a job of its own ends or another agent messages it, and otherwise wakes once every
  2 hours as a fallback. Usage is still by far the larger cost, and it comes out of your Claude
  plan or API budget, not out of `limits.toml`.
  Nearly all of it is re-reading: at every step an agent reads its whole conversation again. In
  the lab's first 9.5 hours the three agents read 767 million tokens that way, in conversations
  that grew to 960,000 tokens. Two settings in `limits.toml` bound it: `session_hours` (a new,
  empty conversation, see "Fresh starts") and `compact_tokens` (Claude Code summarises the
  conversation once it holds that many tokens; the launcher passes it as `claude --autocompact`).
  Replayed on those 9.5 hours, 500,000 would have roughly halved the tokens read, and 300,000
  would have cut them by about 60%, with about twice as many summaries.

## Known limits of this setup

- Waking is event-driven: a finished background command wakes the agent that started it, and a
  message from another agent wakes an idle session. The only timer is a fallback: each agent
  keeps one recurring job in its own session that fires every 2 hours (`PROGRAM.md`, "A tick").
  A session is launched with a self-paced `/loop` and ends that loop once the job exists. The
  job lives only in the session: it is gone if the session is restarted (the agent creates it
  again at its first tick) and it expires after seven days (likewise).
- `lab status` shows each agent's last tool call. An idle agent can be silent for over 2 hours;
  restart one that stays silent with `.\lab\launch.ps1 -Roles <role>`. Agents are told to wake
  a peer silent for more than 3 hours, which does not help when all three hit the same usage
  limit.
- One poke between agents is a file in the inbox plus a cross-session message that wakes the
  receiving session. If the message is not delivered, the file is found at the next tick, which
  can now be up to 2 hours away.
- All three agents share this laptop's CPU and one Modal GPU quota (also used by your other
  projects), so jobs can queue.
- Fresh starts end the session's process abruptly; there is no gentler way from outside. They
  have been tested with a stand-in program in place of Claude Code, not yet with a real
  session: watch the first one. A background command the agent forgot about dies with the
  conversation (cloud GPU jobs are checked for, other commands are not).
- `-Continue` resumes the agent's last conversation by the id the guard recorded
  (`claude --continue` skips conversations that began with `/loop`); this has not been tried on
  a real session yet. Its scheduled fallback job comes back if it has not expired; its
  background commands do not.
