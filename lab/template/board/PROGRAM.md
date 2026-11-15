# The sandwm lab: program

The human's standing instructions to the lab's agents. Agents read this file at the start of
every tick and never edit it. The human edits it whenever they like; the next tick picks it up.

## What we are doing

We are building a world model of a small 2D falling-sand world, in the spirit of Noita and The
Powder Toy: given a few frames and the brush actions that follow, imagine what the world does
next. Two aims, in this order, for as long as the lab runs:

1. **Roughly right.** The model's imagination of the current world should look right to someone
   watching it: walls stay where they are, sand falls and piles, water finds its level, the brush
   does what it does in the simulator, nothing appears from nowhere or vanishes. It does not have
   to be perfect.
2. **Richer.** Once the current world is roughly right, the world gets one new feature (a
   material, a reaction, a behaviour, a size), and the model has to become roughly right again
   without losing what it could already do.

"Roughly right" is decided by the eval agent, with the gates in `gates.toml` and its own eyes on
the rollout pictures. It is a judgement, not a wait for the last number: when no blocker is open
and what still fails is minor, the next feature can start. One new feature is in flight at a time.

Starting point: `README.md` in your worktree describes the simulator, the model (a JEPA-style
latent world model with a reconstruction term) and what was found before the lab existed. The
reference run is `recon-k4`; its scorecard and pictures are in `{{RUNS}}\recon-k4\eval`, and
`gaps.md` lists what is known to be wrong with it.

## The agents

|               | eval | data | train |
|---------------|------|------|-------|
| looks after   | what the model actually does | the world and its data | the model and how it is trained |
| code          | evaluate.py, scorecard.py, compare.py, metrics.py, viz.py | sim.py, materials.py, scenes.py, generate.py, data.py | models.py, train.py |
| board files   | gaps.md, coverage.md, features.md, gates.toml, champion.txt | datasets.md | recipes.md |
| produces      | verdicts on runs, the list of gaps, the map of what is tested, new checks, the next feature | datasets, benchmarks, scene generators | runs |

Work flows in a circle: eval finds what is wrong and writes it in `gaps.md`; data and train take
their work from there; data delivers datasets to train; train delivers runs to eval; eval judges
them and updates the gaps. Nobody waits for a turn: each agent works on its own all the time,
and hands over with a message plus something concrete on the board or the trunk.

A fourth agent, **research**, stands outside that circle. It owns no code and has no timer: it
sleeps until someone writes to it. It is called when the lab wants to try something of a
different kind from what it has. It plans that work as a *track* beside the main line and
coordinates the three of you through it (see "Tracks"). It keeps `tracks.md` and `tracks\`.

## A tick

Each time you are woken:

1. Read this file and your role file (`roles\<you>.md`).
2. `uv run python -m lab status`. If it says your fresh start is due, see "Fresh starts".
3. Read every message in your inbox, act on it or note it, then `uv run python -m lab ack`.
4. Do the most valuable piece of work your role file describes, to a point where it is recorded
   and shared. In the first tick of a new conversation, take the fresh look first ("Fresh starts").
5. Leave a trace: `uv run python -m lab note "what I did, what I found, what comes next"`. Your
   context can be summarised at any moment, and every few hours your conversation is replaced
   by an empty one. What is not on the board or in git is lost.
6. End the turn. Do not schedule a wake-up after a tick, and never poll. You are woken by one
   of three things and by nothing else:
   - a command you started in the background ends (a cloud run, a long test);
   - another agent, or the human, sends your session a message;
   - the fallback: once every 2 hours, in case one of the first two was missed.
   Waiting costs nothing; a wake-up with nothing to do costs tokens.

   The fallback is a recurring job in your own session, and you keep exactly one. At each tick
   look with CronList. If none is listed (a new session, or it expired after its 7 days),
   create it with CronCreate: recurring, cron `M */2 * * *` with M = 11 for eval, 23 for data
   and 37 for train, and this prompt: `Lab tick (fallback wake-up). Read PROGRAM.md and your
   role file on the board, run uv run python -m lab status, handle your inbox, then carry on
   with your work.` If your session was started with a self-paced /loop, end that loop as soon
   as the job exists (ScheduleWakeup with stop set to true). If you have no CronCreate tool,
   keep the self-paced loop and always choose the longest delay it allows.
   The research agent is the exception: it keeps no fallback job and is woken by messages only.

Nobody is at the keyboard. Do not ask whether to go on: the lab runs until the human stops it.
Ending a turn because you are waiting for a run or for another agent is expected. Before you do,
make sure the agent you are waiting for knows it (a message), and that nothing in your own area
is left that you could be doing instead. When you are out of ideas, look harder at the pictures,
reread the gaps and your journal.

## Pace

Iterate fast and wide. The lab learns from how many different ideas it tests, not from how
carefully it retests one.

- **Work in batches.** Do not send one thing round the whole circle and wait for it to come
  back. Data offers several generators or mixtures at once. Train runs several different
  questions side by side, as many as `limits.toml` allows at once, with the next ones ready to
  start when a slot frees up. Eval reports every gap it sees in a run, not only the worst, so
  that data and train can work on different gaps at the same time.
- **Screen short, confirm long.** Most ideas can be judged from a short run or a small dataset.
  Only what survives screening gets a full-length run.
- **Do not wait for each other.** Hand over what you have as soon as it is usable and start the
  next thing. A quick verdict from the scorecard numbers now is worth more than a thorough one
  later; the thorough look is for runs that could become champion.
- **Do not repeat a run to feel sure.** A second seed of the same setting is the slowest way to
  learn something. See "Seeds" under the working rules.
- **Grow the world a little early rather than late**, as eval sees fit: a few rough edges are
  acceptable, a model that is plainly wrong is not.

## Fresh starts

A conversation that runs for many hours narrows. Each step builds on the one before, and in the
end an agent only polishes the idea it already has, in smaller and smaller steps. So every
agent's conversation is replaced by an empty one about every `session_hours` hours
(`limits.toml`; data's and train's turns come a little later than eval's, so that you do not
all start afresh at once). The board, the code and the runs stay. What you had in mind and did
not write down is gone, and that is the point: the next conversation sees the work as it is,
not as you remember deciding it.

**When `lab status` says your fresh start is due:**

1. Start no new cloud job: the ledger refuses one from a conversation that is due. Let the
   jobs you have running finish, and record their results as always.
2. Commit what is in your worktree, even unfinished. It need not be published.
3. As the last thing you do: `uv run python -m lab fresh "<handover>"`. The handover is what
   the board does not say yet: what is half done and where it is, what you promised to whom,
   what you are waiting for. It goes into your journal. The conversation ends within seconds
   and a new one starts in the same terminal, so call no other tool after it.

You may ask for a fresh start before it is due, in the same way, when you notice that you are
going round in circles. A conversation that is due and sits idle with nothing running is
replaced without being asked. If `fresh` is refused as an invalid choice, your worktree is
older than this rule: run `uv run python -m lab sync` once.

**The first tick of a new conversation** begins with a fresh look, before you take up what the
last conversation left:

1. Read the evidence before the plans: `gaps.md`, `coverage.md`, the champion's scorecard and
   pictures, then `recipes.md`, `datasets.md`, `features.md` and `tracks.md`.
2. Write it down (`lab note "FRESH LOOK: ..."`): what holds the model back most, on the
   evidence; what the recent work in your area keeps circling around; and two or three things
   you could do that differ from it in kind, not in degree: another cause for the same gap,
   another part of the pipeline, a simpler way, something to drop.
3. Only then read the last notes in your journal, the handover among them, and choose.
   Carrying on as before is right when it wins that comparison; say why in a note. When
   another direction wins, start it in your next batch, beside what is already running.

## What the evaluation does not see

The gates speak only for the situations the benchmarks contain. A model can pass every gate and
still be wrong where nobody has looked: a rollout of a thousand frames, a long silence followed
by a burst of brush strokes, a cluttered scene, sand and water mixed cell by cell. A failure
that no benchmark provokes is never listed, so nobody gets the chance to fix it.

So eval keeps a second list beside `gaps.md`: `coverage.md`, the situations and measurements
the evaluation does not cover yet. Everyone adds to it, with a message to eval: data knows what
the scene generators and the brush never produce, train knows where the model is fragile, the
human sees things while playing. Eval turns an entry into a small frozen probe benchmark (made
by data) and one or two checks. What the champion fails there becomes an ordinary gap in
`gaps.md`. "Roughly right" means roughly right where the lab has looked, so the lab keeps
looking in new places. This runs beside the other work and does not hold up a feature, unless a
probe shows the model plainly wrong.

## Tracks

Most ideas are a setting or an option of what the lab already has, and whoever owns that area
simply tries them in a batch. Some are a different kind of thing: a new kind of task (an agent
that learns to act in this world with reinforcement learning, planning towards a goal), another
family of model, a new way of training (inside the model's imagination, across several worlds),
or anything that needs new work from two or three agents at once. Those go through the research
agent, as a **track**.

- **Proposing.** Anyone, the human included, sends research the idea and why it is worth trying:
  `uv run python -m lab msg research "<idea>" --body "..."`, then the doorbell to `lab-research`.
  (If `research` is refused as an invalid choice, your worktree is older than the research
  agent: run `uv run python -m lab sync` once.) Research answers one of three things: do it
  yourself, it is not track-sized; later, with the reason; or a plan.
- **The plan.** Research writes `tracks\T-00N-<name>.md` and lists the track in `tracks.md`: what
  the idea is, what success means and how it is measured, and a short sequence of stages. A
  stage is small enough for one batch, leaves the trunk working, has an allowance of GPU
  minutes, and ends in a number that says go on or stop. An idea too complicated to try in one
  go simply has more stages. Only the next stage is planned in detail; the rest is revised when
  results arrive. Research sends each agent its part of that stage. You own your area as before:
  if the plan is wrong for it, say so and the plan changes.
- **The main line goes on.** A track is a side line, never a replacement. The gaps, the gates
  and the feature in flight keep at least two thirds of every agent's work and of every batch,
  and blockers and regressions on the main line come before any track. `limits.toml` sets the
  GPU minutes all tracks together may use (`gpu_minutes_tracks`), and the ledger refuses a track
  run beyond it. At most two tracks are active at once.
- **Kept apart.** Track runs are named `t00N-<what>` (main-line runs stay `r....`), track
  datasets `t00N-...`, track benchmarks `bench-t00N-...`. Code for a track is an option that is off
  by default, or a module of its own, so the main recipe behaves exactly as before. Every track
  run has a control without the idea and is also scored on the main benchmarks: with and
  without are always compared. A track run does not become champion while the track is open,
  and what is wrong with a track is recorded in the track's file, not in `gaps.md`.
- **Reporting.** When your part of a stage is done, or its allowance is used up, tell research
  (message and doorbell). It has no timer and sleeps until someone writes to it, so a stage
  nobody reports on never moves.
- **Ending a track.** Research recommends one of three ends. *Adopt*: the idea met its own
  measure and did not make the main gates worse beyond noise; eval decides, the option joins
  the main recipe and its checks join the scorecard. *Park*: not now, with what was learnt and
  what would make it worth another look. *Abandon*: it does not work, or it damages the rest;
  code that served only the track is removed (the run tags keep it). An idea that degrades
  everything is dropped without regret, and a parked one can be reopened by anyone who brings
  new evidence.

## Talking to each other

- `uv run python -m lab msg <eval|data|train|research|human> "<subject>" --body "..."` puts a message file
  in their inbox. Then ring the doorbell: with the SendMessage tool, send the session
  `lab-<role>` one line such as "lab: a message is waiting in your inbox". The file is the
  record; the doorbell only wakes them. If the session is not listed, skip the doorbell: they
  find the message at their next tick.
- A message is short and concrete: what happened, where the evidence is (run, file, check), and
  what you ask of them.
- Facts that others rely on live in board files, not in messages.
- If `lab status` shows another agent without a tool call for more than 3 hours and the lab is
  not stopped, send it one doorbell: "lab: wake up and run your tick". If it stays silent, tell
  the human once.

## Hard rules

1. **Money.** Cloud GPU jobs are started only with `uv run python -m lab train`. It books the job
   against the budget in `limits.toml` and refuses a job that does not fit. When refused: stop
   launching, do the work that needs no GPU, and tell the human once if more budget would change
   what the lab can achieve. Never look for another route. Datasets are simulated on cloud CPUs
   with `uv run python -m lab generate`: that costs cents, not nothing, so no dataset is larger
   than its purpose needs.
2. **Yours and not yours.** You write in your own worktree, in your own files on the board, and
   in the shared folders `{{RUNS}}` and `{{DATA}}`. Nowhere else on this machine: not the main
   checkout `{{HOME}}`, not another agent's worktree, not the human's other folders. On Modal you
   use only the app and the volume named `sandwm`; the workspace also holds the human's other
   projects, which you do not inspect, stop or touch. Never print or copy credentials.
3. **The human's files.** On the board: `PROGRAM.md`, `roles\`, `limits.toml`, `ledger.jsonl`. In
   the code: `lab\` and `cloud.py`. If one of them is wrong or in your way, put the exact change
   you want in the human's inbox; do not make it.
4. **The trunk stays green.** Code reaches the others only through
   `uv run python -m lab publish`, which merges the trunk (branch `{{TRUNK}}`) into your branch,
   runs every test and then moves the trunk. Never commit to `main`. Cloud jobs run published
   code only, so that everyone can load what they produce.
5. **Benchmarks are frozen.** A `bench-*` dataset is never trained on, never changed, and never
   regenerated under the same name. Seeds from 9000 up are reserved for benchmarks.
6. **Stop means stop.** If a tool call is refused with STOP, end your turn at once.
7. **Nothing from outside.** No new dependencies, services, accounts, or downloaded models and
   datasets. Everything is built from what is in the repository.

A guard checks some of this before each of your tool calls. It is there to catch slips. If it
blocks something you believe is right, tell the human; do not work around it.

## Working rules

- **Evidence.** Every claim on the board names its evidence: the run, the check and its value,
  the picture file and window. "Looks better" is not a finding.
- **Seeds.** A difference below 0.02 on a check, or below the run-to-run noise eval has recorded
  for that check, is noise. A difference several times larger needs no second seed: act on it.
  Repeat a run only for a close call that would change the champion, and start the repeat
  alongside new experiments, never as a turn of its own.
- **One question per run, several runs at once.** Each run changes one thing against a known
  reference, and its `--note` states the hypothesis before it starts, so a batch of runs answers
  a batch of questions. A negative result is recorded as carefully as a positive one.
- **Cheapest test first.** A crash is found on the laptop in seconds (`--steps 4` with a tiny
  model), a dead idea in a short cloud run. Full-length runs are for confirming.
- **Simplicity.** A small gain that adds a lot of code is not worth keeping, and the same result
  with less code is a win. Options that lost are removed; the git tag `run/<name>` keeps the
  exact code of every run.
- **Regressions first.** A check the champion passes and a new run fails matters more than an
  improvement elsewhere.
- **Stay in your area.** When the fix belongs in someone else's code, send them the request and
  the evidence. A minimal edit in another agent's file is acceptable only to keep the trunk
  working, and you tell them at once.
- **Go easy on this laptop.** It has no GPU and the three of you share its CPU. No local training
  beyond smoke tests, `--threads 2` for anything local, one heavy job at a time.

## Environment

- Windows 11, CPU only. Run Python through uv and nothing else: `uv run python ...`,
  `uv run pytest`. The machine's own Python breaks PyTorch.
- Your worktree is a complete checkout with its own `.venv`. Run `uv run python -m lab ...` from
  its root.
- Change source files with the editing tools, not with shell heredocs or `sed`: escapes have
  been mangled that way before.
- Modal tries GPU types in the order L4, A10G, T4. The workspace's GPU limit is shared with the
  human's other projects, so a job can wait in a queue before it starts. The reference recipe
  (20,000 steps) took about 6 minutes on an A10G.
- `lab train` stays in the foreground until the job has ended and its results are fetched. Start
  it as a background command and you are woken when it exits. Ending the command stops the cloud
  job.
- Datasets and runs are on the Modal volume `sandwm` as `data/<name>` and `runs/<name>`; local
  copies go to `{{DATA}}` and `{{RUNS}}` (`lab fetch`). The runs described in `README.md` are on
  the volume too; of those, only `recon-k4` has been scored on the benchmark.

## The board

`{{BOARD}}` is the lab's memory. Every file has one keeper; `lab save` records hand edits.

| file | keeper | what it holds |
|------|--------|---------------|
| `PROGRAM.md`, `roles\`, `limits.toml` | human | these instructions and the budget |
| `ledger.jsonl` | the tooling | every cloud job and its cost |
| `gaps.md` | eval | everything known to be wrong with the model, most important first |
| `coverage.md` | eval | what the evaluation does not cover yet: situations, measurements, numbers not to be trusted |
| `gates.toml` | eval | the thresholds that define "roughly right" |
| `champion.txt` | eval | the best run so far |
| `features.md` | eval | what the world learns next |
| `datasets.md` | data | every dataset: purpose, how it was made, what is known about it |
| `recipes.md` | train | the reference recipe, what was tried against it, what is planned |
| `tracks.md`, `tracks\` | research | large ideas tried beside the main line: their plans and what came of them |
| `journal\<role>.md` | each agent | dated notes (`lab note`) |
| `inbox\<role>\` | everyone | messages (`lab msg`); handled ones move to `done\` |
