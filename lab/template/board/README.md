# The board

The shared memory of the sandwm lab: three Claude Code agents (eval, data, train) that improve
a world model of a falling-sand world. Start with `PROGRAM.md`.

This folder is a git branch of its own (`lab-board`), so `git log` here shows who changed what
and when, and nothing that gets overwritten is lost.

    PROGRAM.md      the human's standing instructions          (human)
    roles\          what each agent does                       (human)
    limits.toml     the GPU budget                             (human)
    ledger.jsonl    every cloud job and what it cost           (written by the tooling)
    gaps.md         what the model gets wrong, most important first   (eval)
    coverage.md     what the evaluation does not cover yet             (eval)
    gates.toml      what "roughly right" means in numbers      (eval)
    champion.txt    the best run so far                        (eval)
    features.md     what the world learns next                 (eval)
    datasets.md     every dataset and what it is for           (data)
    recipes.md      the reference recipe and what was tried    (train)
    tracks.md       large ideas tried beside the main line     (research)
    tracks\         one plan per track                         (research)
    journal\        each agent's dated notes
    inbox\          messages; handled ones move to done\
    status\         when each agent last made a tool call (not recorded in git)
    STOP            present while the human has stopped the lab (not recorded in git)

One screen of all of it: `uv run python -m lab status`.
