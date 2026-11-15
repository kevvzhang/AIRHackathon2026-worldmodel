# Coverage: what the evaluation does not see yet

Kept by the eval agent. `gaps.md` lists what the model gets wrong where the lab looks. This file
lists where the lab is not looking: situations the benchmarks do not contain, properties no
check measures, numbers that cannot be trusted yet. A failure nobody provokes is never found,
so nobody can fix it. How to work with this list is in `roles\eval.md`, "What the evaluation
does not see". Anyone may suggest an entry by telling eval.

Headings must keep the form `### C-001 | title | priority | status`, with priority `high`,
`medium` or `low`, and status `open`, `asked` (a benchmark or check is on its way) or `covered`
(say by which benchmark and checks, and what the champion scored). Most urgent first.

The first entries come from the human. Eval: check them, fill in what exists, decide the order,
and add your own.

## Not covered yet

### C-001 | very long rollouts: hundreds to thousands of frames | high | open
- asked by: the human
- not tested: what the imagination does far beyond the distances that are scored. Does the
  picture drift, do walls wear away, does material multiply or vanish, does it settle into
  something that is no longer a world?
- closest so far: `bench-basic-v1` is read 16 and 48 frames ahead.
- to cover it: eval decides. Longer episodes from data, read at several distances, with the
  checks that exist (walls kept, mass) and one for drift over time.

### C-002 | intermittent actions | high | open
- asked by: the human
- not tested: patterns of brush use the scripted brush does not produce. A long silence and then
  a burst of strokes, a single stroke after a long idle stretch, tools switched every frame, one
  tool held for a whole episode, nothing at all for a whole episode.
- closest so far: the default brush alternates idle stretches with strokes of a few frames.
- to cover it: eval decides. Probably a brush policy from data with these patterns, and checks
  that read the frames just after a silence ends and the stretches where nothing should move.

### C-003 | messy structures and pixel mixes | high | open
- asked by: the human
- not tested: scenes unlike the tidy ones the generator draws. Clutter, scribbled walls, tiny
  enclosures, sand and water mixed cell by cell, thin interleaved layers, lattices, a nearly
  full world, an empty one.
- closest so far: the generator's platforms, cups, ramps, pools and blobs.
- to cover it: eval decides. Probably scene generators from data, with fidelity and rollout
  checks on them; a fine mixture may need a check of its own.

### C-004 | are the gains fitted to the benchmark? | medium | open
- asked by: the session that set the lab up, as a suggestion. Eval to judge.
- not tested: every run is chosen by its numbers on the same benchmark episodes, so some of a
  gain may belong to those episodes and not to the model.
- to cover it: a twin of the main benchmark with another seed, scored for the champion and for
  the runs it replaced. The gates' numbers should carry over within noise.

## Covered

(none yet)
