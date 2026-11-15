# Gaps: what the model gets wrong

Kept by the eval agent; the data and train agents take their work from here, top first.
Format and rules are in `roles\eval.md`. Headings must keep the form
`### G-000 | title | severity | status`.

The entries below were seeded by the human's setup session from the scorecard and pictures of
the reference run `recon-k4` on `{{BENCH}}` (one run, one seed). Eval: verify each one, correct
it, and take the list over.

## Open

### G-001 | interior walls are lost by the latent | blocker | open
- seen in: recon-k4; fidelity/walls = 0.670, walls/kept@16 = 0.666, walls/kept@48 = 0.654;
  `eval\bench\worst-1.png` window 39 (a wall box, a cup and a ramp come back as different shapes
  already in the decode of the last frame the model saw)
- what: about a third of the wall cells inside the border are not wall after a frame is encoded
  and decoded, before any prediction. The number barely falls further during imagination, so the
  walls are lost by the representation and not moved by the predictor.
- check: fidelity/walls (gate 0.90), walls/kept@48 (gate 0.90)
- probably: model. One 192-number vector per frame has to hold the whole layout; README "Known
  limits" reports that a 512-number vector did not help.
- log: seeded at setup.

### G-002 | walls appear where there are none | major | open
- seen in: recon-k4; walls/invented@16 = 0.202, walls/invented@48 = 0.228
- what: imagined wall cells without a wall in the truth amount to a fifth of the true wall
  cells. Probably the same cause as G-001 (walls decoded in the wrong place), to be confirmed.
- check: walls/invented@48 (gate 0.10)
- probably: model
- log: seeded at setup.

### G-003 | fine structure does not survive encoding | major | open
- seen in: recon-k4; fidelity/miou = 0.788 (wall 0.850, sand 0.783, water 0.732)
- what: thin walls, single grains and falling streams are blurred or missing in the decode of a
  true frame. README "Known limits" has the same finding from before the lab.
- check: fidelity/miou (gate 0.85)
- probably: model
- log: seeded at setup.

### G-004 | imagination barely beats repeating the last frame | major | open
- seen in: recon-k4; rollout/beats_copy@16 = 0.002, rollout/beats_copy@48 = 0.066;
  dynamics/changed@16 = 0.593
- what: 16 frames ahead the imagined frame is no closer to the truth than the last true frame
  is. Part of this is G-003 (the copy is not limited by the latent, the imagination is).
- check: rollout/beats_copy@16 (gate 0.03), dynamics/changed@16 (gate 0.70)
- probably: both the representation and the predictor; separate them
- log: seeded at setup.

### G-005 | material is slowly lost | minor | open
- seen in: recon-k4; mass/sand@48 = 0.132, mass/water@48 = 0.134; bias/sand@48 = -0.071,
  bias/water@48 = -0.037
- what: the imagined amount of sand and water is off by about 13% per window on average, with a
  small net loss. No sign of material multiplying.
- check: mass/sand@48, mass/water@48 (gates 0.10)
- probably: model
- log: seeded at setup.

### G-006 | imagined frames blur where the physics is random | minor | open
- seen in: README "Known limits" (before the lab); not measured by any check yet
- what: the predictor is deterministic, so it predicts an average of water surfaces and sinking
  sand, and detail fades as the rollout goes on.
- check: none yet. Eval: find a measurement, or close this.
- probably: model
- log: seeded at setup.

## Fixed

(none yet)
