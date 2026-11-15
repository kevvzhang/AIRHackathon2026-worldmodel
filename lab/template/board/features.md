# Features: what the world learns next

Kept by the eval agent. A feature is opened only when the current world is roughly right (the
champion passes every gate in `gates.toml` and no blocker is open in `gaps.md`), and only one is
in flight at a time.

Entry format:

    ### F-001 | oil that floats on water | proposed
    - what: ...
    - why it is interesting for a world model: ...
    - the world needs (data): ...
    - roughly right means (eval): the checks and gates that will be added
    - risks: ...
    - log: dated lines; status moves proposed -> building -> training -> done

## Ideas, roughly in the order the human finds them interesting

- More materials. The simulator already has an `extended` set with oil (floats on water) and
  smoke (a gas that rises).
- Reactions: fire that burns oil or wood, water that puts fire out, acid that eats walls.
- New behaviours: stone that falls as a rigid piece, plants that grow, ice that melts.
- A larger world (96 or 128 cells), bigger brushes, longer rollouts.

The eval agent chooses, and may propose others.

## In flight

(none: the current world is not roughly right yet)

## Done

(none yet)
