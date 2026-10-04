# A state machine finds the weed; the trained model only picks it

The baseline model had to drive to the matching weed and pick it. It failed at the
driving: in 81 of 140 recorded examples the weed is more than 1.5 m away at the start,
outside the bed camera's view, so the examples showed a drive to a place the pictures
gave no clue to. We split the task. A state machine backs the base to the start of the
rail, scans along the bed while a weed detector builds a weed map from the bed camera,
parks beside the first matching weed, and only then hands the arm to the pick model.

## Considered options

- **New examples in which the robot searches, one model for everything.** Rejected for
  now: about 10 hours to record and retrain on the laptop, and the model would still have
  to tell the weed types apart with its vision part frozen.
- **Reading weed positions from the simulator.** Rejected: it proves nothing about seeing.
  The detector uses only the bed camera picture and the camera's own position.

## Consequences

- The pick model no longer decides which weed the prompt means. The state machine reads
  the weed type and bed from the prompt, and tells the pick model the type and the bed.
- Telling the weed types apart is the detector's job, and can be measured on its own.
- The scan stops at the first matching weed, so the weed map is only complete up to
  there.
- The demo counts a grasp when the jaws shut within 1 cm of the stem. The recorded
  examples needed the fingertips within 1 mm of a 3 mm stem that cannot bend, which the
  pick model does not reach.
