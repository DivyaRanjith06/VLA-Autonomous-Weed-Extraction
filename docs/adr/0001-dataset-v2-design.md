# Dataset version 2 is built around what the robot can see

Version 1 taught the pulling motion but not which weed a prompt means: the matching weed
was often out of view at the start, images were 144 x 256, and only 26 of 70 episodes
named a weed type. Version 2 adds a bed camera that follows the robot and shows about
1.5 m of bed ahead, stores 384 x 384 images at 25 pictures a second, and gives every
episode a prompt that names a weed type.

## Considered options

- **A camera over the whole bed.** Rejected: at 256 x 256 a weed would be a few pixels,
  and a real robot could not carry it. The cost of the 1.5 m view is that the robot must
  drive forward until a matching weed comes into view.
- **A random matching weed in each example.** Rejected: the same picture and prompt would
  lead to different actions. Every example removes the nearest matching weed instead, which
  is also the first one the bed camera shows. Any matching weed still counts as success
  when testing.
- **256 pixel images.** Rejected: the thin shepherd weed is about 10 pixels tall in the bed
  camera. At 384 it is about 15. 512, the size the model works at, is the next step if the
  model still confuses the weed types.

## Consequences

- With a fixed start and the nearest-match rule, each layout yields at most six different
  examples (two types, alone or with left or right), so variety comes from the number of
  layouts: 38 layouts with 4 episodes each.
- Version 1 and version 2 cannot be mixed: the cameras, image size and rate differ. So
  version 2 replaces version 1, which is removed once version 2 is verified.
