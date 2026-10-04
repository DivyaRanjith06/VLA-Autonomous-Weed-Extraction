# Plan: pick the weed the prompt names

Written 2026-10-02. Terms follow `GLOSSARY.md`.

## Goal

The user types a prompt naming a weed type ("remove the shepherd weed", "remove one
sowthistle weed from the left bed"). The robot drives to a matching weed, pulls it and
drops it in the bin, without touching the crop.

Pass mark: 80% success on seen scenes, 60% on held-out scenes, 20 runs each.

## Verdict on the current dataset

**Not good enough for this goal.** It is a sound dataset for learning the pulling motion,
and it is clean (every episode verified, both weed types present, 1 or 2 attempts each).
But a model trained on it has no way to learn which weed the prompt means. Six problems,
worst first.

1. **The robot cannot see the weed it is asked for.** I rendered the start of a training
   scene (midday_dry 0, nine weeds). The overview camera follows the robot and only shows
   the stretch of bed next to it: about five of the nine weeds. The wrist camera shows
   bare soil. The target of that episode is 3.7 m away, out of both views. The model would
   have to memorise where to drive, which cannot work on a held-out scene.
2. **The images are too small to tell the types apart.** They are stored at 144 x 256. A
   weed in the overview is about 10 pixels tall. SmolVLA enlarges every image to 512 x 512,
   so we are throwing away detail it could use.
3. **Too few type prompts.** 26 of 70 training episodes name a weed type. The other 44 say
   "remove weed 3", "nearest", or left/right.
4. **No contrast pairs.** Not one scene asks for a sowthistle in one episode and a shepherd
   in another with the same wording. That pair is what forces a model to read the type
   word. Without it the model can learn "this scene, go there".
5. **One wording per prompt.** 24 distinct sentences in total, each from a fixed template.
6. **Every layout is seen 3 or 4 times, each time for a different weed.** So each
   weed-and-prompt combination appears once. LeRobot's own guidance is about 10 episodes
   per variation, and reports that 25 episodes in total was not enough.

What is fine and should stay: 28-value state, 9-value action, two weed types in 29 of 30
scenes, 4 to 9 weeds per scene, the two held-out groups, the validation checks.

## Best practices found

**SmolVLA fine-tuning** (LeRobot docs, SmolVLA paper)

- Start from `lerobot/smolvla_base`. About 50 episodes is the floor for one simple task;
  repeat each variation about 10 times.
- 20,000 steps at batch 64 is the standard recipe, about 4 hours on an A100. Lower the
  batch if the card is small.
- By default the vision part and the language model are frozen and only the action part
  trains. Unfreezing costs much more memory and overfits small datasets.
- It takes colour images, enlarged to 512 x 512. The released model was trained on 30 fps
  data and predicts 50 actions at a time. Our data is 50 Hz, so one prediction covers 1
  second instead of 1.7.

**Making a model follow the prompt** (LIBERO-CF, LangGap, CAST, "Why your VLA ignores language")

- Models ignore the prompt when the picture alone predicts the action. The fix is data
  where the same picture leads to different actions depending on the prompt.
- Rewording existing episodes in several ways is the cheapest gain reported (one paper:
  0% to 90% on the same trajectories).
- Measure two things separately: did it go to the right object, and did it finish the task.
- Test with prompts that swap the object in a scene the model has seen.
- A test-time trick (counterfactual action guidance) can add about 15 points of
  prompt-following without retraining. Keep it in reserve.
- One study found fine-tuning SmolVLA on about 60 demonstrations made it worse. Small
  datasets are a real risk here.

## Skills online

Nothing ready-made fits. I checked:

- `huggingface/skills`: has `hf-cli` and `huggingface-datasets`. Useful only if we upload
  the dataset or model to the Hugging Face Hub. Nothing for LeRobot or robot policies.
- `nvidia/skills`: Isaac and healthcare simulation. Not relevant to MuJoCo or SmolVLA.
- `arpitg1304/robotics-agent-skills`: ROS code quality. Not relevant to training.

Recommendation: install none. The LeRobot docs cover what a skill would.

## The plan

Fix what the robot can see and what the data asks for, then train. Training on the
current 70 first would mostly tell us what the list above already does.

1. **Give the robot a view of the bed.** Add a fixed camera that sees every weed the robot
   can reach, and keep the wrist camera for the close-up. Check: at the start of every
   scene, the matching weed is in view.
2. **Store bigger images.** At least 256 x 256 for the wrist, and enough for the bed view
   that a weed is 40 or more pixels tall. Check: a person can name the weed type from a
   stored frame.
3. **Rebuild the prompts around weed type.** Type-only and type-plus-place prompts for
   every episode, three or four wordings each. Drop numbered prompts. Check: every training
   prompt names a type.
4. **Add contrast pairs.** In each scene, record one episode per weed type with otherwise
   identical wording. Check: every two-type scene has at least one pair.
5. **Generate more.** About 300 episodes: 30 layouts, 10 episodes each, balanced across
   the two types. Record at 25 Hz. Check: the validation step accepts them and the counts
   balance.
6. **Build the tester.** A saved model drives the robot in the simulator on 20 seen and 20
   held-out scenes. It reports: right weed, wrong type, wrong place, nothing picked,
   dropped outside the bin, crop touched. It also runs the swapped-type test. Check: the
   scripted expert scores near 100%.
7. **Smoke test on the M4.** 10 to 20 steps, colour images only. Check: a model is saved
   with no errors.
8. **Train on the NVIDIA GPU.** 20,000 steps, saved every 5,000, local logs. Two runs:
   the default (vision frozen) and vision unfrozen. Check: both finish.
9. **Test and choose.** Run the tester on every saved model. Keep the best. Compare
   against the pass mark.
10. **Next round from the failure split.** Mostly "wrong type": more contrast pairs or
    larger images. Mostly "nothing picked": more episodes. Then try adding depth.

## Decisions for dataset version 2

Agreed on 2026-10-02; the reasons are in `docs/adr/0001-dataset-v2-design.md`.

- One automated script builds it, as `weed_extraction.dataset` does today.
- Cameras: a new bed camera, the wrist colour and depth camera, and the current overview
  camera. All at 384 x 384.
- The bed camera sits high above the bed, follows the robot, tilts ahead and shows about
  1.5 m of bed.
- About 150 episodes over 38 layouts, 4 each, all for training. The tester builds its own held-out scenes.
- Every prompt names a weed type, alone or with "left bed" or "right bed". Four fixed
  wordings each.
- When several weeds match, the recorded example removes the nearest matching weed.
- The robot always starts at the end of the walkway.
- Recorded at 25 pictures a second.
- Built on the NVIDIA Linux machine. It replaces version 1, which is removed from the
  repository once version 2 is verified.

## Decisions for the baseline training run

Agreed on 2026-10-02.

- Trained on the RTX 4060 Laptop (8 GB), on dataset version 2 (140 episodes, 25 pictures
  a second).
- The model sees the bed camera and the wrist camera, colour only. The overview camera
  and the depth image stay in the dataset but are not given to the model.
- Vision frozen. The memory test showed that training the vision and language parts
  together does not fit in 8 GB at batch 4 or 8. Training only part of them is untested.
- 20,000 steps at batch 16, saved every 5,000, started with `run_scripts/train.sh`. About
  7 hours.
- A live demo comes before the scored tester: type a prompt, watch the model drive the
  robot in the viewer, and read whether a matching weed ended in the bin. It opens a
  seen scene by default.
- If the baseline picks the wrong weed type, the fix is chosen from how it fails.

## Result of the baseline, and the split design

Written 2026-10-03. The reasons are in `docs/adr/0002-state-machine-around-the-pick-model.md`.

The baseline (20,000 steps at batch 5) did not reach a weed. Fed recorded frames from the
middle of an episode it predicts the right arm motion, but at the first frame it must
decide where to drive, and its guess was off by more than 0.3 m in 11 of 14 training
episodes. The cause is the data: in 81 of 140 examples the weed starts more than 1.5 m
away, outside the bed camera's view.

So the task is split:

1. **Back up**: the base drives to the start of the rail.
2. **Scan**: it drives along the bed at 0.3 m a second. The weed detector builds the weed
   map from the bed camera.
3. **Go**: at the first matching weed within 0.6 m ahead, it parks beside it.
4. **Pick**: the pick model moves the arm. It is told the weed type and the bed.

Decided with the user: a trained detector (not the simulator's knowledge), covering the
crop as well; stop at the first matching weed; the demo counts a grasp within 1 cm of the
stem.

Measured so far:

- Weed detector, on the 14 held-out scenes: 382 of 383 sowthistles and 322 of 347
  shepherd weeds found, none given the wrong type, placed within 5 to 6 mm along the bed
  (median).
- Scan and park, with the baseline as the pick model, 8 prompts over 4 scenes: the base
  parked at a weed every time; 2 weeds uprooted, 1 binned. The baseline was trained on
  driving and reaching mixed together, and often skips ahead in the motion.

Next: a pick-only dataset, `python -m weed_extraction.dataset outputs/dataset_pick
--parked`. 137 episodes in the 38 training layouts; the base is parked beside the weed
before recording starts, and every prompt names the weed type and the bed. Then
`run_scripts/train.sh` trains the pick model on it, and `run_scripts/demo.sh` runs it.

## Sources

- https://huggingface.co/docs/lerobot/smolvla
- https://arxiv.org/html/2506.01844v1
- https://arxiv.org/html/2602.17659v1
- https://labelstud.io/blog/why-your-vla-model-ignores-language-and-the-cheapest-fix-in-robotics/
- https://medium.com/correll-lab/when-fine-tuning-hurts-failure-modes-of-visuomotor-imitation-learning-on-a-low-cost-robot-5f5013df7c95
- https://github.com/huggingface/skills
- https://github.com/nvidia/skills
- https://github.com/arpitg1304/robotics-agent-skills
