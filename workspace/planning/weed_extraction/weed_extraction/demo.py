"""Live demo: type a prompt, and the robot finds a matching weed and bins it.

    python -m weed_extraction.demo "remove the shepherd weed"
    python -m weed_extraction.demo "remove all sowthistle in the left bed" --log-actions

When the robot is home again it asks for the next prompt, in the same scene.

A state machine runs the robot. No trained policy is used: the weed detector finds the
weed, and the pick is the scripted expert's (uproot.Uprooter), the same one
run_scripts/test_up_root.sh shows.

    back up  the base drives to the start of the rail
    scan     it drives along the bed while the weed detector maps what the bed camera
             shows, until a matching weed is close ahead
    go       it parks beside that weed
    pick     the expert reaches in, grips, twists, lifts, carries over the bin, drops
    retract  if the weed is not in the bin, the arm goes back to its home pose and the
             expert tries again, up to TRIES picks in all
    resume   an "all" prompt only: after a weed is binned, or given up on, the arm goes
             back to its home pose and the scan carries on along the rail
    return home  arm and base ease back to where the robot started; also after a scan
             that found no matching weed

The expert is handed the weed nearest where the detector placed it, and aims at that
weed's own stem.

Prompts are read by a fixed grammar (Command), not by a language model: one weed type,
optionally one bed, and "all"/"every" for every matching weed rather than the first.
"""

import argparse
import collections
import dataclasses
import json
import re
import time
from pathlib import Path

import numpy as np

from weed_extraction.detector import WeedDetector
from weed_extraction.ik import ArmIK
from weed_extraction.recorder import base_x, bin_interior, gripper_width, in_bin
from weed_extraction.scene import Manifest, build_env
from weed_extraction.uproot import Uprooter, between


@dataclasses.dataclass(frozen=True)
class Command:
    """What a prompt asks for, by keyword: a weed type, a bed side or both beds, and one
    weed or all of them. Keyword matching, not language understanding."""

    species: str
    side: str | None  # "left", "right", or None for both beds
    every: bool  # every matching weed, not just the first the scan reaches
    # "from your standing position": accepted, but the search is the usual back up and scan.
    standing: bool

    VERBS = ("remove", "clear", "pull", "rid", "take", "extract", "uproot")
    EVERY = ("all", "every", "each")

    @classmethod
    def parse(cls, prompt):
        text = prompt.lower()
        words = set(re.findall(r"[a-z]+", text))
        species = [s for s in Manifest.WEED_SPECIES if s in text]
        if not species:
            if "strawberr" in text:
                raise ValueError("strawberries are the crop and are never removed; the prompt "
                                 f"must name a weed type: {Manifest.WEED_SPECIES}")
            raise ValueError(f"the prompt must name a weed type: {Manifest.WEED_SPECIES}")
        if len(species) > 1:
            raise ValueError(f"name one weed type per prompt, not {' and '.join(species)}")
        if not words & set(cls.VERBS):
            raise ValueError("the prompt must say what to do: remove, clear, pull out, take "
                             "out or get rid of")
        sides = [s for s in ("left", "right") if s in words]
        if len(sides) > 1:
            raise ValueError("name one bed, left or right, or neither for both")
        field = "field" in words
        if field and sides:
            raise ValueError("name either one bed or the whole field")
        return cls(species[0], sides[0] if sides else None,
                   field or bool(words & set(cls.EVERY)), "standing" in words)


def outcome(requested, binned):
    """An "all" prompt's tally: (removed, remaining, success). Only a weed in the bin
    counts; one that was detected, or picked and dropped elsewhere, does not. Success
    needs every requested weed."""
    removed = sorted(set(requested) & set(binned))
    remaining = sorted(set(requested) - set(binned))
    return removed, remaining, bool(requested) and not remaining


class ActionLog:
    """Every action the demo applies to the robot, to a JSONL file, and a few printed.

    The demo runs no trained policy, so the source is always the scripted expert: the
    state machine drives the base and the Uprooter the arm and gripper.
    """

    SOURCE = "scripted_expert"
    NAMES = (*(f"q{i}" for i in range(1, 8)), "gripper", "base_x")

    def __init__(self, directory, first, last, every):
        directory.mkdir(parents=True, exist_ok=True)
        self.path, self.results = directory / "actions.jsonl", directory / "results.jsonl"
        self.file = self.path.open("w")
        self.first, self.every = first, every
        self.last = collections.deque(maxlen=last)
        self.step = 0  # over the whole session
        self.start()

    def start(self):
        """Begin a prompt's summary."""
        self.count, self.low, self.high = 0, None, None
        self.last.clear()

    def record(self, demo, state, phase):
        # As applied: the environment clips what it is given to the actuator ranges.
        action = demo.env.data.ctrl.copy()
        record = {"step": self.step, "simulation_time": round(float(demo.env.data.time), 4),
                  "source": self.SOURCE, "dim": len(action),
                  "action": [round(float(v), 6) for v in action], "prompt": demo.prompt,
                  "target": demo.weed, "state": state, "phase": phase}
        self.file.write(json.dumps(record) + "\n")
        # For the terminal only: what the robot measures once the action is applied.
        model, data = demo.env.model, demo.env.data
        shown = (record, base_x(model, data), gripper_width(model, data))
        if self.count < self.first or (self.every and self.step % self.every == 0):
            print(self.line(*shown))
        self.last.append(shown)
        self.low = action if self.low is None else np.minimum(self.low, action)
        self.high = action if self.high is None else np.maximum(self.high, action)
        self.count += 1
        self.step += 1

    @staticmethod
    def line(record, base, width):
        """One applied action as terminal telemetry: simulated time, step, the measured
        state after it, the 9-D command [q1..q7 rad, gripper 0-255, base_x m] and the phase."""
        a = record["action"]
        mode = record["state"].replace(" ", "_")
        return (f"[ACTION] t={record['simulation_time']:.3f}s step={record['step']:06d}\n"
                f"  state=[mode={mode}, target={record['target'] or 'none'}, "
                f"base_x={base:.3f} m, gripper_width={width * 1000:.1f} mm]\n"
                f"  action=[{', '.join(f'{v:.3f}' for v in a[:7])}, {a[7]:.1f}, {a[8]:.3f}]\n"
                f"  phase={(record['phase'] or mode).replace(' ', '_')}")

    def summary(self):
        """Print the prompt's last actions and per-dimension range, then start afresh."""
        for shown in self.last:
            print(self.line(*shown))
        print(f"[ACTION SUMMARY] {self.count} actions this prompt ({self.step} this session), "
              f"dim={len(self.NAMES)}")
        if self.count:
            for name, low, high in zip(self.NAMES, self.low, self.high):
                print(f"  {name:8s} min {low:9.3f}  max {high:9.3f}")
        self.file.flush()
        self.start()

    def result(self, record):
        with self.results.open("a") as file:
            file.write(json.dumps(record) + "\n")

    def close(self):
        self.file.close()


class Demo:
    SCAN_SPEED = 0.3  # metres a second along the bed
    # A weed is placed from the direction it is seen in, which is surest when it is
    # nearly below the camera. The scan stops for a matching weed this close ahead.
    NEAR = 0.6
    DETECT_EVERY = 3  # control ticks between looks; the base moves 36 mm in that time
    # Roots hold this fraction of their recorded strength in the demo: 7.5 to 10 N of
    # straight pull instead of 15 to 20. Recording keeps the full strength.
    ROOT_EASE = 0.5
    TRIES = 3  # picks at one weed before giving up
    RETURN_SECONDS = 4.0  # arm and base ease back to the home pose over this long
    # A sighting this close to where a pick already happened is that weed again.
    SAME_PLACE = 0.1

    def __init__(self, detector_file, manifest, scene_dir):
        self.manifest = manifest
        self.detector = WeedDetector(detector_file)
        size = self.detector.SIZE
        self.env = build_env(manifest, scene_dir, cameras=(self.detector.CAMERA,), width=size,
                             height=size, control_hz=25, max_steps=10**6)
        self.obs, _ = self.env.reset()
        for eq in self.env.roots.strength:  # after reset, which sets them
            self.env.roots.strength[eq] *= self.ROOT_EASE
        model = self.env.model
        self.dt = self.env.n_substeps * model.opt.timestep
        self.base = model.actuator("base_x").id
        slide = model.joint("base_slide_x")
        self.base_qpos, self.base_qvel = slide.qposadr[0], slide.dofadr[0]
        self.ik = ArmIK(self.env.scene)
        self.home = model.key_ctrl[model.key("home").id].copy()
        self.state, self.ticks = "done", 0
        self.log = None  # an ActionLog, if actions are logged
        self.prompt = self.weed = None

    def command(self, prompt):
        """Start on a new prompt from wherever the robot is; the weeds already binned
        stay binned."""
        order = Command.parse(prompt)
        self.prompt, self.species, self.side = prompt, order.species, order.side
        self.every, self.standing = order.every, order.standing
        self.state, self.began, self.tries = "back up", self.env.data.time, 1
        self.binned_before = set(self.binned())  # a weed binned earlier does not count
        # What an "all" prompt is judged against: the scene's own list, never used to steer.
        self.requested = [w for w in self.matching() if w not in self.binned_before]
        self.handled, self.failed = [], []  # places picked at; weeds given up on
        self.landed = []  # weeds in the bin when their own pick ended
        self.map = []  # every plant seen: [type, x, y]
        self.target = self.weed = self.expert = None
        self.return_from = self.return_began = None

    def step(self):
        """One control tick of the current state. Returns the weeds uprooted in it."""
        env, data = self.env, self.env.data
        action = data.ctrl.copy()
        x = data.qpos[self.base_qpos]
        low, high = env.model.actuator_ctrlrange[self.base]
        state, phase = self.state, None
        if self.state == "back up":
            action[self.base] = low
            if abs(x - low) < 0.01:
                self.state = "scan"
        elif self.state == "scan":
            action[self.base] = min(action[self.base] + self.SCAN_SPEED * self.dt, high)
            ahead = [m for m in self.look() if m[0] - x <= self.NEAR and not self.handled_at(m)]
            if ahead:
                self.target, self.state = min(ahead), "go"
            elif x > high - 0.01:
                self.start_return()
        elif self.state == "go":
            # Still in view on the way in, and nearer: the later sightings place it better.
            again = [m for m in self.look() if abs(m[0] - self.target[0]) < 0.1]
            self.target = min(again, default=self.target)
            action[self.base] = self.target[0]
            if abs(x - self.target[0]) < 0.005 and abs(data.qvel[self.base_qvel]) < 0.01:
                self.weed = self.nearest_weed()
                if self.every and self.weed in self.failed:
                    self.handled.append(self.target)  # given up on already: scan on
                    self.state = "scan"
                else:
                    self.start_pick()
        elif self.state == "pick":
            # The expert writes the actuator targets itself.
            phase = self.expert.step()
            if phase is not None:
                action = data.ctrl.copy()
            elif self.every:
                self.next_target()
            elif set(self.binned()) - self.binned_before or self.tries >= self.TRIES:
                self.start_return()
            else:
                self.tries += 1
                self.retract("retract")
        elif self.state in ("retract", "resume"):
            # Back to the home pose over the parked base, then try again or scan on.
            t = (data.time - self.return_began) / self.RETURN_SECONDS
            action = between(self.return_from, self.home, t)
            action[7], action[self.base] = Uprooter.OPEN, self.target[0]
            if t >= 1:
                if self.state == "retract":
                    self.start_pick()
                else:
                    self.state = "scan"
        elif self.state == "return home":
            t = (data.time - self.return_began) / self.RETURN_SECONDS
            action = between(self.return_from, self.home, t)
            if t >= 1 and abs(x - self.home[self.base]) < 0.005:
                self.state = "done"
        self.ticks += 1
        self.obs, _, _, _, info = env.step(action)
        if self.log is not None:
            self.log.record(self, state, phase)
        return info["uprooted"]

    def next_target(self):
        """An "all" prompt, after a pick: try the same weed again, or count it and scan on.
        A weed counts as removed only once it is in the bin."""
        landed = self.weed in self.binned()
        if not landed and self.tries < self.TRIES:
            self.tries += 1
            self.retract("retract")
            return
        (self.landed if landed else self.failed).append(self.weed)
        self.handled.append(self.target)
        self.tries = 1
        self.retract("resume")

    def handled_at(self, seen):
        return any(abs(seen[0] - x) < self.SAME_PLACE and abs(seen[1] - y) < self.SAME_PLACE
                   for x, y in self.handled)

    def nearest_weed(self):
        """The weed standing nearest where the detector placed the target. Weeds already
        in the bin are not standing anywhere."""
        model, data = self.env.model, self.env.data
        binned = set(self.binned())
        return min((p.name for p in self.manifest.weeds if p.name not in binned),
                   key=lambda name: np.linalg.norm(
                       data.xpos[model.body(name).id][:2] - self.target))

    def start_pick(self):
        self.expert = Uprooter(self.env, self.weed, ik=self.ik)
        self.state = "pick"

    def retract(self, state):
        self.return_from, self.return_began = self.env.data.ctrl.copy(), self.env.data.time
        self.state = state

    def start_return(self):
        self.return_from, self.return_began = self.env.data.ctrl.copy(), self.env.data.time
        self.state = "return home"

    def look(self):
        """Run the detector on the bed camera, add what it shows to the map, and return
        the matching weeds seen whole, as (x, y)."""
        if self.ticks % self.DETECT_EVERY:
            return []
        matching = []
        for label, _, box in self.detector.detect(self.obs[self.detector.CAMERA]):
            if box.min() <= 1 or box.max() >= self.detector.SIZE - 1:
                continue  # cut by the picture's edge: its middle is not the plant's
            x, y = self.detector.locate(self.env.model, self.env.data, label, box)
            seen = next((m for m in self.map if m[0] == label
                         and abs(m[1] - x) < 0.15 and abs(m[2] - y) < 0.15), None)
            if seen is None:
                self.map.append([label, x, y])
            else:
                seen[1:] = x, y
            if label == self.species and self.side in (None, "left" if y > 0 else "right"):
                matching.append((x, y))
        return matching

    def binned(self):
        """The weeds lying in the bin right now."""
        model, data = self.env.model, self.env.data
        lo, hi = bin_interior(model, data)
        return [p.name for p in self.manifest.weeds
                if in_bin(data.xpos[model.body(p.name).id], lo, hi)]

    def matching(self):
        """The weeds the prompt asks for: its weed type, in its bed if it names one."""
        return [p.name for p in self.manifest.weeds if p.species == self.species
                and self.side in (None, "left" if p.y > 0 else "right")]


def run(demo, seconds, viewer):
    """Carry out the current prompt and print what it put in the bin."""
    seconds = seconds or (600.0 if demo.every else 90.0)
    state, before, uprooted = None, demo.binned_before, set()
    if demo.standing:
        print("  from the standing position: the usual search, back along the rail and "
              "scan forward")
    while demo.state != "done" and demo.env.data.time - demo.began < seconds:
        if viewer is not None and not viewer.is_running():
            break
        began = time.time()
        if demo.state != state:
            state = demo.state
            print(f"  [{demo.env.data.time:5.1f}s] {state}", flush=True)
        for weed in demo.step():
            uprooted.add(weed)
            print(f"  [{demo.env.data.time:5.1f}s] uprooted {weed}")
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0.0, demo.dt - (time.time() - began)))
    print(f"  [{demo.env.data.time:5.1f}s] {demo.state}")
    print("map: " + ", ".join(f"{label} at {x:.2f} m {'left' if y > 0 else 'right'}"
                              for label, x, y in sorted(demo.map, key=lambda m: m[1])))
    binned, matching = set(demo.binned()) - before, set(demo.matching()) - before
    record = {"prompt": demo.prompt, "family": demo.manifest.family,
              "variant": demo.manifest.variant, "species": demo.species, "side": demo.side,
              "all": demo.every, "simulated_seconds": round(demo.env.data.time - demo.began, 2),
              "finished": demo.state == "done", "binned": sorted(binned),
              "uprooted": sorted(uprooted)}
    if demo.every:
        removed, remaining, success = outcome(demo.requested, binned)
        where = f"the {demo.side} bed" if demo.side else "both beds"
        print(f"requested targets ({len(demo.requested)}, every {demo.species} weed in {where}, "
              f"from the scene's list): {', '.join(demo.requested) or 'none left'}")
        print(f"successfully removed ({len(removed)} of {len(demo.requested)}): "
              f"{', '.join(removed) or 'none'}")
        print(f"failed targets: {', '.join(demo.failed) or 'none'}")
        print(f"remaining matching targets: {', '.join(remaining) or 'none'}")
        if wrong := sorted(binned - set(demo.requested)):
            print(f"removed but not requested: {', '.join(wrong)}")
        if lost := sorted(set(demo.landed) - binned):
            print(f"in the bin after its pick, but not at the end: {', '.join(lost)}")
        if stray := sorted(uprooted - set(demo.requested)):
            print(f"uprooted but not requested: {', '.join(stray)}")
        print(f"simulated time: {record['simulated_seconds']:.1f} s")
        result = (f"SUCCESS, all {len(removed)} requested weeds are in the bin" if success
                  else f"FAILURE, {len(removed)} of {len(demo.requested)} requested weeds "
                       "are in the bin")
        record.update(requested=demo.requested, removed=removed, failed=demo.failed,
                      remaining=remaining, success=success)
    else:
        print(f"matching weeds: {', '.join(sorted(matching)) or 'none left in this scene'}")
        print(f"put in the bin: {', '.join(sorted(binned)) or 'nothing'}")
        result = ("a matching weed is in the bin" if binned & matching
                  else "wrong weed in the bin" if binned
                  else "no matching weed seen" if demo.target is None
                  else "no weed in the bin")
    print("RESULT: " + result)
    if demo.log is not None:
        demo.log.summary()
        demo.log.result({**record, "result": result})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("prompt", nargs="?",
                        help='e.g. "remove the shepherd weed" or "remove all sowthistle in the '
                             'left bed"; asked for if left out. After it, the demo asks for '
                             "the next")
    parser.add_argument("--detector", type=Path, default=Path("models/detector.pt"))
    parser.add_argument("--family", default="midday_dry", choices=list(Manifest.FAMILIES))
    parser.add_argument("--variant", type=int, default=0,
                        help=f"scene of the family; {Manifest.HELD_OUT_VARIANT} was not "
                             "trained on")
    parser.add_argument("--seconds", type=float,
                        help="simulated time allowed per prompt; 90 by default, 600 for an "
                             '"all" prompt')
    parser.add_argument("--headless", action="store_true", help="no viewer, result only")
    parser.add_argument("--log-actions", action="store_true",
                        help="save every applied 9-D action to "
                             "outputs/demo_logs/<time>/actions.jsonl and print a few")
    parser.add_argument("--log-first", type=int, default=5,
                        help="actions printed at the start of each prompt")
    parser.add_argument("--log-last", type=int, default=5,
                        help="actions printed at the end of each prompt")
    parser.add_argument("--log-every", type=int, default=0,
                        help="also print every Kth action; 0 for none")
    args = parser.parse_args()
    if args.prompt is None:
        try:
            args.prompt = input("Enter your instruction: ").strip()
        except EOFError:
            args.prompt = ""
        if not args.prompt:
            raise SystemExit("no instruction given")

    manifest = Manifest.variants(args.family, args.variant + 1)[args.variant]
    scene_dir = Path("outputs/demo") / f"{args.family}_{args.variant}"
    demo = Demo(args.detector, manifest, scene_dir.resolve())
    try:
        demo.command(args.prompt)
    except ValueError as error:
        raise SystemExit(error)
    print(f"{args.family} {args.variant} ({manifest.split}); weeds: "
          f"{', '.join(p.name for p in manifest.weeds)}")
    if args.log_actions:
        demo.log = ActionLog(Path("outputs/demo_logs") / time.strftime("%Y%m%d_%H%M%S"),
                             args.log_first, args.log_last, args.log_every)
        print("logging every applied 9-D action")
    if args.headless:
        prompts(demo, args.seconds, None)
    else:
        import mujoco.viewer  # fails without a display

        with mujoco.viewer.launch_passive(demo.env.model, demo.env.data) as viewer:
            prompts(demo, args.seconds, viewer)
    if demo.log is not None:
        demo.log.close()
    demo.env.close()


def prompts(demo, seconds, viewer):
    """Run the given prompt, then ask for the next, until an empty answer."""
    while True:
        run(demo, seconds, viewer)
        while True:
            try:
                prompt = input("next prompt (Enter to quit): ").strip()
            except EOFError:
                return
            if not prompt or (viewer is not None and not viewer.is_running()):
                return
            try:
                demo.command(prompt)
                break
            except ValueError as error:
                print(error)


if __name__ == "__main__":
    main()
