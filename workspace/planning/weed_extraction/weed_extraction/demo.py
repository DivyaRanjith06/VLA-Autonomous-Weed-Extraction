"""Live demo: type a prompt, and the robot finds a matching weed and bins it.

    python -m weed_extraction.demo "remove the shepherd weed"

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
    return home  arm and base ease back to where the robot started; also after a scan
             that found no matching weed

The expert is handed the weed nearest where the detector placed it, and aims at that
weed's own stem.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from weed_extraction.detector import WeedDetector
from weed_extraction.ik import ArmIK
from weed_extraction.recorder import bin_interior, in_bin
from weed_extraction.scene import Manifest, build_env
from weed_extraction.uproot import Uprooter, between


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

    def command(self, prompt):
        """Start on a new prompt from wherever the robot is; the weeds already binned
        stay binned."""
        species = next((s for s in self.manifest.WEED_SPECIES if s in prompt), None)
        if species is None:
            raise ValueError(f"the prompt must name a weed type: {self.manifest.WEED_SPECIES}")
        self.species = species
        self.side = next((s for s in ("left", "right") if f"{s} bed" in prompt), None)
        self.state, self.began, self.tries = "back up", self.env.data.time, 1
        self.binned_before = set(self.binned())  # a weed binned earlier does not count
        self.map = []  # every plant seen: [type, x, y]
        self.target = self.weed = self.expert = None
        self.return_from = self.return_began = None

    def step(self):
        """One control tick of the current state. Returns the weeds uprooted in it."""
        env, data = self.env, self.env.data
        action = data.ctrl.copy()
        x = data.qpos[self.base_qpos]
        low, high = env.model.actuator_ctrlrange[self.base]
        if self.state == "back up":
            action[self.base] = low
            if abs(x - low) < 0.01:
                self.state = "scan"
        elif self.state == "scan":
            action[self.base] = min(action[self.base] + self.SCAN_SPEED * self.dt, high)
            ahead = [m for m in self.look() if m[0] - x <= self.NEAR]
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
                self.start_pick()
        elif self.state == "pick":
            # The expert writes the actuator targets itself.
            if self.expert.step() is not None:
                action = data.ctrl.copy()
            elif set(self.binned()) - self.binned_before or self.tries >= self.TRIES:
                self.start_return()
            else:
                self.tries += 1
                self.return_from, self.return_began = data.ctrl.copy(), data.time
                self.state = "retract"
        elif self.state == "retract":
            # Back to the home pose over the parked base, then try again.
            t = (data.time - self.return_began) / self.RETURN_SECONDS
            action = between(self.return_from, self.home, t)
            action[7], action[self.base] = Uprooter.OPEN, self.target[0]
            if t >= 1:
                self.start_pick()
        elif self.state == "return home":
            t = (data.time - self.return_began) / self.RETURN_SECONDS
            action = between(self.return_from, self.home, t)
            if t >= 1 and abs(x - self.home[self.base]) < 0.005:
                self.state = "done"
        self.ticks += 1
        self.obs, _, _, _, info = env.step(action)
        return info["uprooted"]

    def nearest_weed(self):
        """The weed standing nearest where the detector placed the target."""
        model, data = self.env.model, self.env.data
        return min((p.name for p in self.manifest.weeds),
                   key=lambda name: np.linalg.norm(
                       data.xpos[model.body(name).id][:2] - self.target))

    def start_pick(self):
        self.expert = Uprooter(self.env, self.weed, ik=self.ik)
        self.state = "pick"

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
    state, before = None, demo.binned_before
    while demo.state != "done" and demo.env.data.time - demo.began < seconds:
        if viewer is not None and not viewer.is_running():
            break
        began = time.time()
        if demo.state != state:
            state = demo.state
            print(f"  [{demo.env.data.time:5.1f}s] {state}", flush=True)
        for weed in demo.step():
            print(f"  [{demo.env.data.time:5.1f}s] uprooted {weed}")
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0.0, demo.dt - (time.time() - began)))
    print(f"  [{demo.env.data.time:5.1f}s] {demo.state}")
    print("map: " + ", ".join(f"{label} at {x:.2f} m {'left' if y > 0 else 'right'}"
                              for label, x, y in sorted(demo.map, key=lambda m: m[1])))
    binned, matching = set(demo.binned()) - before, set(demo.matching()) - before
    print(f"matching weeds: {', '.join(sorted(matching)) or 'none left in this scene'}")
    print(f"put in the bin: {', '.join(sorted(binned)) or 'nothing'}")
    print("RESULT: " + ("a matching weed is in the bin" if binned & matching
                        else "wrong weed in the bin" if binned
                        else "no matching weed seen" if demo.target is None
                        else "no weed in the bin"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("prompt", nargs="?",
                        help='e.g. "remove the shepherd weed"; asked for if left out. After '
                             "it, the demo asks for the next")
    parser.add_argument("--detector", type=Path, default=Path("models/detector.pt"))
    parser.add_argument("--family", default="midday_dry", choices=list(Manifest.FAMILIES))
    parser.add_argument("--variant", type=int, default=0,
                        help=f"scene of the family; {Manifest.HELD_OUT_VARIANT} was not "
                             "trained on")
    parser.add_argument("--seconds", type=float, default=90.0,
                        help="simulated time allowed per prompt")
    parser.add_argument("--headless", action="store_true", help="no viewer, result only")
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
    if args.headless:
        prompts(demo, args.seconds, None)
    else:
        import mujoco.viewer  # fails without a display

        with mujoco.viewer.launch_passive(demo.env.model, demo.env.data) as viewer:
            prompts(demo, args.seconds, viewer)
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
