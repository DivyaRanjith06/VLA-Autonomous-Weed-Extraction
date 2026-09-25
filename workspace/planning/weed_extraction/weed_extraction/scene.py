"""Scene manifests, and the MJCF generated from them.

A manifest is plain data: the beds, where each plant stands and how it is turned,
leaned and sized, the light, the soil and the overview camera. `write_scene` templates
the hand-written scene with it, so a generated scene is the nominal one with only those
values changed: same robot, wrist camera, physics, collision settings and welds.
"""

import copy
import dataclasses
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np

from weed_extraction.env import ROOT_BREAK_FORCE, ROOT_BREAK_TORQUE, SCENE, WeedExtractionEnv
from weed_extraction.prompts import Prompt


@dataclasses.dataclass(frozen=True)
class Plant:
    name: str
    species: str
    x: float
    y: float
    z: float  # where the stem meets the soil
    yaw: float = 0.0  # radians about the vertical
    lean: float = 0.0  # radians off vertical
    lean_toward: float = 0.0  # compass direction the top leans to, radians
    size: float = 1.0  # strawberries: maturity, scaling the whole plant
    spread: float = 1.0  # strawberries: leaf spread, scaling it across on top of size
    # Weeds: root strength against the nominal 15 N and 0.07 N.m. It stands in for root
    # depth: planting a weed deeper would bury its lowest leaves, which already touch
    # the soil, and eat the 2 mm the gripper pads clear the bed by.
    root: float = 1.0
    # Weeds: the stalk joints' spring and damping, and how softly the soil holds the
    # root (the root weld's time constant), each against the hand-written plant.
    stiffness: float = 1.0
    damping: float = 1.0
    soil_give: float = 1.0

    def quat(self):
        """Turned about the vertical, then leaned."""
        turn = np.array([math.cos(self.yaw / 2), 0, 0, math.sin(self.yaw / 2)])
        axis = np.array([-math.sin(self.lean_toward), math.cos(self.lean_toward), 0])
        tilt = np.array([math.cos(self.lean / 2), *(math.sin(self.lean / 2) * axis)])
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, tilt, turn)
        return out


@dataclasses.dataclass(frozen=True)
class Bed:
    name: str
    x: float  # centre of the box
    y: float
    z: float
    length: float  # half sizes, as the MJCF gives them
    width: float
    depth: float
    friction: float  # sliding friction of the soil
    roll: float = 0.0  # radians about the bed's long axis: a slope across it
    pitch: float = 0.0  # radians about the vertical-crossing axis: a slope along it

    def quat(self):
        roll = [math.cos(self.roll / 2), math.sin(self.roll / 2), 0, 0]
        pitch = [math.cos(self.pitch / 2), 0, math.sin(self.pitch / 2), 0]
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, np.array(pitch), np.array(roll))
        return out

    def top(self, x, y):
        """Height of the soil surface at (x, y)."""
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, self.quat())
        up = rotation.reshape(3, 3)[:, 2]
        centre = np.array([self.x, self.y, self.z]) + up * self.depth
        return float(centre[2] - (up[0] * (x - centre[0]) + up[1] * (y - centre[1])) / up[2])

    def holds(self, x, y, keep):
        """True when a footprint `keep` wide around (x, y) lies on this bed."""
        return (abs(x - self.x) <= self.length - keep
                and abs(y - self.y) <= self.width - keep)


@dataclasses.dataclass(frozen=True)
class Light:
    dir: tuple[float, float, float]  # the sun's
    diffuse: tuple[float, float, float]
    ambient: tuple[float, float, float]
    headlight: tuple[float, float, float]  # the viewer headlight's diffuse
    headlight_ambient: tuple[float, float, float]
    preset: str = "nominal"
    exposure: float = 1.0


@dataclasses.dataclass(frozen=True)
class Manifest:
    beds: tuple[Bed, ...]
    weeds: tuple[Plant, ...]
    strawberries: tuple[Plant, ...]
    light: Light
    soil: str  # bed texture, relative to the scene's folder
    path: str  # walkway texture, likewise
    camera: tuple[float, float, float] = (0.0, 0.0, 0.0)  # overview camera offset, m
    occlusion: str = "nominal"  # how closely the crop crowds the weeds; see OCCLUSION
    family: str = "nominal"
    variant: int | None = None
    split: str = "train"  # train, held_out_family or held_out_variant
    seed: int | None = None

    SOILS = {
        "dry": ("assets/ground/variants/dirt_diff_2k.jpg",
                "assets/ground/variants/rocky_trail_02_diff_2k.jpg",
                "assets/ground/variants/ground108_color_2k.jpg"),
        "loose": ("assets/ground/brown_mud_03_diff_2k.png",
                  "assets/ground/variants/farm_soil_diff_2k.jpg"),
    }
    # Sun elevation in degrees, sun colour and strength, the grey ambient, and the
    # headlight's diffuse and ambient. Midday matches the hand-written scene; low light
    # keeps the sun but drops the fill, so the shadows go dark.
    LIGHTS = {
        "midday": {"elevation": (55, 75), "colour": (1.0, 0.97, 0.9),
                   "strength": (1.1, 1.3), "ambient": 0.12, "headlight": (0.35, 0.10)},
        "evening_warm": {"elevation": (10, 25), "colour": (1.0, 0.72, 0.45),
                         "strength": (0.8, 1.0), "ambient": 0.08, "headlight": (0.25, 0.07)},
        "evening_cool": {"elevation": (10, 25), "colour": (0.7, 0.8, 1.0),
                         "strength": (0.7, 0.9), "ambient": 0.08, "headlight": (0.25, 0.07)},
        "low_light": {"elevation": (20, 40), "colour": (0.85, 0.85, 0.9),
                      "strength": (0.45, 0.6), "ambient": 0.05, "headlight": (0.15, 0.04)},
    }
    # The walkway, from the ground textures already in the repository; never the same as
    # the beds' soil, so the two stay apart.
    PATHS = ("assets/ground/sparse_grass_diff_4k.png",
             "assets/ground/variants/rocky_trail_02_diff_2k.jpg",
             "assets/ground/variants/dirt_diff_2k.jpg",
             "assets/ground/variants/ground108_color_2k.jpg")
    EXPOSURE = (0.85, 1.15)  # one factor on every light in the scene
    # The six scene families. Each fixes the conditions; a variant draws everything else.
    FAMILIES = {
        "midday_dry": {"lights": ("midday",), "soils": ("dry",)},
        "midday_loose": {"lights": ("midday",), "soils": ("loose",)},
        "evening_warm": {"lights": ("evening_warm",), "soils": ("dry", "loose")},
        "evening_cool": {"lights": ("evening_cool",), "soils": ("dry", "loose")},
        "low_light": {"lights": ("low_light",), "soils": ("dry", "loose")},
        "mixed_clutter": {"lights": ("midday", "evening_warm", "evening_cool", "low_light"),
                          "soils": ("dry", "loose"), "weeds": (7, 9), "strawberries": (8, 8),
                          "occlusion": ("crowded",)},
    }
    # Kept out of training for evaluation: one whole family, a condition never seen, and
    # the last variant of every other family, a layout never seen in a known condition.
    HELD_OUT_FAMILY = "evening_cool"
    HELD_OUT_VARIANT = 4
    WEED_SPECIES = ("sowthistle", "shepherd")
    WEED_COUNT = (3, 9)
    STRAWBERRY_COUNT = (6, 8)

    # Ground. Half widths of the beds, the gap between them, the walkway's offset from
    # the rail, and each bed's height offset, cross slope and slope along it.
    BED_WIDTH = (0.30, 0.375)
    WALKWAY = (0.30, 0.40)
    RAIL_CLEARANCE = 0.15  # nearest a bed may come to the rail: the bin rides out to 0.122
    BED_RISE = (-0.01, 0.015)
    BED_ROLL = math.radians(1.0)
    BED_PITCH = math.radians(0.2)
    CAMERA_SHIFT = 0.03  # metres, each axis, for the overview camera only

    # Plants.
    LEAN = math.radians(5)
    STRAWBERRY_SIZE = (0.85, 1.1)
    STRAWBERRY_SPREAD = (0.9, 1.1)
    ROOT = (1.0, 20 / 15)  # root strength: 15 to 20 N of straight pull
    YAW_SEPARATION = math.radians(15)  # least turn between two strawberries in a scene
    # Physics, each a factor on the hand-written value, and the soil's sliding friction.
    # Measured stable at both ends: the stalk stands and springs back, the roots rest
    # unloaded and give way under a pull past their strength.
    STALK_STIFFNESS = (0.8, 1.25)
    STALK_DAMPING = (0.8, 1.25)
    SOIL_GIVE = (0.8, 1.25)
    BED_FRICTION = (0.5, 0.9)
    # How closely the crop crowds the weeds, as the clear gap between a weed's footprint
    # and the nearest strawberry's. Open: every weed at least OPEN clear. Crowded: half
    # the weeds, rounded up, within CROWDED of one. Partial: whatever the draw gives.
    OCCLUSION = ("open", "partial", "crowded")
    OPEN = 0.20
    CROWDED = 0.08
    DUPLICATE = 0.10  # two scenes whose plants all stand this close are one scene

    # Horizontal reach of each plant's collision shapes from its stem, in metres,
    # rounded up to cover the leaves as drawn. The strawberry collides as its convex
    # hull, which is why it is so wide; it scales with the plant.
    RADIUS = {"sowthistle": 0.08, "shepherd": 0.03, "strawberry": 0.15}
    GAP = 0.01  # clear space kept between two footprints, and a footprint and a bed edge
    # How far from the rail, across the bed, the extraction sequence can take each weed.
    # Measured with Uprooter: both species come out from 0.40 m. The sowthistle comes
    # out to 0.56 and not at 0.60, the taller plant lining the hand up higher. The
    # shepherd comes out at 0.58 wherever it stands along the bed, leaned 5 degrees or
    # upright; from 0.61 the pads graze the soil and, at some places along the bed, the
    # grip fails.
    REACH = {"sowthistle": (0.45, 0.55), "shepherd": (0.45, 0.57)}
    # The hand reaches a weed from the walkway, so no strawberry may stand between the
    # two: none nearer the rail than a weed and within this far of it along the bed,
    # the hand and wrist spanning about 0.1 m either side of the stalk.
    HAND = 0.10
    ATTEMPTS = 1000

    @classmethod
    def nominal(cls):
        """The hand-written scene, read from its XML so the two cannot drift apart."""
        scene = ET.parse(SCENE).getroot()
        weeds = ET.parse(SCENE.with_name("weeds.xml")).getroot()
        sun = scene.find("worldbody/light[@name='sun']")
        head = scene.find("visual/headlight")
        return cls(
            beds=tuple(_bed(g) for g in scene.findall("worldbody/geom")
                       if g.get("name").startswith("bed")),
            weeds=tuple(_plant(b) for b in weeds.findall("worldbody/body")),
            strawberries=tuple(_plant(b) for b in scene.findall("worldbody/body")
                               if b.get("name").startswith("strawberry")),
            light=Light(*(_floats(sun.get(k)) for k in ("dir", "diffuse", "ambient")),
                        *(_floats(head.get(k)) for k in ("diffuse", "ambient"))),
            soil=scene.find("asset/texture[@name='soil']").get("file"),
            path=scene.find("asset/texture[@name='grass']").get("file"),
        )

    @classmethod
    def sample(cls, seed, family=None):
        """One scene of `family` (drawn from the seed if not given).

        The conditions come first: light, soil, ground and counts. Then each plant is
        redrawn until it clears those already down, and a layout that cannot fit its
        next plant is started again. Every draw comes from the one generator seeded
        here, so a seed always lands on the same accepted manifest. Counts are drawn
        once, outside the retries, or crowded scenes would be rejected more often and
        the dataset would lean sparse.
        """
        rng = np.random.default_rng(seed)
        family = family or list(cls.FAMILIES)[rng.integers(len(cls.FAMILIES))]
        rules = cls.FAMILIES[family]
        soils = [s for group in rules["soils"] for s in cls.SOILS[group]]
        soil = soils[rng.integers(len(soils))]
        paths = [p for p in cls.PATHS if p != soil]
        occlusion = rules.get("occlusion", cls.OCCLUSION)
        base = dataclasses.replace(
            cls.nominal(), family=family, seed=seed, split=cls._split(family, None),
            light=cls._light(rng, rules["lights"][rng.integers(len(rules["lights"]))]),
            soil=soil, path=paths[rng.integers(len(paths))],
            beds=cls._ground(rng, cls.nominal().beds),
            camera=tuple(float(v) for v in rng.uniform(-cls.CAMERA_SHIFT, cls.CAMERA_SHIFT, 3)),
            occlusion=occlusion[rng.integers(len(occlusion))])
        low, high = rules.get("weeds", cls.WEED_COUNT)
        species = [str(s) for s in rng.choice(cls.WEED_SPECIES, rng.integers(low, high + 1))]
        low, high = rules.get("strawberries", cls.STRAWBERRY_COUNT)
        strawberries = rng.integers(low, high + 1)
        # The wide strawberries go down first, while there is still room for them. In a
        # crowded scene the first half of the weeds must grow close to one.
        crowd = math.ceil(len(species) / 2) if base.occlusion == "crowded" else 0
        plan = ([("strawberry", f"strawberry_{i + 1}") for i in range(strawberries)]
                + [(s, f"{s}_{species[:i + 1].count(s)}") for i, s in enumerate(species)])
        for _ in range(cls.ATTEMPTS):
            placed = []
            for i, (kind, name) in enumerate(plan):
                crowded = strawberries <= i < strawberries + crowd
                for _ in range(cls.ATTEMPTS):
                    plant = cls._scatter(rng, base.beds, kind, name)
                    if base._fits(plant, placed, crowded):
                        placed.append(plant)
                        break
                else:
                    break
            else:
                manifest = dataclasses.replace(base, strawberries=tuple(placed[:strawberries]),
                                               weeds=tuple(placed[strawberries:]))
                if not (manifest.problems() or manifest.repeats() or manifest.misses()):
                    return manifest
        raise RuntimeError(f"no valid layout for seed {seed}")

    def _fits(self, plant, placed, crowded):
        """Whether `plant` can join `placed`: every pairwise rule, then the scene's
        occlusion, which only a weed's distance to the strawberries decides."""
        for other in placed:
            if self._clash(plant, other) or self._same_way(plant, other):
                return False
            if (plant.species in self.REACH and other.species in self.REACH
                    and abs(plant.x - other.x) < Prompt.TIE):
                return False  # the weeds could not be counted along the rail
        if plant.species not in self.REACH:
            return True
        gap = self._crowding(plant, [p for p in placed if p.species == "strawberry"])
        return gap <= self.CROWDED if crowded else (self.occlusion != "open" or gap >= self.OPEN)

    @classmethod
    def catalogue(cls, count=5):
        """The first `count` variants of every family, none a near-copy of any other
        scene in any family.

        Drawn round-robin, variant 0 of every family before variant 1 of any, so a
        family's first variants never depend on how many were asked for. Each family's
        seeds run from 1000 times its number.
        """
        seeds = {family: 1000 * (i + 1) for i, family in enumerate(cls.FAMILIES)}
        found = {family: [] for family in cls.FAMILIES}
        accepted = []
        for variant in range(count):
            for family in cls.FAMILIES:
                for _ in range(cls.ATTEMPTS):
                    manifest = cls.sample(seeds[family], family)
                    seeds[family] += 1
                    if not any(manifest.similar(other) for other in accepted):
                        break
                else:
                    raise RuntimeError(f"no distinct variant {variant} of {family}")
                manifest = dataclasses.replace(manifest, variant=variant,
                                               split=cls._split(family, variant))
                found[family].append(manifest)
                accepted.append(manifest)
        return found

    @classmethod
    def variants(cls, family, count=5):
        """The first `count` scenes of `family`, as the catalogue has them."""
        return cls.catalogue(count)[family]

    @classmethod
    def _split(cls, family, variant):
        if family == cls.HELD_OUT_FAMILY:
            return "held_out_family"
        return "held_out_variant" if variant == cls.HELD_OUT_VARIANT else "train"

    def to_json(self):
        """Every field, then what follows from them. All of it is hidden metadata: for
        auditing a policy, never for showing one."""
        return json.dumps({
            "manifest": dataclasses.asdict(self),
            "metadata": {
                "weed_count": len(self.weeds),
                "strawberry_count": len(self.strawberries),
                "species": {s: sum(p.species == s for p in self.weeds)
                            for s in self.WEED_SPECIES},
                "root_break_force_n": {p.name: ROOT_BREAK_FORCE * p.root for p in self.weeds},
                "root_break_torque_nm": {p.name: ROOT_BREAK_TORQUE * p.root
                                         for p in self.weeds},
                "crowding_m": {p.name: self._crowding(p, self.strawberries)
                               for p in self.weeds},
                "prompts": [dataclasses.asdict(p) for p in Prompt.valid(self)],
            },
        }, indent=2)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    @classmethod
    def load(cls, path):
        fields = json.loads(Path(path).read_text())["manifest"]
        return cls(**{
            **fields,
            "beds": tuple(Bed(**b) for b in fields["beds"]),
            "weeds": tuple(Plant(**p) for p in fields["weeds"]),
            "strawberries": tuple(Plant(**p) for p in fields["strawberries"]),
            "light": Light(**{k: tuple(v) if isinstance(v, list) else v
                              for k, v in fields["light"].items()}),
            "camera": tuple(fields["camera"]),
        })

    @classmethod
    def _light(cls, rng, preset):
        rules = cls.LIGHTS[preset]
        elevation = math.radians(rng.uniform(*rules["elevation"]))
        azimuth = rng.uniform(0, 2 * math.pi)
        strength = rng.uniform(*rules["strength"])
        exposure = float(rng.uniform(*cls.EXPOSURE))

        def grey(level):
            return (level * exposure,) * 3

        return Light(
            dir=(-math.cos(elevation) * math.cos(azimuth),
                 -math.cos(elevation) * math.sin(azimuth), -math.sin(elevation)),
            diffuse=tuple(c * strength * exposure for c in rules["colour"]),
            ambient=grey(rules["ambient"]), headlight=grey(rules["headlight"][0]),
            headlight_ambient=grey(rules["headlight"][1]),
            preset=preset, exposure=exposure)

    @classmethod
    def _ground(cls, rng, nominal):
        """Both beds moved and reshaped around a walkway of drawn width and offset."""
        walkway = rng.uniform(*cls.WALKWAY)
        slack = walkway / 2 - cls.RAIL_CLEARANCE
        offset = rng.uniform(-slack, slack)
        beds = []
        for bed in nominal:
            side = math.copysign(1.0, bed.y)
            width = rng.uniform(*cls.BED_WIDTH)
            beds.append(dataclasses.replace(
                bed, y=float(offset + side * (walkway / 2 + width)), width=float(width),
                z=float(bed.z + rng.uniform(*cls.BED_RISE)),
                roll=float(rng.uniform(-cls.BED_ROLL, cls.BED_ROLL)),
                pitch=float(rng.uniform(-cls.BED_PITCH, cls.BED_PITCH)),
                friction=float(rng.uniform(*cls.BED_FRICTION))))
        return tuple(beds)

    @classmethod
    def _scatter(cls, rng, beds, species, name):
        """A plant anywhere on either bed that keeps its footprint off the edges."""
        bed = beds[rng.integers(len(beds))]
        size = spread = root = 1.0
        if species == "strawberry":
            size = float(rng.uniform(*cls.STRAWBERRY_SIZE))
            spread = float(rng.uniform(*cls.STRAWBERRY_SPREAD))
        else:
            root = float(rng.uniform(*cls.ROOT))
        keep = cls.RADIUS[species] * size * spread + cls.GAP
        low, high = bed.y - bed.width + keep, bed.y + bed.width - keep
        if species in cls.REACH:
            near, far = cls.REACH[species]
            low, high = ((max(low, near), min(high, far)) if bed.y > 0
                         else (max(low, -far), min(high, -near)))
        x = float(rng.uniform(bed.x - bed.length + keep, bed.x + bed.length - keep))
        y = float(rng.uniform(low, high))
        lean = lean_toward = 0.0
        stiffness = damping = soil_give = 1.0
        if species != "strawberry":
            lean = float(rng.uniform(0, cls.LEAN))
            lean_toward = float(rng.uniform(0, 2 * math.pi))
            stiffness = float(rng.uniform(*cls.STALK_STIFFNESS))
            damping = float(rng.uniform(*cls.STALK_DAMPING))
            soil_give = float(rng.uniform(*cls.SOIL_GIVE))
        return Plant(name, species, x, y, bed.top(x, y), float(rng.uniform(0, 2 * math.pi)),
                     lean, lean_toward, size, spread, root, stiffness, damping, soil_give)

    def problems(self):
        """Every physical rule this scene breaks, as readable lines; empty if none."""
        found = []
        for bed in self.beds:
            if abs(bed.y) - bed.width < self.RAIL_CLEARANCE:
                found.append(f"{bed.name} crowds the rail")
        plants = self.weeds + self.strawberries
        for p in plants:
            keep = self._radius(p) + self.GAP
            if not any(bed.holds(p.x, p.y, keep) for bed in self.beds):
                found.append(f"{p.name} is within {keep:.2f} m of a bed edge")
            elif not any(bed.holds(p.x, p.y, 0) and abs(bed.top(p.x, p.y) - p.z) < 1e-6
                         for bed in self.beds):
                found.append(f"{p.name} is not planted on the soil")
            if p.lean > self.LEAN:
                found.append(f"{p.name} leans too far")
        for p in self.weeds:
            near, far = self.REACH[p.species]
            if not near <= abs(p.y) <= far:
                found.append(f"{p.name} is out of the arm's reach")
        for i, a in enumerate(plants):
            for b in plants[i + 1:]:
                if why := self._clash(a, b):
                    found.append(why)
        return found

    def repeats(self):
        """Strawberries turned too nearly the same way. Not a physical fault -- the
        hand-written scene has them all alike -- but a sampled scene must vary them."""
        return [f"{a.name} and {b.name} face the same way"
                for i, a in enumerate(self.strawberries) for b in self.strawberries[i + 1:]
                if self._same_way(a, b)]

    def misses(self):
        """What a sampled scene was drawn to have and lacks: a valid prompt of every
        REQUIRED kind, weeds that can be counted along the rail, and its occlusion."""
        found = [f"no prompt of kind {kind}" for kind in Prompt.missing(self)]
        gaps = [self._crowding(p, self.strawberries) for p in self.weeds]
        if self.occlusion == "open" and min(gaps, default=math.inf) < self.OPEN:
            found.append("a weed grows too close to the crop for an open scene")
        if (self.occlusion == "crowded"
                and sum(g <= self.CROWDED for g in gaps) < math.ceil(len(gaps) / 2)):
            found.append("too few weeds grow close to the crop for a crowded scene")
        return found

    def similar(self, other):
        """True when `other` is too like this scene to count as another one: every
        plant has one of its kind standing within DUPLICATE of it, or the strawberries
        are turned the same ways, each within YAW_SEPARATION of one there."""
        def matched(mine, theirs, close):
            theirs = list(theirs)
            for p in mine:
                q = next((q for q in theirs if close(p, q)), None)
                if q is None:
                    return False
                theirs.remove(q)
            return not theirs

        return (matched(self.weeds + self.strawberries, other.weeds + other.strawberries,
                        lambda p, q: p.species == q.species
                        and math.hypot(p.x - q.x, p.y - q.y) < self.DUPLICATE)
                or matched(self.strawberries, other.strawberries,
                           lambda p, q: _turn(p.yaw, q.yaw) < self.YAW_SEPARATION))

    @classmethod
    def _same_way(cls, a, b):
        return (a.species == b.species == "strawberry"
                and _turn(a.yaw, b.yaw) < cls.YAW_SEPARATION)

    @classmethod
    def _radius(cls, plant):
        return cls.RADIUS[plant.species] * plant.size * plant.spread

    @classmethod
    def _crowding(cls, weed, strawberries):
        """The clear gap from a weed's footprint to the nearest strawberry's, metres."""
        return min((math.hypot(weed.x - s.x, weed.y - s.y) - cls._radius(weed)
                    - cls._radius(s) - cls.GAP for s in strawberries), default=math.inf)

    @classmethod
    def _clash(cls, a, b):
        """Why two plants cannot stand where they are, or None if they can."""
        short = cls._radius(a) + cls._radius(b) + cls.GAP - math.hypot(a.x - b.x, a.y - b.y)
        if short > 0:
            return f"{a.name} and {b.name} are {short:.3f} m too close"
        weed, crop = (a, b) if b.species == "strawberry" else (b, a)
        if (crop.species == "strawberry" and weed.species in cls.REACH
                and weed.y * crop.y > 0 and abs(crop.y) < abs(weed.y)
                and abs(crop.x - weed.x) < cls.HAND + cls._radius(crop)):
            return f"{crop.name} stands in the arm's way to {weed.name}"
        return None


def build_env(manifest, out_dir, **env_args):
    """The environment for `manifest`: its MJCF written into `out_dir`, then compiled.

    This is the only way a sampled layout reaches the simulation. A scene is compiled
    once, when the environment is built, so `reset(seed=...)` cannot change it; the
    seed that decides the layout is the manifest's. The roots take the manifest's
    strengths at every reset that does not name its own.
    """
    return WeedExtractionEnv(scene=write_scene(manifest, out_dir),
                             root_strength={p.name: p.root for p in manifest.weeds},
                             **env_args)


def write_catalogue(out_dir, count=5):
    """Save every family's first `count` variants as `<family>/<variant>/manifest.json`
    under `out_dir`, with an index of them all, and return the index's path."""
    out_dir = Path(out_dir)
    index = []
    for family, variants in Manifest.catalogue(count).items():
        for m in variants:
            path = m.save(out_dir / family / str(m.variant) / "manifest.json")
            index.append({"family": family, "variant": m.variant, "seed": m.seed,
                          "split": m.split, "manifest": str(path.relative_to(out_dir))})
    path = out_dir / "index.json"
    path.write_text(json.dumps(index, indent=2))
    return path


def write_scene(manifest, out_dir):
    """Write the scene for `manifest` into `out_dir`, with the manifest beside it as
    `manifest.json`, and return the scene's path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    field = SCENE.parent
    scene = ET.parse(SCENE)
    root = scene.getroot()

    # MuJoCo resolves includes, meshdir and textures from the top-level file's folder,
    # which is now out_dir, so point each back at the originals.
    robot = root.find("include[@file!='weeds.xml']")
    robot_file = (field / robot.get("file")).resolve()
    robot.set("file", str(robot_file))
    meshdir = (field / ET.parse(robot_file).find("compiler").get("meshdir")).resolve()
    root.insert(list(root).index(robot) + 1,
                ET.Element("compiler", meshdir=str(meshdir), texturedir=str(field)))

    for name, texture in (("soil", manifest.soil), ("grass", manifest.path)):
        image = field / texture
        if image.suffix != ".png":  # MuJoCo 3.8 decodes PNG but not JPEG
            png = out_dir / f"{image.stem}.png"
            if not png.exists():
                cv2.imwrite(str(png), cv2.imread(str(image)))
            image = png
        root.find(f"asset/texture[@name='{name}']").set("file", str(image))

    light = manifest.light
    sun = root.find("worldbody/light[@name='sun']")
    for key in ("dir", "diffuse", "ambient"):
        sun.set(key, _text(getattr(light, key)))
    head = root.find("visual/headlight")
    head.set("diffuse", _text(light.headlight))
    head.set("ambient", _text(light.headlight_ambient))

    camera = root.find("worldbody/camera[@name='rgb_camera']")
    camera.set("pos", _text(np.add(_floats(camera.get("pos")), manifest.camera)))

    for bed in manifest.beds:
        geom = root.find(f"worldbody/geom[@name='{bed.name}']")
        geom.set("pos", _text((bed.x, bed.y, bed.z)))
        geom.set("size", _text((bed.length, bed.width, bed.depth)))
        geom.set("quat", _text(bed.quat()))
        geom.set("friction", _text((bed.friction, *_floats(geom.get("friction"))[1:])))

    worldbody = root.find("worldbody")
    old = [b for b in worldbody.findall("body") if b.get("name").startswith("strawberry")]
    at = list(worldbody).index(old[0])
    for body in old:
        worldbody.remove(body)
    mesh = root.find("asset/mesh[@name='strawberry_mesh']")
    assets = next(a for a in root.findall("asset") if mesh in list(a))
    for i, plant in enumerate(manifest.strawberries):
        body = _placed(old[0], old[0].get("name"), plant)
        if plant.size != 1 or plant.spread != 1:  # a mesh scales as an asset, not a geom
            scaled = copy.deepcopy(mesh)
            scaled.set("name", f"strawberry_mesh_{plant.name}")
            across = plant.size * plant.spread
            scaled.set("scale", _text((across, across, plant.size)))
            assets.append(scaled)
            body.find("geom").set("mesh", scaled.get("name"))
        worldbody.insert(at + i, body)

    weeds = ET.parse(SCENE.with_name("weeds.xml"))
    _template_weeds(weeds.getroot(), manifest.weeds)

    for tree, name in ((scene, "scene.xml"), (weeds, "weeds.xml")):
        ET.indent(tree)
        tree.write(out_dir / name)
    manifest.save(out_dir / "manifest.json")
    return out_dir / "scene.xml"


def _template_weeds(root, plants):
    """Rebuild every weed, and everything naming one, from the first of its species.

    Each species' first weed is the template for its body, its root and grasp welds and
    its contact excludes; the other hand-written weeds are dropped. A run of elements
    for one template is copied as a block per weed, which keeps the original order.
    """
    worldbody = root.find("worldbody")
    stalk = root.find("default//default[@class='stalk']/joint")
    names = [b.get("name") for b in worldbody.findall("body")]
    templates = {}
    for name in names:
        templates.setdefault(name.rsplit("_", 1)[0], name)
    of_species = {s: [p for p in plants if p.species == s] for s in templates}

    def owner(elem):
        return next((n for n in names for v in elem.attrib.values()
                     if v == n or v.startswith(n + "_")), None)

    for section in (worldbody, root.find("equality"), root.find("contact")):
        kept, block = [], []

        def flush():
            if block:
                template = owner(block[0])
                for plant in of_species[template.rsplit("_", 1)[0]]:
                    kept.extend(_placed(e, template, plant, stalk) for e in block)
                block.clear()

        for elem in section:
            weed = owner(elem)
            if block and weed != owner(block[0]):
                flush()
            if weed is None:
                kept.append(elem)
            elif weed in templates.values():
                block.append(elem)
        flush()
        section[:] = kept


def _placed(elem, template, plant, stalk=None):
    """A copy of `elem` renamed from `template` to `plant`, moved to where it stands and
    given its physics. `stalk` is the weeds' default stalk joint, which the factors scale;
    a factor of 1 leaves the class to set the value."""
    elem = copy.deepcopy(elem)
    for e in elem.iter():
        for key, value in e.attrib.items():
            if value == template or value.startswith(template + "_"):
                e.set(key, plant.name + value[len(template):])
    if elem.tag == "body":
        elem.set("pos", _text((plant.x, plant.y, plant.z)))
        elem.set("quat", _text(plant.quat()))
        for joint in elem.iter("joint"):
            for key in ("stiffness", "damping"):
                if (factor := getattr(plant, key)) != 1:
                    joint.set(key, _text((float(stalk.get(key)) * factor,)))
    elif "anchor" in elem.attrib:  # a root weld, anchored in the world where it stands
        elem.set("anchor", _text((plant.x, plant.y, plant.z)))
        if plant.soil_give != 1:
            timeconst, ratio = _floats(elem.get("solref"))
            elem.set("solref", _text((timeconst * plant.soil_give, ratio)))
    return elem


def _bed(geom):
    (x, y, z), (length, width, depth) = _floats(geom.get("pos")), _floats(geom.get("size"))
    return Bed(geom.get("name"), x, y, z, length, width, depth,
               friction=_floats(geom.get("friction"))[0])


def _plant(body):
    x, y, z = _floats(body.get("pos"))
    w, _, _, turn = _floats(body.get("quat", "1 0 0 0"))
    name = body.get("name")
    return Plant(name, name.rsplit("_", 1)[0], x, y, z, 2 * math.atan2(turn, w))


def _turn(a, b):
    """The angle between two headings, 0 to pi."""
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _floats(text):
    return tuple(float(v) for v in text.split())


def _text(values):
    return " ".join(f"{v:.9g}" for v in values)
