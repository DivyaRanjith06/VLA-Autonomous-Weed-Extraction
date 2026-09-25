import dataclasses
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from weed_extraction.env import SCENE, WeedExtractionEnv
from weed_extraction.ik import ArmIK, pose
from weed_extraction.prompts import Prompt
from weed_extraction.scene import Manifest, build_env, write_catalogue, write_scene
from weed_extraction.uproot import Uprooter

SAMPLES = [Manifest.sample(seed) for seed in range(100)]


def moved(manifest, name, **where):
    """`manifest` with the plant called `name` changed."""
    def move(plants):
        return tuple(dataclasses.replace(p, **where) if p.name == name else p for p in plants)
    return dataclasses.replace(manifest, weeds=move(manifest.weeds),
                               strawberries=move(manifest.strawberries))


def up(model, body):
    """The body's own z axis in the world."""
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.body_quat[model.body(body).id])
    return rotation.reshape(3, 3)[:, 2]


# Reproduction

def test_the_same_seed_gives_the_same_manifest():
    assert Manifest.sample(7) == Manifest.sample(7)
    assert Manifest.sample(7, "low_light") == Manifest.sample(7, "low_light")


def test_the_same_family_gives_the_same_variants():
    assert Manifest.variants("evening_cool") == Manifest.variants("evening_cool")


def test_different_seeds_give_different_valid_scenes():
    layouts = {tuple((p.species, p.x, p.y) for p in m.weeds + m.strawberries) for m in SAMPLES}
    assert len(layouts) == len(SAMPLES)
    for m in SAMPLES:
        assert not (m.problems() or m.repeats() or m.misses()), m.seed
        assert len({p.name for p in m.weeds + m.strawberries}) == len(m.weeds + m.strawberries)
    # Scenes of one species and of both, and every count, all turn up.
    assert {len({p.species for p in m.weeds}) for m in SAMPLES} == {1, 2}
    assert {len(m.weeds) for m in SAMPLES} == set(range(3, 10))
    assert {len(m.strawberries) for m in SAMPLES} == {6, 7, 8}


# Families and variants

@pytest.mark.parametrize("family", list(Manifest.FAMILIES))
def test_each_family_keeps_to_its_conditions(family):
    rules = Manifest.FAMILIES[family]
    low, high = rules.get("weeds", Manifest.WEED_COUNT)
    few, many = rules.get("strawberries", Manifest.STRAWBERRY_COUNT)
    variants = Manifest.variants(family)
    assert [m.variant for m in variants] == list(range(5))
    for m in variants:
        assert m.family == family and not (m.problems() or m.repeats() or m.misses())
        assert m.light.preset in rules["lights"]
        assert any(m.soil in Manifest.SOILS[group] for group in rules["soils"])
        assert low <= len(m.weeds) <= high and few <= len(m.strawberries) <= many
        assert m.occlusion in rules.get("occlusion", Manifest.OCCLUSION)


def test_no_scene_in_the_catalogue_is_a_near_copy_of_another_in_any_family():
    scenes = [m for variants in Manifest.catalogue().values() for m in variants]
    assert len({m.seed for m in scenes}) == len(scenes) == 5 * len(Manifest.FAMILIES)
    for i, a in enumerate(scenes):
        for b in scenes[i + 1:]:
            assert not a.similar(b), (a.family, a.variant, b.family, b.variant)


def test_a_near_copy_from_another_family_is_rejected(monkeypatch):
    """If one family's draw repeats another's layout, the catalogue moves on a seed."""
    real = Manifest.sample.__func__
    copy = Manifest.catalogue(1)["midday_dry"][0]

    def sample(cls, seed, family=None):
        if family == "midday_loose" and seed == 2000:  # its first draw copies midday_dry's
            return dataclasses.replace(copy, family=family, seed=seed)
        return real(cls, seed, family)

    monkeypatch.setattr(Manifest, "sample", classmethod(sample))
    assert Manifest.catalogue(1)["midday_loose"][0].seed == 2001


def test_asking_for_more_variants_keeps_the_first_ones():
    few, more = Manifest.catalogue(2), Manifest.catalogue(3)
    assert all(few[f] == more[f][:2] for f in Manifest.FAMILIES)


def test_the_held_out_split_is_one_family_and_one_variant_of_the_rest():
    for family, variants in Manifest.catalogue().items():
        for m in variants:
            if family == Manifest.HELD_OUT_FAMILY:
                assert m.split == "held_out_family"
            elif m.variant == Manifest.HELD_OUT_VARIANT:
                assert m.split == "held_out_variant"
            else:
                assert m.split == "train"
    assert Manifest.sample(3, Manifest.HELD_OUT_FAMILY).split == "held_out_family"
    assert Manifest.nominal().split == "train"


# Ground

def test_the_ground_varies_within_its_limits():
    nominal = Manifest.nominal().beds
    for m in SAMPLES:
        for bed, home in zip(m.beds, nominal):
            assert abs(bed.y) - bed.width >= Manifest.RAIL_CLEARANCE
            assert Manifest.BED_WIDTH[0] <= bed.width <= Manifest.BED_WIDTH[1]
            assert Manifest.BED_RISE[0] <= bed.z - home.z <= Manifest.BED_RISE[1]
            assert abs(bed.roll) <= Manifest.BED_ROLL and abs(bed.pitch) <= Manifest.BED_PITCH
            assert bed.y * home.y > 0  # each bed stays on its own side of the rail
            assert Manifest.BED_FRICTION[0] <= bed.friction <= Manifest.BED_FRICTION[1]
        assert all(abs(c) <= Manifest.CAMERA_SHIFT for c in m.camera)
    for field in ("y", "width", "z", "roll", "pitch", "friction"):
        assert len({getattr(m.beds[0], field) for m in SAMPLES}) == len(SAMPLES)


def test_a_sloped_bed_puts_its_soil_where_the_box_is(tmp_path):
    """Plants are planted at `Bed.top`, so it has to agree with what MuJoCo builds."""
    m = next(m for m in SAMPLES if abs(m.beds[0].roll) > math.radians(0.5))
    model = mujoco.MjModel.from_xml_path(str(write_scene(m, tmp_path)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bed = m.beds[0]
    for x, y in ((bed.x - 2, bed.y - 0.2), (bed.x + 2, bed.y + 0.2), (bed.x, bed.y)):
        start = np.array([x, y, 1.0])
        hit = mujoco.mj_ray(model, data, start, np.array([0, 0, -1.0]), None, 1, -1,
                            np.zeros(1, np.int32))
        assert 1.0 - hit == pytest.approx(bed.top(x, y), abs=1e-6)


# Plants

def test_plants_vary_within_their_limits():
    weeds = [p for m in SAMPLES for p in m.weeds]
    strawberries = [p for m in SAMPLES for p in m.strawberries]
    assert all(0 <= p.lean <= Manifest.LEAN for p in weeds)
    assert all(Manifest.ROOT[0] <= p.root <= Manifest.ROOT[1] for p in weeds)
    assert all(Manifest.STRAWBERRY_SIZE[0] <= p.size <= Manifest.STRAWBERRY_SIZE[1]
               and Manifest.STRAWBERRY_SPREAD[0] <= p.spread <= Manifest.STRAWBERRY_SPREAD[1]
               for p in strawberries)
    for key, bounds in (("stiffness", Manifest.STALK_STIFFNESS),
                        ("damping", Manifest.STALK_DAMPING), ("soil_give", Manifest.SOIL_GIVE)):
        values = [getattr(p, key) for p in weeds]
        assert all(bounds[0] <= v <= bounds[1] for v in values)
        assert min(values) < bounds[0] + 0.05 and max(values) > bounds[1] - 0.05
    assert all(p.lean == 0 and p.root == p.stiffness == p.damping == p.soil_give == 1
               for p in strawberries)
    assert all(p.size == p.spread == 1 for p in weeds)
    assert max(p.lean for p in weeds) > 0.9 * Manifest.LEAN
    assert max(p.root for p in weeds) > 0.95 * Manifest.ROOT[1]


def test_the_scene_is_built_as_the_manifest_says(tmp_path):
    m = Manifest.variants("mixed_clutter", 1)[0]
    model = mujoco.MjModel.from_xml_path(str(write_scene(m, tmp_path)))
    for p in m.weeds:
        tilt = math.degrees(math.acos(up(model, p.name)[2]))
        assert tilt == pytest.approx(math.degrees(p.lean), abs=1e-6)
        assert model.body(p.name).pos == pytest.approx([p.x, p.y, p.z])
    for p in m.strawberries:
        geom = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(p.name).id)
        across = p.size * p.spread
        assert model.mesh_scale[model.geom_dataid[geom]] == pytest.approx([across, across, p.size])
    assert model.light("sun").diffuse == pytest.approx(m.light.diffuse)
    assert model.light("sun").ambient == pytest.approx(m.light.ambient)
    assert model.vis.headlight.diffuse == pytest.approx(m.light.headlight)
    nominal = mujoco.MjModel.from_xml_path(str(SCENE))
    assert (model.camera("rgb_camera").pos
            == pytest.approx(nominal.camera("rgb_camera").pos + m.camera))


# Lighting

@pytest.mark.parametrize("preset", list(Manifest.LIGHTS))
def test_each_light_keeps_to_its_preset(preset):
    rules = Manifest.LIGHTS[preset]
    family = next(f for f, r in Manifest.FAMILIES.items() if r["lights"] == (preset,))
    for light in (Manifest.sample(seed, family).light for seed in range(30)):
        assert light.preset == preset
        elevation = math.degrees(math.asin(-light.dir[2] / np.linalg.norm(light.dir)))
        assert rules["elevation"][0] <= elevation <= rules["elevation"][1]
        assert Manifest.EXPOSURE[0] <= light.exposure <= Manifest.EXPOSURE[1]
        # The sun's colour is the preset's, scaled by its strength and the exposure.
        strength = np.array(light.diffuse) / (np.array(rules["colour"]) * light.exposure)
        assert np.allclose(strength, strength[0])
        assert rules["strength"][0] <= strength[0] <= rules["strength"][1]
        assert light.ambient == pytest.approx((rules["ambient"] * light.exposure,) * 3)


def test_evening_and_low_light_are_darker_than_midday():
    def brightness(preset):
        return np.mean([sum(m.light.diffuse) for m in SAMPLES if m.light.preset == preset])
    assert brightness("low_light") < brightness("evening_cool") < brightness("midday")
    assert brightness("evening_warm") < brightness("midday")
    warm = [m.light.diffuse for m in SAMPLES if m.light.preset == "evening_warm"]
    cool = [m.light.diffuse for m in SAMPLES if m.light.preset == "evening_cool"]
    assert all(r > b for r, _, b in warm) and all(b > r for r, _, b in cool)


# Rules

def test_the_nominal_scene_passes_every_physical_rule():
    """The hand-written scene stays the reference: the rules must not reject it."""
    assert Manifest.nominal().problems() == []


@pytest.mark.parametrize("name, where, complaint", [
    ("shepherd_1", {"x": 0.40}, "sowthistle_2 and shepherd_1 are"),        # weed and weed
    ("shepherd_1", {"x": 0.05}, "shepherd_1 and strawberry_2 are"),        # weed and crop
    ("strawberry_2", {"spread": 1.3}, "shepherd_1 and strawberry_2 are"),  # a wider crop
    ("strawberry_1", {"y": -0.80}, "strawberry_1 is within"),              # off the side
    ("shepherd_3", {"x": -1.68}, "shepherd_3 is within"),                  # off the end
    ("shepherd_1", {"z": 0.12}, "shepherd_1 is not planted on the soil"),
    ("shepherd_1", {"lean": math.radians(8)}, "shepherd_1 leans too far"),
    ("sowthistle_1", {"y": -0.60}, "sowthistle_1 is out of the arm's reach"),
    ("shepherd_3", {"y": 0.62}, "shepherd_3 is out of the arm's reach"),
    ("shepherd_1", {"y": -0.40}, "shepherd_1 is out of the arm's reach"),
    ("strawberry_2", {"x": 0.05, "y": -0.33}, "strawberry_2 stands in the arm's way to shepherd_1"),
])
def test_each_placement_rule_rejects_a_scene_that_breaks_it(name, where, complaint):
    problems = moved(Manifest.nominal(), name, **where).problems()
    assert len(problems) == 1 and problems[0].startswith(complaint), problems


def test_a_bed_may_not_crowd_the_rail():
    nominal = Manifest.nominal()
    crowded = dataclasses.replace(nominal.beds[0], y=-0.45)
    assert dataclasses.replace(nominal, beds=(crowded, nominal.beds[1])).problems() == [
        "bed1 crowds the rail"]


def test_strawberries_turned_alike_are_repeats():
    """The hand-written strawberries all face the same way; a sampled scene may not."""
    nominal = Manifest.nominal()
    assert len(nominal.repeats()) == 28  # every pair of the eight
    turned = dataclasses.replace(nominal, strawberries=tuple(
        dataclasses.replace(p, yaw=math.radians(20 * i)) for i, p in enumerate(nominal.strawberries)))
    assert turned.repeats() == []
    close = moved(turned, "strawberry_2", yaw=math.radians(10))
    assert close.repeats() == ["strawberry_1 and strawberry_2 face the same way"]


def test_near_copies_of_a_scene_are_similar():
    m = Manifest.variants("midday_dry", 1)[0]
    assert m.similar(m)
    nudged = dataclasses.replace(m, weeds=tuple(dataclasses.replace(p, x=p.x + 0.05) for p in m.weeds),
                                 strawberries=tuple(dataclasses.replace(p, x=p.x + 0.05)
                                                    for p in m.strawberries))
    assert m.similar(nudged)
    moved_away = dataclasses.replace(m, strawberries=tuple(
        dataclasses.replace(p, x=p.x + 1.0) for p in m.strawberries))
    assert m.similar(moved_away)  # the strawberries still face the same ways
    other = next(o for o in SAMPLES if len(o.strawberries) == len(m.strawberries))
    assert not m.similar(other)


# Building and running: seed -> manifest -> MJCF -> environment

def differences(a, b):
    """Every compiled field two models disagree on, bar the text of their asset paths,
    which are absolute in a generated scene."""
    differs = []
    for field in dir(a):
        value = getattr(a, field)
        if field.startswith("_") or field in ("nbuffer", "npaths", "paths") \
                or field.endswith("_pathadr"):
            continue
        if isinstance(value, np.ndarray):
            other = getattr(b, field)
            if value.shape != other.shape or not np.array_equal(value, other):
                differs.append(field)
        elif isinstance(value, int) and value != getattr(b, field):
            differs.append(field)
    return differs


def test_the_nominal_manifest_regenerates_the_hand_written_scene(tmp_path):
    """Templating must change nothing but what the manifest says."""
    original = mujoco.MjModel.from_xml_path(str(SCENE))
    generated = mujoco.MjModel.from_xml_path(str(write_scene(Manifest.nominal(), tmp_path)))
    assert differences(original, generated) == []


def test_the_default_environment_is_still_the_hand_written_scene():
    env = WeedExtractionEnv(cameras=())
    env.reset()
    assert env.scene == SCENE and env.root_strength == {}
    assert set(env.roots.strength.values()) == {1.0}
    assert differences(env.model, mujoco.MjModel.from_xml_path(str(SCENE))) == []
    env.close()


def test_a_seed_builds_the_same_environment_every_time(tmp_path):
    first = build_env(Manifest.sample(11), tmp_path / "a", cameras=())
    second = build_env(Manifest.sample(11), tmp_path / "b", cameras=())
    assert first.scene == tmp_path / "a" / "scene.xml"
    assert differences(first.model, second.model) == []
    assert first.root_strength == second.root_strength
    first.close()
    second.close()


def test_the_environment_is_the_sampled_scene(tmp_path):
    manifest = Manifest.sample(11)
    env = build_env(manifest, tmp_path, cameras=())
    env.reset()
    model, data = env.model, env.data
    for p in manifest.weeds + manifest.strawberries:
        assert data.xpos[model.body(p.name).id] == pytest.approx([p.x, p.y, p.z])
    for bed in manifest.beds:
        assert model.geom(bed.name).pos == pytest.approx([bed.x, bed.y, bed.z])
        assert model.geom(bed.name).size == pytest.approx([bed.length, bed.width, bed.depth])
    # Every reset gives each root the manifest's strength, unless the reset names its own.
    expected = {env.roots.of_body[model.body(p.name).id]: p.root for p in manifest.weeds}
    assert env.roots.strength == expected
    weed = manifest.weeds[0]
    env.reset(options={"root_strength": {weed.name: 1.0}})
    assert env.roots.strength[env.roots.of_body[model.body(weed.name).id]] == 1.0
    env.reset()
    assert env.roots.strength == expected
    env.close()


def test_reset_seed_does_not_change_the_scene(tmp_path):
    """The layout is fixed when the environment is built; a reset seed cannot move it."""
    env = build_env(Manifest.sample(11), tmp_path, cameras=())
    env.reset(seed=1)
    first = env.data.qpos.copy()
    env.reset(seed=2)
    assert np.array_equal(env.data.qpos, first)
    env.close()


def test_a_sampled_scene_compiles_and_settles_untouched(tmp_path):
    manifest = Manifest.variants("mixed_clutter", 1)[0]
    env = build_env(manifest, tmp_path, cameras=(), control_hz=50)
    env.reset()
    model, data = env.model, env.data
    for _ in range(100):
        _, _, _, _, info = env.step(data.ctrl)
        assert not info["uprooted"]
    # Leaned stems and sloped beds must not leave a root loaded at rest.
    assert max(env.roots.load(data, eq) for eq in env.roots.eqs) < 0.1
    plants = {model.body(p.name).id for p in manifest.weeds + manifest.strawberries}
    for contact in data.contact[:data.ncon]:
        roots = {model.body_rootid[model.geom_bodyid[g]] for g in (contact.geom1, contact.geom2)}
        assert not (len(roots) == 2 and roots <= plants), "two plants are touching"
    env.close()


# IK against the generated beds

def test_the_ik_is_built_on_the_generated_beds(tmp_path):
    manifest = Manifest.variants("mixed_clutter", 1)[0]
    env = build_env(manifest, tmp_path, cameras=())
    env.reset()
    runner = Uprooter(env, manifest.weeds[0].name)
    assert runner.ik.scene == env.scene
    nominal = mujoco.MjModel.from_xml_path(str(SCENE))
    for bed in manifest.beds:
        geom = runner.ik.model.geom(bed.name)
        assert geom.pos == pytest.approx([bed.x, bed.y, bed.z])
        assert geom.size == pytest.approx([bed.length, bed.width, bed.depth])
        assert geom.quat == pytest.approx(bed.quat())
        assert geom.pos != pytest.approx(nominal.geom(bed.name).pos)
    env.close()


def test_the_ik_keeps_the_gripper_off_a_raised_bed(tmp_path):
    """A target 15 mm above the nominal soil is fine there, but on a bed raised 15 mm the
    pads would be in the soil: only an IK that knows the raised bed refuses it."""
    nominal = Manifest.nominal()
    raised = dataclasses.replace(
        nominal, beds=(dataclasses.replace(nominal.beds[0], z=nominal.beds[0].z + 0.015),
                       nominal.beds[1]))
    env = WeedExtractionEnv(cameras=())
    env.reset()
    env.data.qpos[env.model.joint("base_slide_x").qposadr[0]] = 0.2
    mujoco.mj_forward(env.model, env.data)
    target = pose([0.2, -0.5, nominal.beds[0].top(0.2, -0.5) + 0.015], [0.0, 0.0, -1.0])
    assert ArmIK().solve(env.data.qpos, target)[1] < 1e-3
    assert ArmIK(write_scene(raised, tmp_path)).solve(env.data.qpos, target)[1] > 5e-3
    env.close()


def test_an_ik_built_for_another_scene_is_refused(tmp_path):
    manifest = Manifest.sample(11)
    env = build_env(manifest, tmp_path, cameras=())
    env.reset()
    with pytest.raises(ValueError, match="IK was built for"):
        Uprooter(env, manifest.weeds[0].name, ik=ArmIK())
    env.close()


@pytest.mark.slow
def test_the_farthest_weeds_of_a_sampled_scene_can_be_extracted(tmp_path):
    """The whole path on a varied scene: seed to manifest to MJCF to environment, the
    IK built from that scene's beds, and the unchanged extraction sequence pulling the
    weed of each species planted farthest from the rail, and nothing else."""
    manifest = Manifest.variants("mixed_clutter", 1)[0]
    nominal = Manifest.nominal()
    assert all(bed != home for bed, home in zip(manifest.beds, nominal.beds))
    for species in Manifest.WEED_SPECIES:
        weed = max((p for p in manifest.weeds if p.species == species), key=lambda p: abs(p.y))
        env = build_env(manifest, tmp_path, cameras=(), control_hz=50)
        env.reset()
        runner, torn = Uprooter(env, weed.name), []
        while runner.step() is not None:
            torn += env.step(env.data.ctrl)[4]["uprooted"]
        env.close()
        assert torn == [weed.name], (weed, torn, runner.peak_load)


# Saved manifests

def test_a_manifest_survives_saving_and_loading(tmp_path):
    for m in (Manifest.nominal(), Manifest.sample(5), Manifest.variants("mixed_clutter", 1)[0]):
        assert Manifest.load(m.save(tmp_path / "manifest.json")) == m


def test_the_same_seed_writes_the_same_files(tmp_path):
    """Byte for byte, bar the folder each was written into."""
    for folder in ("a", "b"):
        write_scene(Manifest.sample(5), tmp_path / folder)
    for name in ("manifest.json", "scene.xml", "weeds.xml"):
        a = (tmp_path / "a" / name).read_text().replace(str(tmp_path / "a"), "")
        b = (tmp_path / "b" / name).read_text().replace(str(tmp_path / "b"), "")
        assert a == b, name


def test_every_catalogued_scene_is_saved_with_an_index(tmp_path):
    index = json.loads(write_catalogue(tmp_path).read_text())
    catalogue = Manifest.catalogue()
    assert len(index) == 5 * len(Manifest.FAMILIES)
    for entry in index:
        saved = Manifest.load(tmp_path / entry["manifest"])
        assert saved == catalogue[entry["family"]][entry["variant"]]
        assert (entry["seed"], entry["split"]) == (saved.seed, saved.split)


def test_the_saved_metadata_carries_what_the_plan_asks_for(tmp_path):
    m = Manifest.variants("mixed_clutter", 1)[0]
    build_env(m, tmp_path, cameras=()).close()
    saved = json.loads((tmp_path / "manifest.json").read_text())
    fields, meta = saved["manifest"], saved["metadata"]
    assert (fields["seed"], fields["family"], fields["variant"], fields["split"]) == (
        m.seed, "mixed_clutter", 0, "train")
    assert meta["weed_count"] == len(m.weeds) and meta["strawberry_count"] == len(m.strawberries)
    assert sum(meta["species"].values()) == len(m.weeds)
    for p in m.weeds:
        assert meta["root_break_force_n"][p.name] == pytest.approx(15.0 * p.root)
        assert meta["root_break_torque_nm"][p.name] == pytest.approx(0.07 * p.root)
        assert meta["crowding_m"][p.name] >= 0
    assert {tuple(p["targets"]) for p in meta["prompts"]} == {
        p.targets for p in Prompt.valid(m)}
    for p in fields["weeds"]:  # position, species and every physical parameter
        assert {"name", "species", "x", "y", "z", "yaw", "lean", "root", "stiffness",
                "damping", "soil_give"} <= set(p)


# Occlusion

def test_open_and_crowded_scenes_keep_to_their_level():
    seen = set()
    for m in SAMPLES:
        gaps = [Manifest._crowding(p, m.strawberries) for p in m.weeds]
        seen.add(m.occlusion)
        if m.occlusion == "open":
            assert min(gaps) >= Manifest.OPEN
        if m.occlusion == "crowded":
            assert sum(g <= Manifest.CROWDED for g in gaps) >= math.ceil(len(gaps) / 2)
    assert seen == set(Manifest.OCCLUSION)
    assert all(m.occlusion == "crowded" for m in Manifest.variants("mixed_clutter"))


def test_a_scene_that_misses_its_occlusion_level_is_flagged():
    crowded = next(m for m in SAMPLES if m.occlusion == "crowded")
    assert crowded.misses() == []
    assert dataclasses.replace(crowded, occlusion="open").misses() == [
        "a weed grows too close to the crop for an open scene"]
    opened = next(m for m in SAMPLES if m.occlusion == "open")
    assert dataclasses.replace(opened, occlusion="crowded").misses() == [
        "too few weeds grow close to the crop for a crowded scene"]


# Physics

def test_the_physics_is_built_as_the_manifest_says(tmp_path):
    m = Manifest.variants("mixed_clutter", 1)[0]
    model = mujoco.MjModel.from_xml_path(str(write_scene(m, tmp_path)))
    nominal = mujoco.MjModel.from_xml_path(str(SCENE))

    def stalk(model, weed):
        """The weed's six stalk joints, two per sprung segment."""
        segments = {model.body(f"{weed}_s{k}").id for k in (1, 2, 3)}
        return [j for j in range(model.njnt) if model.jnt_bodyid[j] in segments]

    joint = stalk(nominal, "shepherd_1")[0]
    stiffness = nominal.jnt_stiffness[joint]
    damping = nominal.dof_damping[nominal.jnt_dofadr[joint]]
    weld = nominal.eq_solref[nominal.eq("shepherd_1_root").id][0]
    for p in m.weeds:
        joints = stalk(model, p.name)
        assert len(joints) == 6
        assert model.jnt_stiffness[joints] == pytest.approx(stiffness * p.stiffness)
        assert model.dof_damping[model.jnt_dofadr[joints]] == pytest.approx(damping * p.damping)
        assert model.eq_solref[model.eq(f"{p.name}_root").id][0] == pytest.approx(weld * p.soil_give)
    for bed in m.beds:
        assert model.geom(bed.name).friction[0] == pytest.approx(bed.friction)


@pytest.mark.parametrize("extreme", ["soft", "firm"])
def test_the_physics_is_stable_at_both_ends_of_its_ranges(tmp_path, extreme):
    """The stalk stands and springs back, the root rests unloaded, and it still holds
    below its strength and gives way above it."""
    pick = min if extreme == "soft" else max
    nominal = Manifest.nominal()
    physics = {"stiffness": pick(Manifest.STALK_STIFFNESS), "damping": pick(Manifest.STALK_DAMPING),
               "soil_give": max(Manifest.SOIL_GIVE) if extreme == "soft" else min(Manifest.SOIL_GIVE)}
    m = dataclasses.replace(nominal, weeds=tuple(dataclasses.replace(p, **physics)
                                                 for p in nominal.weeds))
    env = build_env(m, tmp_path, cameras=())
    env.reset()
    model, data = env.model, env.data
    tip = model.body("sowthistle_1_s3").id
    mujoco.mj_forward(model, data)
    rest = data.xpos[tip].copy()
    mujoco.mj_step(model, data, nstep=3000)
    assert np.linalg.norm(data.xpos[tip] - rest) < 0.002
    assert max(env.roots.load(data, eq) for eq in env.roots.eqs) < 0.1
    data.xfrc_applied[tip, 0] = 0.15
    mujoco.mj_step(model, data, nstep=800)
    assert np.linalg.norm(data.xpos[tip] - rest) > 0.02
    data.xfrc_applied[:] = 0
    mujoco.mj_step(model, data, nstep=3000)
    assert np.linalg.norm(data.xpos[tip] - rest) < 0.002
    root = env.roots.of_body[model.body("shepherd_1").id]
    body = model.body("shepherd_1").id
    for fraction, gives in ((0.9, False), (1.1, True)):
        env.reset()
        gave_way = False
        for _ in range(1000):
            data.xfrc_applied[body, :3] = fraction * 15.0 * env.roots.up[root]
            mujoco.mj_step(model, data)
            gave_way |= "shepherd_1" in env.roots.snap(data)
        data.xfrc_applied[:] = 0
        assert gave_way is gives, fraction
    env.close()


# Walkway

def test_the_walkway_varies_over_existing_textures_and_never_matches_the_soil(tmp_path):
    for m in SAMPLES:
        assert m.path in Manifest.PATHS and m.path != m.soil
    assert {m.path for m in SAMPLES} == set(Manifest.PATHS)
    m = next(m for m in SAMPLES if m.path.endswith(".jpg"))
    root = ET.parse(write_scene(m, tmp_path)).getroot()
    texture = root.find("asset/texture[@name='grass']").get("file")
    assert texture == str(tmp_path / (Path(m.path).stem + ".png"))
    assert (tmp_path / (Path(m.path).stem + ".png")).exists()
