import dataclasses

import mujoco
import pytest

from weed_extraction.env import SCENE
from weed_extraction.prompts import Prompt
from weed_extraction.scene import Manifest


def scene(*weeds, strawberries=True):
    """The nominal scene with only `weeds`, given as (species, x, y)."""
    nominal = Manifest.nominal()
    template = {p.species: p for p in nominal.weeds}
    count = {}
    plants = []
    for species, x, y in weeds:
        count[species] = count.get(species, 0) + 1
        plants.append(dataclasses.replace(template[species], name=f"{species}_{count[species]}",
                                          x=x, y=y))
    return dataclasses.replace(nominal, weeds=tuple(plants),
                               strawberries=nominal.strawberries if strawberries else ())


def texts(manifest):
    return {p.text: p.targets for p in Prompt.valid(manifest)}


def test_start_is_the_home_position_of_the_base():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    home = model.key_qpos[model.key("home").id]
    assert home[model.joint("base_slide_x").qposadr[0]] == Prompt.START


def test_the_species_prompt_needs_exactly_one_of_that_species():
    prompts = texts(scene(("sowthistle", 0.0, -0.5), ("shepherd", 1.0, -0.5), ("shepherd", 2.0, 0.5)))
    assert prompts["remove the sowthistle weed"] == ("sowthistle_1",)
    assert "remove the shepherd weed" not in prompts  # two of them: which one?
    assert prompts["remove all shepherd weeds"] == ("shepherd_1", "shepherd_2")


def test_weeds_are_counted_along_the_rail_from_the_start():
    prompts = texts(scene(("shepherd", 2.0, 0.5), ("sowthistle", -1.0, -0.5), ("shepherd", 0.5, -0.5)))
    assert prompts["remove weed 1"] == ("sowthistle_1",)
    assert prompts["remove weed 2"] == ("shepherd_2",)
    assert prompts["remove weed 3"] == ("shepherd_1",)
    assert prompts["remove the nearest weed"] == ("sowthistle_1",)
    assert prompts["remove the nearest weed but do not touch the strawberries"] == ("sowthistle_1",)


def test_weeds_level_along_the_rail_cannot_be_counted():
    prompts = {p.kind for p in Prompt.valid(scene(("shepherd", 1.0, 0.5), ("shepherd", 1.02, -0.5)))}
    assert not prompts & {"instance", "nearest"}


def test_left_and_right_are_the_robots_facing_along_the_rail():
    prompts = texts(scene(("shepherd", 0.0, 0.5), ("shepherd", 1.0, -0.5), ("sowthistle", 2.0, -0.5)))
    assert prompts["remove one weed from the left bed"] == ("shepherd_1",)
    assert prompts["remove the weed on the left side of the bed"] == ("shepherd_1",)
    assert prompts["remove one weed from the right bed"] == ("shepherd_2", "sowthistle_1")
    assert "remove the weed on the right side of the bed" not in prompts
    assert prompts["remove all shepherd weeds from the right bed"] == ("shepherd_2",)


def test_the_front_of_the_bed_is_the_end_the_robot_starts_from():
    prompts = texts(scene(("shepherd", -1.0, 0.5), ("shepherd", 2.5, -0.5)))
    assert prompts["remove one shepherd weed near the front of the bed"] == ("shepherd_1",)


def test_crop_safe_prompts_need_a_crop():
    assert "crop_safe" in {p.kind for p in Prompt.valid(scene(("shepherd", 0.0, 0.5)))}
    assert "crop_safe" not in {p.kind for p in Prompt.valid(
        scene(("shepherd", 0.0, 0.5), strawberries=False))}


def test_a_scene_missing_a_bed_of_weeds_misses_that_prompt():
    assert Prompt.missing(scene(("shepherd", 0.0, 0.5), ("shepherd", 1.0, 0.5))) == ["right_bed"]


def test_every_catalogued_scene_answers_every_required_prompt():
    catalogue = [m for variants in Manifest.catalogue().values() for m in variants]
    kinds = set()
    for m in catalogue:
        assert Prompt.missing(m) == [], (m.family, m.variant)
        names = {p.name for p in m.weeds}
        for prompt in Prompt.valid(m):
            assert prompt.targets and set(prompt.targets) <= names
            kinds.add(prompt.kind)
    # The draw-dependent kinds turn up somewhere in the catalogue too.
    assert kinds >= set(Prompt.REQUIRED) | {"species_one", "side_one", "species_region",
                                            "species_area", "species_front"}


@pytest.mark.parametrize("text", ["remove all weeds", "remove all shepherd weeds"])
def test_prompts_that_ask_for_every_target_say_so(text):
    prompt = next(p for p in Prompt.valid(scene(("shepherd", 0.0, 0.5))) if p.text == text)
    assert prompt.every


def test_a_typed_prompt_means_the_nearest_weed_of_its_type():
    m = scene(("shepherd", 1.0, 0.5), ("shepherd", -1.0, -0.5), ("sowthistle", 0.0, 0.5))
    typed = {p.text: p.targets for p in Prompt.typed(m)}
    assert typed["remove the shepherd weed"] == ("shepherd_2",)
    assert typed["pull out the shepherd weed in the left bed"] == ("shepherd_1",)
    assert typed["get rid of the sowthistle weed"] == ("sowthistle_1",)
    assert not any("sowthistle" in text and "right" in text for text in typed)
    assert len(typed) == 4 * 5  # every wording of the five prompts the scene offers
