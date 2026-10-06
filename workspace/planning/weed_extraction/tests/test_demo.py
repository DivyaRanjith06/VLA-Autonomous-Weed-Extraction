"""The demo's command grammar, the "all" tally, and one "all" prompt run end to end."""

from pathlib import Path

import pytest

pytest.importorskip("torchvision")
from weed_extraction.demo import ActionLog, Command, Demo, outcome  # noqa: E402
from weed_extraction.scene import Manifest  # noqa: E402


@pytest.mark.parametrize("prompt, species, side, every", [
    ("remove the shepherd weed", "shepherd", None, False),
    ("remove the sowthistle weed", "sowthistle", None, False),
    ("remove the shepherd weed from the left bed", "shepherd", "left", False),
    ("remove the shepherd weed from the right bed", "shepherd", "right", False),
    ("remove the sowthistle weed from the left bed", "sowthistle", "left", False),
    ("remove the sowthistle weed from the right bed", "sowthistle", "right", False),
    ("remove all shepherd weeds", "shepherd", None, True),
    ("remove all sowthistle weeds", "sowthistle", None, True),
    ("remove all shepherd in the left bed", "shepherd", "left", True),
    ("remove all shepherd in the right bed", "shepherd", "right", True),
    ("remove all sowthistle in the left bed", "sowthistle", "left", True),
    ("remove all sowthistle in the right bed", "sowthistle", "right", True),
    ("remove all sowthistle weeds from the left bed", "sowthistle", "left", True),
    ("clear all shepherd weeds from the right bed", "shepherd", "right", True),
    ("get rid of every sowthistle in the left bed", "sowthistle", "left", True),
    ("pull out every shepherd weed on the right", "shepherd", "right", True),
    ("remove all sowthistle in the entire field", "sowthistle", None, True),
    ("remove all shepherd in the entire field", "shepherd", None, True),
    ("remove every sowthistle weed in the field", "sowthistle", None, True),
    ("remove every shepherd weed in the field", "shepherd", None, True),
    ("clear all sowthistle", "sowthistle", None, True),
    ("clear all shepherd", "shepherd", None, True),
    ("take the shepherd weed out of the bed", "shepherd", None, False),
    ("get rid of the sowthistle weed in the left bed", "sowthistle", "left", False),
])
def test_the_grammar_reads_type_bed_and_how_many(prompt, species, side, every):
    command = Command.parse(prompt)
    assert (command.species, command.side, command.every) == (species, side, every)
    assert not command.standing


@pytest.mark.parametrize("prompt, species", [
    ("from your standing position, remove all shepherd", "shepherd"),
    ("from your standing position, remove all sowthistle", "sowthistle"),
    ("from the standing position, remove all shepherd weeds", "shepherd"),
    ("from where you are standing, remove all sowthistle", "sowthistle"),
])
def test_the_standing_phrase_is_an_all_prompt_for_both_beds(prompt, species):
    assert Command.parse(prompt) == Command(species, None, True, True)


@pytest.mark.parametrize("prompt, message", [
    ("remove the strawberry", "strawberries are the crop"),
    ("remove all strawberries from the left bed", "strawberries are the crop"),
    ("remove the dandelion weed", "must name a weed type"),
    ("remove all weeds", "must name a weed type"),
    ("shepherd please", "must say what to do"),
    ("remove the shepherd and the sowthistle weeds", "one weed type"),
    ("remove all shepherd from the left and right beds", "one bed"),
    ("remove all shepherd in the left bed of the field", "either one bed or the whole field"),
])
def test_what_the_grammar_cannot_read_is_refused(prompt, message):
    with pytest.raises(ValueError, match=message):
        Command.parse(prompt)


def test_an_all_prompt_succeeds_only_when_every_requested_weed_is_in_the_bin():
    assert outcome(["shepherd_1", "shepherd_5"], {"shepherd_5"}) == (
        ["shepherd_5"], ["shepherd_1"], False)
    assert outcome(["shepherd_1", "shepherd_5"], {"shepherd_1", "shepherd_5"}) == (
        ["shepherd_1", "shepherd_5"], [], True)
    # A wrong weed in the bin is not a requested one, and nothing requested is no success.
    assert outcome(["shepherd_1"], {"sowthistle_2"}) == ([], ["shepherd_1"], False)
    assert outcome([], set()) == ([], [], False)


@pytest.mark.slow
def test_an_all_prompt_picks_more_than_one_weed_and_counts_only_the_binned(tmp_path):
    """midday_dry 0 has two shepherd weeds in the left bed."""
    detector = Path(__file__).resolve().parents[4] / "models" / "detector.pt"
    if not detector.is_file() or detector.stat().st_size < 10**6:
        pytest.skip("needs models/detector.pt from Git LFS")
    demo = Demo(detector, Manifest.variants("midday_dry", 1)[0], tmp_path / "scene")
    demo.command("remove all shepherd in the left bed")
    assert len(demo.requested) >= 2
    while demo.state != "done" and demo.env.data.time - demo.began < 600:
        demo.step()
    removed, remaining, success = outcome(demo.requested, set(demo.binned()))
    assert len(demo.handled) >= 2  # it went on after the first weed
    assert success == (not remaining)
    assert set(removed) | set(remaining) == set(demo.requested)
    assert success, (removed, remaining, demo.failed)
    demo.env.close()


@pytest.mark.slow
def test_the_trained_smolvla_answers_with_a_nine_value_action(tmp_path):
    pytest.importorskip("lerobot")
    model = Path(__file__).resolve().parents[4] / "models" / "smolvla_weeds"
    weights = model / "model.safetensors"
    if not weights.is_file() or weights.stat().st_size < 10**6:
        pytest.skip("needs models/smolvla_weeds from Git LFS")
    from weed_extraction.smolvla_probe import probe

    result = probe(model, "remove the shepherd weed", Manifest.variants("midday_dry", 1)[0],
                   tmp_path / "scene")
    assert result["state_shape"] == [28]
    assert result["action_shape"] == [9]


def test_an_action_prints_as_time_step_state_nine_values_and_phase():
    record = {"step": 1, "simulation_time": 0.08, "source": "scripted_expert", "dim": 9,
              "action": [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 255.0, -2.0],
              "prompt": "remove the shepherd weed", "target": None, "state": "back up",
              "phase": None}
    lines = ActionLog.line(record, -1.7, 0.08).splitlines()
    assert lines[0] == "[ACTION] t=0.080s step=000001"
    assert lines[1] == "  state=[mode=back_up, target=none, base_x=-1.700 m, gripper_width=80.0 mm]"
    assert lines[2] == "  action=[0.000, 0.000, 0.000, -1.571, 0.000, 1.571, -0.785, 255.0, -2.000]"
    assert lines[3] == "  phase=back_up"
    assert "source" not in "".join(lines) and "/" not in "".join(lines)
    picking = {**record, "state": "pick", "phase": "close in", "target": "shepherd_5"}
    assert ActionLog.line(picking, 0.19, 0.0019).splitlines()[3] == "  phase=close_in"
