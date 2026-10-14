"""The browser UI: held-out environments with their pictures, prompt checking, a scene that a
bad prompt never touches, the demo's state machine acting in the scene the native viewer
shows, and the viewer and scene cleaned up on switching, closing and shutdown. Also
vla_runner, which the UI no longer uses: SmolVLA's real actions applied to a scene."""

import json
import os
import socket
import subprocess
import sys
import textwrap
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torchvision")
from weed_extraction.prompts import Prompt  # noqa: E402
from weed_extraction.scene import Manifest  # noqa: E402
from weed_extraction.ui import App, Live, PromptCheck, Scenes, Shared, Worker, parser  # noqa: E402


def test_only_held_out_scenes_are_offered():
    scenes = Scenes.held_out()
    expected = {f"evening_cool_{v}" for v in range(9)} | {
        f"{f}_4" for f in Manifest.FAMILIES if f != "evening_cool"}
    assert set(scenes) == expected and len(scenes) == 14
    for scene in scenes:
        family, variant = scene.rsplit("_", 1)
        assert Manifest._split(family, int(variant)) != "train"


def test_the_suggestions_are_the_24_trained_wordings_and_all_valid():
    trained = {w.format(species=s, side=b) for kind in Prompt.WORDINGS
               for w in Prompt.WORDINGS[kind] for s in Manifest.WEED_SPECIES
               for b in ("left", "right")}
    assert set(PromptCheck.SUGGESTIONS) == trained and len(PromptCheck.SUGGESTIONS) == 24
    for suggestion in PromptCheck.SUGGESTIONS:
        assert PromptCheck.check(suggestion) is not None


@pytest.mark.parametrize("prompt, species, side, every", [
    ("extract the sowthistle", "sowthistle", None, False),
    ("Remove the Shepherd weed.", "shepherd", None, False),
    ("uproot the shepherd weed on the left", "shepherd", "left", False),
    ("pull out the sowthistle weed in the right bed", "sowthistle", "right", False),
    ("get rid of the shepherd's purse in the right bed", "shepherd", "right", False),
    ("remove all shepherd weeds", "shepherd", None, True),
    ("clear all sowthistle in the left bed", "sowthistle", "left", True),
    ("from your standing position, remove all shepherd", "shepherd", None, True),
])
def test_a_valid_prompt_is_read_by_the_demo_grammar(prompt, species, side, every):
    command = PromptCheck.check(prompt)
    assert (command.species, command.side, command.every) == (species, side, every)


@pytest.mark.parametrize("prompt", [
    "make the robot dance", "move forward", "pick the strawberry", "turn the camera",
    "asdf qwerty", "", "   ", "shepherd",
    "remove the shepherd weed near the strawberries",
    "don't remove the shepherd weed", "do not remove the sowthistle weed",
    "never pull out the sowthistle weed", "remove the shepherd weed except the left one",
    "remove the sowthistle weed without touching anything", "remove the shepherd weed? no, isn't it",
    "remove the shepherd and sowthistle weeds",
    "remove the shepherd weed from the left and right beds", "remove weed 3",
])
def test_an_invalid_prompt_is_refused(prompt):
    assert PromptCheck.check(prompt) is None


def idle_worker():
    """A worker whose scene preparation is recorded instead of run."""
    worker = Worker(Shared(), parser().parse_args([]))
    worker.calls = []
    worker.prepare = lambda scene: worker.calls.append(("prepare", scene))
    return worker


def test_an_invalid_prompt_builds_no_scene_opens_no_viewer_and_moves_nothing():
    worker = idle_worker()
    for prompt in ("make the robot dance", "don't remove the shepherd weed",
                   "remove all strawberries"):
        worker.execute("evening_cool_0", prompt)
        assert worker.calls == [] and worker.demo is None and worker.viewer is None
        state = worker.shared.snapshot()
        assert state["status"] == "READY" and state["message"] == PromptCheck.INVALID
        assert state["step"] == 0 and state["frame_id"] == 0


def test_a_valid_prompt_needs_a_held_out_environment():
    worker = idle_worker()
    for scene in ("", "midday_dry_0", "../etc"):
        worker.execute(scene, "remove the shepherd weed")
        assert worker.calls == []
        assert worker.shared.snapshot()["message"] == "Select an environment."


def test_the_run_is_over_once_stop_is_pressed_or_the_viewer_closed():
    """demo.run checks is_running before every tick, so a STOP pressed while the run
    loads ends it before any step."""
    worker = Worker(Shared(), parser().parse_args(["--no-viewer"]))
    live = Live(worker)
    assert live.is_running()
    worker.stop.set()
    assert not live.is_running()
    worker.stop.clear()
    worker.viewer = types.SimpleNamespace(is_running=lambda: False)
    assert not live.is_running()


def get(url):
    with urllib.request.urlopen(url, timeout=10) as reply:
        return reply.status, reply.read()


def post(url, body):
    request = urllib.request.Request(url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as reply:
        return reply.status


def test_the_page_shows_the_gallery_and_suggestions_without_annotations():
    app = App(parser().parse_args(["--port", "0", "--no-browser", "--no-viewer"]))
    app.start()
    try:
        status, page = get(app.url)
        text = page.decode()
        assert status == 200
        assert all(s in text for s in Scenes.held_out()) and 'id="gallery"' in text
        assert all(s in text for s in PromptCheck.SUGGESTIONS)
        assert 'id="wrist"' in text and 'id="sim"' not in text and 'id="bed"' not in text
        for word in ("success", "fail", "recommend", "trained", "verified"):
            assert word not in text.lower()
        assert json.loads(get(app.url + "api/state")[1])["status"] == "IDLE"
        for path in ("frame/wrist.jpg", "thumb/midday_dry_0.jpg", "thumb/..%2Fx.jpg"):
            with pytest.raises(urllib.error.HTTPError):  # nothing yet, or not held out
                get(app.url + path)
        with pytest.raises(urllib.error.HTTPError):
            post(app.url + "api/load", {"scene": "evening_cool_0"})  # loading is on execute
    finally:
        app.close()
    assert not app.worker.is_alive()


@pytest.mark.slow
def test_a_thumbnail_is_the_scene_seen_from_its_overview_camera(tmp_path, monkeypatch):
    monkeypatch.setattr(Scenes, "THUMBS", tmp_path)
    tmp_path.mkdir(exist_ok=True)
    Scenes.thumbnail("low_light_4")
    from PIL import Image

    with Image.open(tmp_path / "low_light_4.jpg") as image:
        assert image.format == "JPEG" and image.size == (480, 360)
        assert np.asarray(image).std() > 5  # a rendered scene, not a blank frame


def detector():
    file = Path(__file__).resolve().parents[4] / "models" / "detector.pt"
    if not file.is_file() or file.stat().st_size < 10**6:
        pytest.skip("needs models/detector.pt from Git LFS")
    return file


def model_dir():
    model = Path(__file__).resolve().parents[4] / "models" / "smolvla_weeds"
    weights = model / "model.safetensors"
    if not weights.is_file() or weights.stat().st_size < 10**6:
        pytest.skip("needs models/smolvla_weeds from Git LFS")
    return model


@pytest.mark.slow
def test_two_chunks_of_real_smolvla_actions_are_applied_to_the_scene(tmp_path):
    pytest.importorskip("lerobot")
    from weed_extraction.vla_runner import VLARunner

    runner = VLARunner(model_dir())
    runner.load(Manifest.variants("low_light", 5)[4], tmp_path / "scene")
    runner.start("remove the shepherd weed", ["shepherd_1"])
    low, high = runner.env.model.actuator_ctrlrange.T
    for _ in range(51):  # 50 from the first chunk, then the first of the second
        applied = runner.step()
        assert applied.shape == (9,) and np.isfinite(applied).all()
        assert np.array_equal(applied, runner.env.data.ctrl)
        assert (applied >= low - 1e-9).all() and (applied <= high + 1e-9).all()
    assert runner.inferences == 2 and runner.steps == 51
    assert runner.time == pytest.approx(51 / runner.CONTROL_HZ)
    assert set(runner.frames()) == {"wrist", "bed"}
    runner.close()
    assert runner.env is None


def has_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(url, condition, seconds):
    end = time.time() + seconds
    while time.time() < end:
        state = json.loads(get(url + "api/state")[1])
        if condition(state):
            return state
        time.sleep(0.5)
    raise AssertionError(f"timed out; last state {state}")


@pytest.mark.slow
def test_the_ui_refuses_a_bad_prompt_then_runs_the_demo_in_two_environments():
    model = detector()
    port = free_port()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    flags = [] if has_display() else ["--no-viewer"]
    process = subprocess.Popen(
        [sys.executable, "-m", "weed_extraction.ui", "--no-browser", "--port", str(port),
         "--detector", str(model), *flags], env=env, cwd=model.parents[1],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}/"
    try:
        end = time.time() + 60
        while True:
            try:
                get(url + "api/state")
                break
            except OSError:
                assert time.time() < end and process.poll() is None
                time.sleep(0.5)

        assert post(url + "api/execute",
                    {"scene": "evening_cool_0", "prompt": "make the robot dance"}) == 202
        refused = wait_for(url, lambda s: not s["busy"], 30)
        assert refused["message"] == PromptCheck.INVALID
        assert refused["scene"] is None and not refused["viewer"]  # nothing was built
        assert refused["step"] == 0 and refused["frame_id"] == 0

        for scene, prompt in (("evening_cool_0", "remove the sowthistle weed"),
                              ("low_light_4", "uproot the shepherd weed on the left")):
            assert post(url + "api/execute", {"scene": scene, "prompt": prompt}) == 202
            running = wait_for(url, lambda s: s["status"] == "ERROR" or (
                s["scene"] == scene and s["step"] >= 2 and s["status"] == "EXECUTING"), 300)
            assert running["status"] == "EXECUTING", running["message"]
            assert running["scene"] == scene and len(running["action"]) == 9
            assert running["viewer"] == has_display()
            status, image = get(url + "frame/wrist.jpg")
            assert status == 200 and image[:2] == b"\xff\xd8"  # a JPEG
            assert post(url + "api/stop", {}) == 200
            stopped = wait_for(url, lambda s: not s["busy"], 120)
            assert stopped["status"] == "STOPPED"
    finally:
        process.terminate()
        assert process.wait(timeout=60) == 0  # SIGTERM closes viewer, scene and worker


@pytest.mark.slow
def test_the_viewer_is_reopened_replaced_and_closed_with_the_scene():
    if not has_display():
        pytest.skip("the native viewer needs a display")
    model = detector()
    script = textwrap.dedent(f"""
        import json, time
        import numpy as np
        from weed_extraction.ui import Shared, Worker, parser
        worker = Worker(Shared(), parser().parse_args(["--detector", {str(model)!r}]))
        worker.start()

        def run(scene, until):
            assert worker.shared.begin()
            worker.stop.clear()  # as the request handler does
            worker.commands.put(("execute", (scene, "remove the shepherd weed")))
            end = time.time() + 300
            while not until(worker.shared.snapshot()):
                assert time.time() < end, worker.shared.snapshot()
                time.sleep(0.2)

        def idle(state):
            return not state["busy"]

        def going(state):
            return state["status"] == "EXECUTING" and state["step"] >= 1

        out = {{}}
        run("evening_cool_0", going)
        demo, env, first = worker.demo, worker.demo.env, worker.viewer
        out["one_simulation"] = (first.m is env.model and first.d is env.data
                                 and worker.wrist.model is env.model)
        worker.stop.set()
        while not idle(worker.shared.snapshot()):
            time.sleep(0.2)
        shown = worker.shared.snapshot()
        out["shown_is_applied"] = (shown["step"] >= 1
                                   and np.array_equal(shown["action"], env.data.ctrl))
        run("evening_cool_0", going)
        first = worker.viewer
        first.close()  # as if the window were closed by hand
        while not idle(worker.shared.snapshot()):
            time.sleep(0.2)
        out["closed"] = worker.shared.snapshot()
        run("evening_cool_0", going)
        second = worker.viewer
        out["reopened"] = second is not first and second.is_running()
        out["same_scene_kept"] = worker.demo is demo
        worker.stop.set()
        while not idle(worker.shared.snapshot()):
            time.sleep(0.2)
        run("low_light_4", going)
        out["old_viewer_closed"] = not second.is_running()
        out["scene_rebuilt"] = worker.demo is not demo
        worker.stop.set()
        while not idle(worker.shared.snapshot()):
            time.sleep(0.2)
        last = worker.viewer
        worker.shutdown()
        out["shutdown"] = (not worker.is_alive() and worker.viewer is None
                           and worker.demo is None and worker.wrist is None
                           and not last.is_running())
        print("RESULT " + json.dumps(out, default=str))
    """)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(MUJOCO_GL="egl", PYGLFW_LIBRARY_VARIANT="x11")  # as ui.main() sets them
    done = subprocess.run([sys.executable, "-c", script], env=env, cwd=model.parents[1],
                          capture_output=True, text=True, timeout=900)
    line = next((l for l in done.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, done.stderr[-3000:]
    out = json.loads(line[len("RESULT "):])
    assert out["closed"]["status"] == "STOPPED"
    assert out["closed"]["message"] == "Viewer closed."
    for key in ("one_simulation", "shown_is_applied", "reopened", "same_scene_kept",
                "old_viewer_closed", "scene_rebuilt", "shutdown"):
        assert out[key] is True, (key, out)
