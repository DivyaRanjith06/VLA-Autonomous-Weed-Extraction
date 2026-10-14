"""Browser UI for the demo: pick a held-out environment from its picture, type a prompt, and
watch the demo's state machine (demo.Demo, picking with uproot.Uprooter) act in the native
MuJoCo viewer.

    python -m weed_extraction.ui [--detector models/detector.pt] [--port 8765]

A standard-library HTTP server serves one page. One worker thread owns the demo, its scene
and the native viewer, and runs each prompt through demo.run, as demo.sh does; the request
handlers only read a lock-protected snapshot (the wrist image, the applied action, step,
time, status) and put commands on the worker's queue. The wrist camera is rendered
off-screen (EGL) from the same scene the viewer shows, so the worker can render outside the
main thread; main() makes sure of that. No trained policy is used.
"""

import argparse
import contextlib
import io
import json
import multiprocessing
import os
import queue
import re
import signal
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

import mujoco

from weed_extraction.demo import ActionLog, Command, Demo, run
from weed_extraction.prompts import Prompt
from weed_extraction.scene import Manifest, build_env


class PromptCheck:
    """Which prompts may reach the demo."""

    INVALID = "Invalid prompt. Please provide a valid weed-extraction prompt."
    NEGATIONS = ("not", "don't", "dont", "never", "except", "without")
    # The autocomplete list: every wording the prompts were trained on, for both weed types,
    # alone and with each bed. Shown as plain suggestions; typing is never limited to them.
    SUGGESTIONS = tuple(dict.fromkeys(
        wording.format(species=species, side=side) for kind in ("type", "type_side")
        for wording in Prompt.WORDINGS[kind] for species in Manifest.WEED_SPECIES
        for side in ("left", "right")))

    @classmethod
    def check(cls, text):
        """The demo's reading of a valid prompt (a demo.Command), else None. The demo's
        grammar matches keywords, so negations and strawberries are refused here first."""
        lowered = text.lower()
        words = re.findall(r"[a-z']+", lowered)
        if (not words or "strawberr" in lowered
                or any(w in cls.NEGATIONS or w.endswith("n't") for w in words)):
            return None
        try:
            return Command.parse(text)
        except ValueError:
            return None


class Scenes:
    """The scenes kept out of training: the held-out family and the held-out variant of
    every other family, each with a picture from its own overview camera."""

    VARIANTS = 9  # variants per family in the catalogue the detector and dataset v2 use
    THUMBS = Path("outputs/ui/thumbs")

    @classmethod
    def held_out(cls):
        return [f"{family}_{variant}" for family in Manifest.FAMILIES
                for variant in range(cls.VARIANTS)
                if Manifest._split(family, variant) != "train"]

    @staticmethod
    def manifest(scene):
        family, variant = scene.rsplit("_", 1)
        return Manifest.variants(family, int(variant) + 1)[int(variant)]

    @classmethod
    def thumbnail(cls, scene):
        """Render `scene` from its overview camera, robot at home, to THUMBS/<scene>.jpg."""
        env = build_env(cls.manifest(scene), (cls.THUMBS / "scenes" / scene).resolve(),
                        cameras=("rgb_camera",), width=480, height=360)
        obs, _ = env.reset()
        env.close()
        partial = cls.THUMBS / f"{scene}.part.jpg"
        Image.fromarray(obs["rgb_camera"]).save(partial, "JPEG", quality=88)
        partial.replace(cls.THUMBS / f"{scene}.jpg")

    @classmethod
    def make_thumbnails(cls):
        """Every missing picture, each scene in its own process: a compiled scene is never
        freed in-process."""
        missing = [s for s in cls.held_out() if not (cls.THUMBS / f"{s}.jpg").is_file()]
        if missing:
            cls.THUMBS.mkdir(parents=True, exist_ok=True)
            with multiprocessing.get_context("spawn").Pool(2, maxtasksperchild=1) as pool:
                pool.map(cls.thumbnail, missing, chunksize=1)


class Shared:
    """What the page shows, written by the worker and read by the request handlers."""

    BUSY = ("LOADING", "VALIDATING", "EXECUTING")

    def __init__(self):
        self.lock = threading.Lock()
        self.state = {"status": "IDLE", "message": "", "scene": None, "prompt": "",
                      "step": 0, "time": 0.0, "action": None, "frame_id": 0,
                      "viewer": False}
        self.images = {}
        self.busy = False

    def update(self, **fields):
        with self.lock:
            self.state.update(fields)

    def snapshot(self):
        with self.lock:
            return dict(self.state, busy=self.busy)

    def begin(self):
        """Claim the worker for one command; False if it is already working."""
        with self.lock:
            if self.busy:
                return False
            self.busy = True
            return True

    def finish(self):
        with self.lock:
            self.busy = False

    def set_images(self, arrays):
        encoded = {}
        for name, array in arrays.items():
            buffer = io.BytesIO()
            Image.fromarray(array).save(buffer, "JPEG", quality=85)
            encoded[name] = buffer.getvalue()
        with self.lock:
            self.images = encoded
            self.state["frame_id"] += 1

    def image(self, name):
        with self.lock:
            return self.images.get(name)


class Worker(threading.Thread):
    """The one thread that touches MuJoCo, the native viewer and the demo."""

    WRIST = "d435i_color"
    WRIST_SIZE = 384  # pixels a side, as the dataset stores the wrist camera

    def __init__(self, shared, args):
        super().__init__(daemon=True)
        self.shared, self.args = shared, args
        self.commands = queue.Queue()
        self.stop = threading.Event()  # ends the current run; cleared by the next
        self.quitting = threading.Event()  # set once by shutdown; never cleared
        self.demo = self.wrist = self.log = self.viewer = self.viewer_thread = self.scene = None

    def run(self):
        while True:
            name, value = self.commands.get()
            if name == "quit":
                self.close_scene()
                if self.log is not None:
                    self.log.close()
                return
            try:
                getattr(self, name)(*value)
            except Exception as error:  # a scene, render or detector failure ends the command
                traceback.print_exc()
                self.shared.update(status="ERROR", message=f"{type(error).__name__}: {error}")
            finally:
                self.shared.finish()

    def shutdown(self, timeout=120):
        """Stop any run, close the viewer and the scene, and end the thread. A run stops
        between ticks. Returns whether the thread has ended."""
        self.quitting.set()
        self.stop.set()
        self.commands.put(("quit", ()))
        if self.is_alive():
            self.join(timeout)
        return not self.is_alive()

    def prepare(self, scene):
        """The demo for `scene` (built only when the scene changes), and the viewer."""
        if self.log is None:
            self.log = ActionLog(Path("outputs/demo_logs") / time.strftime("%Y%m%d_%H%M%S"),
                                 self.args.log_first, self.args.log_last, self.args.log_every)
        if scene != self.scene:
            self.close_scene()
            self.shared.update(status="LOADING", scene=None)
            self.demo = Demo(self.args.detector, Scenes.manifest(scene),
                             (Path("outputs/ui") / scene).resolve())
            self.demo.log = self.log
            self.wrist = mujoco.Renderer(self.demo.env.model, self.WRIST_SIZE, self.WRIST_SIZE)
            self.scene = scene
            self.shared.update(scene=scene)
            print(f"Environment loaded: {scene}", flush=True)
        if not self.args.no_viewer and (self.viewer is None or not self.viewer.is_running()):
            from mujoco import viewer  # needs a display

            self.close_viewer()
            before = set(threading.enumerate())
            self.viewer = viewer.launch_passive(self.demo.env.model, self.demo.env.data)
            # Python names a thread after its target; the viewer renders in _launch_internal.
            self.viewer_thread = next((t for t in set(threading.enumerate()) - before
                                       if "_launch_internal" in t.name), None)
        self.shared.update(viewer=self.viewer is not None)

    def close_viewer(self):
        """Close the viewer and wait for its render thread to end: a viewer still shutting
        down while the next scene compiles or another viewer opens never finishes, and
        then blocks GLFW's cleanup at exit."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.viewer_thread is not None:
            self.viewer_thread.join(10)
            self.viewer_thread = None
        self.shared.update(viewer=False)

    def close_scene(self):
        self.close_viewer()
        if self.wrist is not None:
            self.wrist.close()
            self.wrist = None
        if self.demo is not None:
            self.demo.env.close()
            self.demo = None
        self.scene = None

    def viewer_closed(self):
        return self.viewer is not None and not self.viewer.is_running()

    def restart(self):
        """Put the scene back as a new Demo leaves it: reset, then the demo's eased roots."""
        demo = self.demo
        demo.obs, _ = demo.env.reset()
        for eq in demo.env.roots.strength:  # after reset, which sets them
            demo.env.roots.strength[eq] *= Demo.ROOT_EASE

    def publish(self, step):
        """The wrist camera and the applied 9-D action, from the scene the demo steps."""
        data = self.demo.env.data
        self.wrist.update_scene(data, self.WRIST)
        self.shared.set_images({"wrist": self.wrist.render()})
        self.shared.update(step=step, time=float(data.time),
                           action=[float(v) for v in data.ctrl] if step else None)

    def execute(self, scene, prompt):
        if self.quitting.is_set():
            return
        self.shared.update(status="VALIDATING", message="", prompt=prompt)
        if PromptCheck.check(prompt) is None:  # nothing touches a scene
            self.shared.update(status="READY", message=PromptCheck.INVALID)
            return
        if scene not in Scenes.held_out():
            self.shared.update(status="READY", message="Select an environment.")
            return
        self.prepare(scene)
        self.restart()  # every prompt starts from the scene's start
        self.demo.command(prompt)
        if self.viewer is not None:
            self.viewer.sync()
        self.publish(0)
        self.shared.update(status="EXECUTING")
        print(f'Execution started: "{prompt}"', flush=True)
        # The demo's own terminal report (states, map, actions, summary) is kept off the
        # UI's terminal; the action log file is still written. Errors go to stderr as usual.
        report = io.StringIO()
        with contextlib.redirect_stdout(report):
            run(self.demo, self.args.seconds, Live(self))
        if self.stop.is_set() or self.quitting.is_set():
            status, message = "STOPPED", ""
        elif self.viewer_closed():
            status, message = "STOPPED", "Viewer closed."
        elif self.demo.state == "done":
            status, message = "FINISHED", ""
        else:
            status, message = "STOPPED", "Time limit reached."
        if status == "FINISHED":
            result = next((line[len("RESULT: "):] for line in report.getvalue().splitlines()
                           if line.startswith("RESULT: ")), "")
            print(f"Execution completed: {result}", flush=True)
        else:
            print(f"Execution stopped{': ' + message.rstrip('.').lower() if message else ''}",
                  flush=True)
        self.shared.update(status=status, message=message,
                           viewer=not self.viewer_closed() and self.viewer is not None)


class Live:
    """What demo.run is handed as its viewer: it syncs the native viewer (if any) and the
    page after every tick, and reports the run over on STOP or when the window is closed."""

    def __init__(self, worker):
        self.worker, self.steps = worker, 0

    def is_running(self):
        worker = self.worker
        return not (worker.stop.is_set() or worker.quitting.is_set() or worker.viewer_closed())

    def sync(self):
        if self.worker.viewer is not None:
            self.worker.viewer.sync()
        self.steps += 1
        self.worker.publish(self.steps)


class Handler(BaseHTTPRequestHandler):
    PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weed Extraction</title>
<style>
:root { --bg:#f6f7f5; --panel:#ffffff; --ink:#1d2421; --muted:#5f6b65; --line:#d9ded9;
        --accent:#2f6b4f; --accent-ink:#ffffff; --warn:#a33a2a; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141917; --panel:#1d2421; --ink:#e7ece9; --muted:#9aa8a1; --line:#2f3934;
          --accent:#5aa77f; --accent-ink:#0f1412; --warn:#e2806f; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.4 system-ui,sans-serif; }
header { padding:14px 20px; border-bottom:1px solid var(--line); font-weight:650;
         letter-spacing:.04em; }
main { display:grid; gap:14px; padding:14px 20px; grid-template-columns:1fr 1fr;
       max-width:1400px; margin:0 auto; }
section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }
.wide { grid-column:1/-1; }
h2 { margin:0 0 8px; font-size:12px; letter-spacing:.08em; color:var(--muted); font-weight:600; }
.row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
input, button { font:inherit; color:var(--ink); background:var(--panel);
         border:1px solid var(--line); border-radius:6px; padding:8px 10px; }
button { background:var(--accent); color:var(--accent-ink); border-color:var(--accent);
         cursor:pointer; font-weight:600; }
button.secondary { background:var(--panel); color:var(--ink); border-color:var(--line); }
button:disabled { opacity:.45; cursor:default; }
#gallery { display:grid; gap:10px; grid-template-columns:repeat(auto-fill, minmax(170px, 1fr)); }
.card { background:var(--panel); color:var(--ink); border:2px solid var(--line); padding:6px;
        text-align:center; font-weight:500; }
.card img { width:100%; aspect-ratio:4/3; object-fit:cover; border-radius:4px; display:block;
            background:var(--bg); }
.card span { display:block; margin-top:6px; font-size:13px; }
.card.on { border-color:var(--accent); box-shadow:0 0 0 2px var(--accent); }
.view { width:100%; aspect-ratio:1/1; background:#000; border-radius:6px; object-fit:contain;
        display:block; }
.cam { position:relative; }
.cam .error { position:absolute; inset:0; display:none; align-items:center;
              justify-content:center; color:#ddd; font-size:13px; }
.prompt { position:relative; flex:1; min-width:260px; }
.prompt input { width:100%; }
#suggest { position:absolute; left:0; right:0; top:100%; margin:2px 0 0; padding:4px 0;
           list-style:none; background:var(--panel); border:1px solid var(--line);
           border-radius:6px; display:none; z-index:5; max-height:260px; overflow:auto; }
#suggest li { padding:6px 10px; cursor:pointer; }
#suggest li.on, #suggest li:hover { background:var(--bg); }
table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
td { padding:5px 6px; border-bottom:1px solid var(--line); }
td.v { text-align:right; font-family:ui-monospace,monospace; }
#status { font-weight:700; font-size:18px; }
#message { color:var(--warn); min-height:1.4em; }
.meta { color:var(--muted); font-size:13px; }
@media (max-width:900px) { main { grid-template-columns:1fr; } }
</style></head><body>
<header>AUTONOMOUS WEED EXTRACTION</header>
<main>
 <section class="wide">
  <h2>1 &middot; ENVIRONMENT</h2>
  <div id="gallery"></div>
 </section>
 <section class="wide">
  <h2>2 &middot; PROMPT</h2>
  <div class="row">
   <div class="prompt"><input id="prompt" autocomplete="off" spellcheck="false"
        placeholder="Type a prompt" aria-label="prompt">
    <ul id="suggest" role="listbox"></ul></div>
   <button id="execute">EXECUTE</button>
   <button id="stop" class="secondary">STOP</button>
  </div>
 </section>
 <section>
  <h2>WRIST CAMERA</h2>
  <div class="cam"><img id="wrist" class="view" alt="wrist camera">
   <div class="error" id="wrist-error">camera unavailable</div></div>
 </section>
 <section>
  <h2>LIVE ACTION</h2>
  <table id="action"></table>
  <h2 style="margin-top:14px">STATUS</h2>
  <div class="row"><span id="status">IDLE</span><span class="meta" id="clock"></span></div>
  <div id="message"></div>
 </section>
</main>
<script>
const SCENES = __SCENES__;
const SUGGESTIONS = __SUGGESTIONS__;
const NAMES = ["q1","q2","q3","q4","q5","q6","q7","gripper","base_x"];
const BUSY = ["LOADING","VALIDATING","EXECUTING"];
const el = id => document.getElementById(id);
const pretty = s => s.replace(/_(\\d+)$/, " \\u00b7 variant $1").replace(/_/g, " ");
let selected = null;
for (const s of SCENES) {
  const card = document.createElement("button"); card.className = "card"; card.type = "button";
  const img = document.createElement("img"); img.alt = pretty(s);
  img.onerror = () => setTimeout(() => img.src = "/thumb/" + s + ".jpg?r=" + Date.now(), 2000);
  img.src = "/thumb/" + s + ".jpg";
  const name = document.createElement("span"); name.textContent = pretty(s);
  card.append(img, name);
  card.onclick = () => { selected = s;
    for (const c of el("gallery").children) c.classList.toggle("on", c === card); };
  el("gallery").append(card);
}
el("action").innerHTML = NAMES.map(n => "<tr><td>" + n + "</td><td class='v' id='a-" + n +
  "'>\\u2014</td></tr>").join("");
function post(path, body) {
  return fetch(path, {method:"POST", headers:{"Content-Type":"application/json"},
                      body: JSON.stringify(body || {})}).then(r => r.json())
    .then(r => { if (r.message) el("message").textContent = r.message; });
}
el("execute").onclick = () => { hide();
  if (!selected) { el("message").textContent = "Select an environment."; return; }
  post("/api/execute", {scene: selected, prompt: el("prompt").value}); };
el("stop").onclick = () => post("/api/stop");
let shown = [], cursor = -1;
function matches(text) {
  const tokens = text.toLowerCase().split(/\\s+/).filter(Boolean);
  if (!tokens.length) return [];
  return SUGGESTIONS.filter(s => { const words = s.split(" ");
    return tokens.every((t, i) => i === tokens.length - 1
      ? words.some(w => w.startsWith(t)) : words.includes(t) || s.includes(t)); });
}
function render() {
  const list = el("suggest"); list.innerHTML = "";
  shown.forEach((s, i) => { const li = document.createElement("li"); li.textContent = s;
    li.setAttribute("role", "option"); if (i === cursor) li.className = "on";
    li.onmousedown = e => { e.preventDefault(); pick(s); }; list.append(li); });
  list.style.display = shown.length ? "block" : "none";
}
function hide() { shown = []; cursor = -1; render(); }
function pick(s) { el("prompt").value = s; hide(); el("prompt").focus(); }
el("prompt").oninput = () => { shown = matches(el("prompt").value); cursor = -1; render(); };
el("prompt").onblur = hide;
el("prompt").onkeydown = e => {
  if (e.key === "ArrowDown" && shown.length) { cursor = (cursor + 1) % shown.length; render(); e.preventDefault(); }
  else if (e.key === "ArrowUp" && shown.length) { cursor = (cursor - 1 + shown.length) % shown.length; render(); e.preventDefault(); }
  else if (e.key === "Escape") hide();
  else if (e.key === "Enter") { e.preventDefault();
    if (cursor >= 0) pick(shown[cursor]); else el("execute").click(); }
};
let frame = -1;
async function poll() {
  try {
    const s = await (await fetch("/api/state")).json();
    el("status").textContent = s.status;
    el("message").textContent = s.message;
    el("clock").textContent = "step " + s.step + " \\u00b7 t = " + s.time.toFixed(2) + " s" +
      (s.scene ? " \\u00b7 " + pretty(s.scene) : "");
    NAMES.forEach((n, i) => el("a-" + n).textContent =
      s.action ? s.action[i].toFixed(i === 7 ? 1 : 3) : "\\u2014");
    const busy = s.busy || BUSY.includes(s.status);
    el("execute").disabled = busy; el("stop").disabled = !busy;
    if (s.frame_id !== frame && s.frame_id > 0) { frame = s.frame_id;
      const img = el("wrist");
      img.onerror = () => el("wrist-error").style.display = "flex";
      img.onload = () => el("wrist-error").style.display = "none";
      img.src = "/frame/wrist.jpg?f=" + frame; }
  } catch (e) { el("status").textContent = "ERROR"; el("message").textContent = "UI server unreachable"; }
  setTimeout(poll, 200);
}
poll();
</script></body></html>
"""

    def do_GET(self):
        app = self.server.app
        if self.path == "/":
            page = (self.PAGE.replace("__SCENES__", json.dumps(Scenes.held_out()))
                    .replace("__SUGGESTIONS__", json.dumps(PromptCheck.SUGGESTIONS)))
            return self.reply(200, page.encode(), "text/html; charset=utf-8")
        if self.path == "/api/state":
            return self.json(200, app.shared.snapshot())
        if self.path == "/api/scenes":
            return self.json(200, Scenes.held_out())
        if re.fullmatch(r"/frame/wrist\.jpg(\?.*)?", self.path):
            image = app.shared.image("wrist")
            if image is None:
                return self.reply(404, b"no image", "text/plain")
            return self.reply(200, image, "image/jpeg")
        match = re.fullmatch(r"/thumb/([a-z_]+_\d)\.jpg(\?.*)?", self.path)
        if match and match.group(1) in Scenes.held_out():
            thumb = Scenes.THUMBS / f"{match.group(1)}.jpg"
            if thumb.is_file():
                return self.reply(200, thumb.read_bytes(), "image/jpeg")
        self.reply(404, b"not found", "text/plain")

    def do_POST(self):
        app = self.server.app
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self.json(400, {"message": "bad request"})
        if self.path == "/api/stop":
            app.worker.stop.set()
            return self.json(200, {})
        if self.path != "/api/execute":
            return self.json(404, {"message": "not found"})
        if not app.shared.begin():
            return self.json(409, {"message": "Busy: wait for the current run, or STOP it."})
        app.worker.stop.clear()  # here, so a STOP pressed while this run loads still counts
        app.worker.commands.put(("execute", (str(body.get("scene", "")),
                                             str(body.get("prompt", "")))))
        self.json(202, {})

    def json(self, code, value):
        self.reply(code, json.dumps(value).encode(), "application/json")

    def reply(self, code, body, kind):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # the terminal is for the action log


class App:
    """The worker, the shared state and the HTTP server, wired together."""

    def __init__(self, args):
        self.shared = Shared()
        self.worker = Worker(self.shared, args)
        self.server = ThreadingHTTPServer((args.host, args.port), Handler)
        self.server.app = self
        self.server.daemon_threads = True

    @property
    def url(self):
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self):
        self.worker.start()
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        """Stop serving, then stop the worker, which closes the viewer and the scene.
        Returns whether the worker has ended."""
        self.server.shutdown()
        self.server.server_close()
        return self.worker.shutdown()


def parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--detector", type=Path, default=Path("models/detector.pt"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="0 picks a free port")
    parser.add_argument("--seconds", type=float,
                        help="simulated time allowed per prompt; 90 by default, 600 for an "
                             '"all" prompt, as in the demo')
    parser.add_argument("--log-first", type=int, default=5)
    parser.add_argument("--log-last", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    parser.add_argument("--no-viewer", action="store_true",
                        help="run without the native MuJoCo viewer (no display)")
    return parser


def main():
    # The renderer backend and the GLFW build are fixed when mujoco and glfw first load, which
    # importing this package already does; so they are set for a fresh process. EGL lets the
    # worker render the wrist camera outside the main thread without touching GLFW, and the
    # viewer uses GLFW's X11 build: on the Wayland build a closed viewer can hang GLFW.
    wanted = {"MUJOCO_GL": "egl"}
    if os.environ.get("DISPLAY"):
        wanted["PYGLFW_LIBRARY_VARIANT"] = "x11"
    if any(os.environ.get(key) != value for key, value in wanted.items()):
        os.execve(sys.executable, [sys.executable, "-m", "weed_extraction.ui", *sys.argv[1:]],
                  {**os.environ, **wanted})
    args = parser().parse_args()
    app = App(args)
    app.start()
    threading.Thread(target=Scenes.make_thumbnails, daemon=True).start()
    print(f"UI server started at {app.url}  (Ctrl+C to quit)", flush=True)
    if not args.no_browser:
        webbrowser.open(app.url)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        if not app.close():
            # The worker is still inside MuJoCo or torch; finalising the interpreter under
            # it would crash, so leave without running exit handlers.
            os._exit(1)


if __name__ == "__main__":
    main()
