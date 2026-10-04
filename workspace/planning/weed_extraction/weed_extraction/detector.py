"""The weed detector: finds every plant in a bed camera picture, names its type and
places it on the bed.

    python -m weed_extraction.detector render <data_dir>
    python -m weed_extraction.detector train <data_dir> <detector.pt>

`render` drives the base along the rail in every catalogue scene and saves the bed
camera's picture at each stop, with a box around each plant. The simulator draws the
boxes: it renders the same view with every pixel tagged by the body it belongs to.
`train` fits a small Faster R-CNN to the training scenes and checks it on the held-out
ones.

A box gives a direction from the camera, not a distance. The plant is placed where that
direction meets a level plane at the height the middle of such a plant usually sits,
which `train` measures per type from the rendered boxes and saves with the detector.
"""

import argparse
import multiprocessing
from pathlib import Path

import mujoco
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from weed_extraction.scene import Manifest, build_env


class WeedDetector:
    CAMERA = "bed_camera"
    SIZE = 384  # pixels a side, as the policy's dataset stores the bed camera
    # What a box can be. A plant's body is named <type>_<number>.
    LABELS = ("sowthistle", "shepherd", "strawberry")

    def __init__(self, path, device="cuda"):
        saved = torch.load(path, map_location=device)
        self.heights = saved["heights"]
        self.device = device
        self.net = self.network()
        self.net.load_state_dict(saved["net"])
        self.net.to(device).eval()

    @classmethod
    def network(cls):
        net = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        inputs = net.roi_heads.box_predictor.cls_score.in_features
        net.roi_heads.box_predictor = FastRCNNPredictor(inputs, len(cls.LABELS) + 1)
        return net

    def detect(self, image, min_score=0.7):
        """Every plant in an RGB uint8 picture, as (type, score, box); a box is
        (left, top, right, bottom) in pixels."""
        with torch.inference_mode():
            found = self.net([as_tensor(image).to(self.device)])[0]
        return [(self.LABELS[int(label) - 1], float(score), box.cpu().numpy())
                for label, score, box in zip(found["labels"], found["scores"], found["boxes"])
                if score >= min_score]

    def locate(self, model, data, label, box):
        """Where on the bed, as world (x, y), the plant in `box` stands."""
        cam = model.camera(self.CAMERA).id
        return place(data.cam_xpos[cam], data.cam_xmat[cam], model.cam_fovy[cam], box,
                     self.heights[label])


def as_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255


def place(cam_pos, cam_mat, fovy, box, height):
    """The world (x, y) where the ray through the middle of `box` meets z = `height`.
    A MuJoCo camera looks down its own -z, with x to the right and y up."""
    half = WeedDetector.SIZE / 2
    focal = half / np.tan(np.radians(fovy) / 2)
    u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    ray = np.asarray(cam_mat).reshape(3, 3) @ [(u - half) / focal, -(v - half) / focal, -1.0]
    return (np.asarray(cam_pos) + ray * (height - cam_pos[2]) / ray[2])[:2]


def render_scene(family, variant, out_dir):
    """Save the bed camera's view from stops along the rail of one catalogue scene, with
    each plant's box, type and true position."""
    manifest = Manifest.variants(family, variant + 1)[variant]
    size = WeedDetector.SIZE
    env = build_env(manifest, Path(out_dir) / "scenes" / f"{family}_{variant}",
                    cameras=(), width=size, height=size)
    env.reset()
    model, data = env.model, env.data
    renderer = mujoco.Renderer(model, size, size)
    cam = model.camera(WeedDetector.CAMERA).id
    base = model.joint("base_slide_x").qposadr[0]
    plants = {model.body(p.name).id: p for p in (*manifest.weeds, *manifest.strawberries)}
    rng = np.random.default_rng([manifest.seed, 7])
    records = {"image": [], "cam_pos": [], "cam_mat": [], "boxes": [], "labels": [], "xy": []}
    # The rail runs from -2 m; the last weed of any scene stands near 3.3 m and the
    # camera sees from 0.2 m ahead of the base.
    for x in np.arange(-2.0, 3.3, 0.2) + rng.uniform(0, 0.2, 27):
        data.qpos[base] = x
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        image = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam)
        geoms = renderer.render()[..., 0].copy()
        renderer.disable_segmentation_rendering()
        roots = np.where(geoms >= 0, model.body_rootid[model.geom_bodyid[geoms]], -1)
        boxes, labels, xy = [], [], []
        for body, plant in plants.items():
            rows, cols = np.nonzero(roots == body)
            if len(rows) < 20:  # out of view, or too little of it to name
                continue
            boxes.append([cols.min(), rows.min(), cols.max() + 1, rows.max() + 1])
            labels.append(WeedDetector.LABELS.index(plant.name.rsplit("_", 1)[0]) + 1)
            xy.append([plant.x, plant.y])
        records["image"].append(image)
        records["cam_pos"].append(data.cam_xpos[cam].copy())
        records["cam_mat"].append(data.cam_xmat[cam].copy())
        records["boxes"].append(np.array(boxes, np.float32).reshape(-1, 4))
        records["labels"].append(np.array(labels, np.int64))
        records["xy"].append(np.array(xy).reshape(-1, 2))
    renderer.close()
    np.savez_compressed(Path(out_dir) / f"{family}_{variant}_{manifest.split}.npz",
                        fovy=model.cam_fovy[cam], image=np.stack(records["image"]),
                        cam_pos=np.stack(records["cam_pos"]), cam_mat=np.stack(records["cam_mat"]),
                        count=[len(b) for b in records["boxes"]],
                        boxes=np.concatenate(records["boxes"]),
                        labels=np.concatenate(records["labels"]), xy=np.concatenate(records["xy"]))


def render(out_dir, variants):
    """Every family's first `variants` scenes, each in its own process: a compiled scene
    is never freed in-process."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(family, variant, out_dir) for family in Manifest.FAMILIES
            for variant in range(variants)]
    with multiprocessing.get_context("spawn").Pool(4, maxtasksperchild=1) as pool:
        pool.starmap(render_scene, jobs, chunksize=1)


def load(data_dir, train):
    """The rendered pictures of the training scenes, or of the held-out ones, one dict
    per picture."""
    pictures = []
    for file in sorted(Path(data_dir).glob("*.npz")):
        if file.stem.endswith("_train") != train:
            continue
        with np.load(file) as scene:
            ends = np.cumsum(scene["count"])
            for i, (end, count) in enumerate(zip(ends, scene["count"])):
                rows = slice(end - count, end)
                pictures.append({"image": scene["image"][i], "cam_pos": scene["cam_pos"][i],
                                 "cam_mat": scene["cam_mat"][i], "fovy": float(scene["fovy"]),
                                 "boxes": scene["boxes"][rows], "labels": scene["labels"][rows],
                                 "xy": scene["xy"][rows]})
    return pictures


def plane_heights(pictures):
    """Per type, the plane height that places the rendered boxes nearest the truth along
    the bed. Boxes cut by the picture's edge are left out: their middle is not the
    plant's."""
    size = WeedDetector.SIZE
    heights = {}
    for index, label in enumerate(WeedDetector.LABELS):
        whole = [(p, box, xy) for p in pictures
                 for box, lab, xy in zip(p["boxes"], p["labels"], p["xy"])
                 if lab == index + 1 and box.min() > 0 and box.max() < size]
        candidates = np.arange(0.05, 0.40, 0.005)
        errors = [np.mean([abs(place(p["cam_pos"], p["cam_mat"], p["fovy"], box, z)[0] - xy[0])
                           for p, box, xy in whole]) for z in candidates]
        heights[label] = float(candidates[int(np.argmin(errors))])
    return heights


def train(data_dir, out_file, epochs, device="cuda"):
    pictures = load(data_dir, train=True)
    heights = plane_heights(pictures)
    net = WeedDetector.network().to(device)
    optimiser = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, 0.02, epochs=epochs,
                                                   steps_per_epoch=-(-len(pictures) // 8))
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        net.train()
        order, total = rng.permutation(len(pictures)), 0.0
        for start in range(0, len(order), 8):
            batch = [pictures[i] for i in order[start:start + 8]]
            images = [as_tensor(p["image"]).to(device) for p in batch]
            targets = [{"boxes": torch.from_numpy(p["boxes"]).to(device),
                        "labels": torch.from_numpy(p["labels"]).to(device)} for p in batch]
            loss = sum(net(images, targets).values())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += loss.item()
        print(f"epoch {epoch + 1}/{epochs}: loss {total / -(-len(order) // 8):.3f}", flush=True)
    torch.save({"net": net.state_dict(), "heights": heights}, out_file)
    return check(WeedDetector(out_file, device), load(data_dir, train=False))


def check(detector, pictures):
    """How the detector does on `pictures`: per type, the plants found, the plants given
    the wrong type, and how far along the bed the found ones are misplaced."""
    report = {label: {"plants": 0, "found": 0, "wrong_type": 0, "x_errors": []}
              for label in detector.LABELS}
    false_boxes = 0
    for p in pictures:
        found = detector.detect(p["image"])
        centres = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for _, _, b in found])
        used = set()
        for box, lab, xy in zip(p["boxes"], p["labels"], p["xy"]):
            if box.min() <= 0 or box.max() >= detector.SIZE:
                continue  # cut by the edge: finding it is not required
            row = report[detector.LABELS[lab - 1]]
            row["plants"] += 1
            inside = [i for i, c in enumerate(centres) if i not in used
                      and box[0] - 4 <= c[0] <= box[2] + 4 and box[1] - 4 <= c[1] <= box[3] + 4]
            if not inside:
                continue
            best = max(inside, key=lambda i: found[i][1])
            used.add(best)
            label = found[best][0]
            if label != detector.LABELS[lab - 1]:
                row["wrong_type"] += 1
                continue
            row["found"] += 1
            placed = place(p["cam_pos"], p["cam_mat"], p["fovy"], found[best][2],
                           detector.heights[label])
            row["x_errors"].append(abs(placed[0] - xy[0]))
        false_boxes += len(found) - len(used)
    for label, row in report.items():
        errors = row.pop("x_errors")
        row["median_x_error_mm"] = round(float(np.median(errors)) * 1000) if errors else None
        row["worst_x_error_mm"] = round(float(np.max(errors)) * 1000) if errors else None
    return {"pictures": len(pictures), "boxes_on_nothing_or_cut_plants": false_boxes, **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    rendering = commands.add_parser("render", help="render labelled bed camera pictures")
    rendering.add_argument("data_dir", type=Path)
    rendering.add_argument("--variants", type=int, default=9, help="scenes per family")
    training = commands.add_parser("train", help="train the detector and check it")
    training.add_argument("data_dir", type=Path)
    training.add_argument("out_file", type=Path)
    training.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    if args.command == "render":
        render(args.data_dir.resolve(), args.variants)
    else:
        for key, value in train(args.data_dir, args.out_file, args.epochs).items():
            print(key, value)


if __name__ == "__main__":
    main()
