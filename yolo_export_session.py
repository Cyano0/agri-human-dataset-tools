#!/usr/bin/env python3
"""
YOLO export for ONE session folder (e.g. footpath1_..._label/).

What it does
------------
- Reads <session>/sync.json (produced by sync_and_match.py)
- Uses an anchor camera (default: cam_zed_rgb) to choose the image per sample
- Reads 2D annotations from <session>/annotations/<camera>_ann.json
- Writes YOLO dataset structure:
    <out>/
      images/{train,val,test}/000000.png
      labels/{train,val,test}/000000.txt
      data.yaml
      classes.txt

Split modes
-----------
A) Use split files (recommended):
   Provide --splits_root pointing to the directory that contains:
     splits/default/train.txt, val.txt, test.txt
   Those split files should contain sample IDs in the form:
     <session_name>\t<timestamp_ns>
   (That’s what build_manifest_and_splits.py typically writes.)

B) Ratio split (if you only have one session or no global splits):
   Use --split 0.8,0.1,0.1 (default) and --seed.

Notes
-----
- YOLO label format: <class_id> <cx> <cy> <w> <h>  (all normalized by image W/H)
- If an image has no objects, we still write an empty label file (common practice).

Run examples
------------
1) Simple ratio split for one session:
   python yolo_export_session.py --session dataset_root/footpath1_..._label --out yolo_out

2) Use existing global splits:
   python yolo_export_session.py \
     --session dataset_root/footpath1_..._label \
     --out yolo_out \
     --splits_root dataset_root

3) Map classes (e.g., human1/human2 -> person):
   python yolo_export_session.py ... --class_map '{"human1":"person","human2":"person"}'
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml  # pyyaml
except ImportError:
    yaml = None


@dataclass(frozen=True)
class SampleKey:
    session: str
    timestamp_ns: int


def load_sync_samples(session_dir: Path) -> List[dict]:
    p = session_dir / "sync.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing sync.json: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    # Accept either {"samples":[...]} or direct list
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ValueError("sync.json schema not recognized (expected dict with 'samples' or a list).")


def parse_ts_ns(sample: dict) -> int:
    # supports either timestamp_ns or timestamp (seconds float)
    if "timestamp_ns" in sample:
        return int(sample["timestamp_ns"])
    if "timestamp" in sample:
        # seconds float -> ns
        return int(float(sample["timestamp"]) * 1e9)
    raise ValueError("sync sample missing timestamp_ns/timestamp.")


def load_ann_index(ann_path: Path) -> Dict[str, List[dict]]:
    """
    Index annotations by filename stem and exact filename.
    Supports a list like:
      {"File":"173...png","Labels":[{"Class":"human1","BoundingBoxes":[x,y,w,h]}, ...]}
    """
    if not ann_path.exists():
        return {}
    data = json.loads(ann_path.read_text(encoding="utf-8"))

    idx: Dict[str, List[dict]] = {}

    def add(key: str, labels: List[dict]):
        if not key:
            return
        stem = Path(key).stem.lower()
        idx.setdefault(stem, []).extend(labels or [])
        idx.setdefault(key.lower(), []).extend(labels or [])

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            k = item.get("File") or item.get("file") or item.get("filename") or item.get("name")
            labels = item.get("Labels") or item.get("labels") or item.get("objects") or []
            add(k, labels)
    elif isinstance(data, dict):
        # could be { "<file>": [labels...] } or {"frames":[...]}
        if "frames" in data and isinstance(data["frames"], list):
            for item in data["frames"]:
                k = item.get("File") or item.get("file") or item.get("filename")
                labels = item.get("Labels") or item.get("labels") or []
                add(k, labels)
        else:
            for k, v in data.items():
                if isinstance(v, list):
                    add(k, v)
                elif isinstance(v, dict) and "Labels" in v:
                    add(k, v["Labels"])
    return idx


def get_labels_for_image(ann_idx: Dict[str, List[dict]], filename: str) -> List[dict]:
    if not filename:
        return []
    k1 = filename.lower()
    k2 = Path(filename).stem.lower()
    return ann_idx.get(k1, ann_idx.get(k2, []))


def read_image_size(img_path: Path) -> Tuple[int, int]:
    """
    Minimal dependency: use Pillow if available, else fall back to OpenCV if available.
    """
    try:
        from PIL import Image  # type: ignore
        with Image.open(img_path) as im:
            return im.size[0], im.size[1]  # (W,H)
    except Exception:
        try:
            import cv2  # type: ignore
            im = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if im is None:
                raise RuntimeError("cv2.imread returned None")
            h, w = im.shape[:2]
            return w, h
        except Exception as e:
            raise RuntimeError(f"Cannot read image size for {img_path}: {e}")


def xywh_to_yolo(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[float, float, float, float]:
    # input bbox: [x,y,w,h] top-left origin, pixel units
    cx = x + w / 2.0
    cy = y + h / 2.0
    return cx / W, cy / H, w / W, h / H


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def symlink_or_copy(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError("--link_mode must be 'symlink' or 'copy'")


def parse_split_arg(s: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("--split must be like 0.8,0.1,0.1")
    a, b, c = (float(parts[0]), float(parts[1]), float(parts[2]))
    total = a + b + c
    if total <= 0:
        raise ValueError("--split sum must be > 0")
    return a / total, b / total, c / total


def load_split_files(splits_root: Path) -> Dict[str, set]:
    """
    Expect:
      <splits_root>/splits/default/train.txt  etc
    Each line: "<session>\\t<timestamp_ns>" OR "<session> <timestamp_ns>".
    Returns dict: {"train": set(SampleKey), ...}
    """
    out = {"train": set(), "val": set(), "test": set()}
    base = splits_root / "splits" / "default"
    for tag in ("train", "val", "test"):
        p = base / f"{tag}.txt"
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                sess, ts = line.split("\t", 1)
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                sess, ts = parts[0], parts[1]
            out[tag].add(SampleKey(sess, int(ts)))
    return out


def write_data_yaml(out_dir: Path, names: List[str]) -> None:
    d = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    p = out_dir / "data.yaml"
    if yaml is None:
        # minimal YAML writer
        lines = []
        lines.append(f"path: {d['path']}")
        lines.append(f"train: {d['train']}")
        lines.append(f"val: {d['val']}")
        lines.append(f"test: {d['test']}")
        lines.append("names:")
        for i, n in enumerate(names):
            lines.append(f"  {i}: {n}")
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="Path to one *_label session folder")
    ap.add_argument("--out", required=True, help="Output YOLO dataset folder")
    ap.add_argument("--anchor_camera", default="cam_zed_rgb", help="Anchor camera modality")
    ap.add_argument("--camera_folder", default=None,
                    help="Override sensor_data/<camera> folder name (default = anchor_camera)")
    ap.add_argument("--ann_json", default=None,
                    help="Override annotation json path (default: annotations/<anchor_camera>_ann.json)")
    ap.add_argument("--link_mode", choices=["symlink", "copy"], default="symlink",
                    help="How to place images into YOLO folder")
    ap.add_argument("--split", default="0.8,0.1,0.1", help="train,val,test ratios (used if --splits_root not set)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits_root", default=None,
                    help="If set, use <splits_root>/splits/default/{train,val,test}.txt to select samples")
    ap.add_argument("--class_map", default=None,
                    help="JSON dict mapping original class -> new class (e.g. '{\"human1\":\"person\"}')")
    ap.add_argument("--drop_unknown", action="store_true",
                    help="Drop labels whose class isn't in class_map (if class_map is provided)")
    args = ap.parse_args()

    session_dir = Path(args.session).resolve()
    out_dir = Path(args.out).resolve()
    ensure_dir(out_dir)

    session_name = session_dir.name
    samples = load_sync_samples(session_dir)

    cam_folder = args.camera_folder or args.anchor_camera
    img_root = session_dir / "sensor_data" / cam_folder

    ann_path = Path(args.ann_json).resolve() if args.ann_json else (session_dir / "annotations" / f"{args.anchor_camera}_ann.json")
    ann_idx = load_ann_index(ann_path)

    class_map = json.loads(args.class_map) if args.class_map else None
    if class_map:
        # normalize keys to lower
        class_map = {str(k).lower(): str(v) for k, v in class_map.items()}

    # Build list of (SampleKey, image_file)
    items: List[Tuple[SampleKey, str]] = []
    for s in samples:
        ts = parse_ts_ns(s)
        # anchor image file can be stored in different ways
        if s.get("anchor_modality") == args.anchor_camera and s.get("anchor_file"):
            img_file = s["anchor_file"]
        else:
            img_file = (s.get("cameras", {}) or {}).get(args.anchor_camera)
        if not img_file or img_file == "null":
            continue
        items.append((SampleKey(session_name, ts), img_file))

    if not items:
        raise RuntimeError("No usable items found (check sync.json and anchor_camera).")

    # Determine split assignment
    tag_for: Dict[SampleKey, str] = {}
    if args.splits_root:
        split_sets = load_split_files(Path(args.splits_root).resolve())
        for k, _img in items:
            if k in split_sets["train"]:
                tag_for[k] = "train"
            elif k in split_sets["val"]:
                tag_for[k] = "val"
            elif k in split_sets["test"]:
                tag_for[k] = "test"
        # Keep only those present in split files
        items = [(k, img) for (k, img) in items if k in tag_for]
    else:
        tr, va, te = parse_split_arg(args.split)
        rng = random.Random(args.seed)
        rng.shuffle(items)
        n = len(items)
        n_tr = int(round(n * tr))
        n_va = int(round(n * va))
        train = items[:n_tr]
        val = items[n_tr:n_tr + n_va]
        test = items[n_tr + n_va:]
        for k, _ in train: tag_for[k] = "train"
        for k, _ in val: tag_for[k] = "val"
        for k, _ in test: tag_for[k] = "test"

    # Prepare output dirs
    for tag in ("train", "val", "test"):
        ensure_dir(out_dir / "images" / tag)
        ensure_dir(out_dir / "labels" / tag)

    # Build class list as we go (stable order by first appearance)
    class_to_id: Dict[str, int] = {}

    def get_class_id(cls_name: str) -> Optional[int]:
        nonlocal class_to_id
        c = cls_name.strip()
        if not c:
            return None
        if class_map is not None:
            key = c.lower()
            if key in class_map:
                c = class_map[key]
            elif args.drop_unknown:
                return None
        if c not in class_to_id:
            class_to_id[c] = len(class_to_id)
        return class_to_id[c]

    # Export
    counter = {"train": 0, "val": 0, "test": 0}
    for k, img_file in items:
        tag = tag_for.get(k)
        if tag not in ("train", "val", "test"):
            continue

        src_img = img_root / img_file
        if not src_img.exists():
            # skip missing files
            continue

        idx = counter[tag]
        counter[tag] += 1
        out_img = out_dir / "images" / tag / f"{idx:06d}{src_img.suffix.lower()}"
        out_lab = out_dir / "labels" / tag / f"{idx:06d}.txt"

        # place image
        symlink_or_copy(src_img, out_img, args.link_mode)

        # labels
        W, H = read_image_size(src_img)
        objs = get_labels_for_image(ann_idx, img_file)

        yolo_lines: List[str] = []
        for obj in objs:
            cls = obj.get("Class") or obj.get("class") or obj.get("type")
            bb = obj.get("BoundingBoxes") or obj.get("bbox") or obj.get("box")
            if cls is None or bb is None:
                continue
            if not isinstance(bb, (list, tuple)) or len(bb) < 4:
                continue
            x, y, w, h = map(float, bb[:4])

            # normalize and clamp
            cx, cy, ww, hh = xywh_to_yolo(x, y, w, h, W, H)
            cx, cy, ww, hh = clamp01(cx), clamp01(cy), clamp01(ww), clamp01(hh)

            cid = get_class_id(str(cls))
            if cid is None:
                continue
            # YOLO expects numbers typically with 6 decimals
            yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")

        out_lab.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

    # Write classes + data.yaml
    classes = [None] * len(class_to_id)
    for name, i in class_to_id.items():
        classes[i] = name
    classes = [c for c in classes if c is not None]

    (out_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    write_data_yaml(out_dir, classes)

    print(f"[done] session={session_name}")
    print(f"  images: train={counter['train']} val={counter['val']} test={counter['test']}")
    print(f"  classes: {classes}")
    print(f"  wrote: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()

