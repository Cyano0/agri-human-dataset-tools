# YOLO Export Toolkit for Agri-Human Dataset

This document describes how to export data from the **Agri-Human dataset**
into **YOLO format** for training, validation, and testing.

The YOLO exporter is designed to:
- Reuse the **same synchronisation (`sync.json`)** as KITTI exports
- Reuse **train/val/test splits** produced by `build_manifest_and_splits.py`
- Support **single-camera YOLO today**
- Plan for **multi-camera YOLO** support (RGB + fisheye) in the near future

---

## 1. What is YOLO format?

A YOLO dataset has the following structure:

```
yolo_out/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
├── classes.txt
└── data.yaml
```

Each image has a matching `.txt` file with the same index:

```
labels/train/000123.txt
```

Each line in a label file is:

```
<class_id> <cx> <cy> <w> <h>
```

All values are **normalized to [0,1]**.

---

## 2. Input Requirements (Important)

Before exporting YOLO, you must have:

1. A session folder:
```
footpath1_..._label/
```

2. A valid `sync.json` inside the session  
(created by `sync_and_match.py`)

3. A 2D annotation JSON, for example:
```
annotations/cam_zed_rgb_ann.json
```

4. (Optional but recommended) global splits:
```
splits/default/train.txt
splits/default/val.txt
splits/default/test.txt
```

These are created by:
```
build_manifest_and_splits.py
```

---

## 3. YOLO Export Script

### Script name
```
yolo_export_session.py
```

This script exports **one session** into YOLO format.

---

## 4. Basic Usage (Ratio-based split)

Use this when exporting a single session without global splits:

```bash
python yolo_export_session.py   --session dataset_test/test/footpath1_..._label   --out yolo_footpath1   --anchor_camera cam_zed_rgb   --split 0.8,0.1,0.1   --seed 42
```

---

## 5. Using Existing Train/Val/Test Splits (Recommended)

```bash
python yolo_export_session.py   --session dataset_test/test/footpath1_..._label   --out yolo_footpath1   --anchor_camera cam_zed_rgb   --splits_root dataset_test
```

---

## 6. Class Mapping

```bash
python yolo_export_session.py   --session ...   --out yolo_out   --class_map '{"human1":"person","human2":"person"}'
```

---

## 7. Output Files Explained

- **images/** – training images
- **labels/** – YOLO label files
- **classes.txt** – class list
- **data.yaml** – YOLO training config

---

## 8. Current Limitations

- Single camera only
- No LiDAR-based projection
- 2D annotations only

---

## 9. TODO: Multi-Camera YOLO (Planned)

[ ] Unified multi-camera YOLO

[ ] LiDAR-assisted pseudo-labels

---

## 10. Relationship to KITTI Export

YOLO and KITTI exporters share:
- `sync.json`
- train/val/test splits
