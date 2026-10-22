"""this python script is as a helper for functions to detect packaging defects in pork rasher trays.

The AI has noted the below.
This module supports the main workflow notebook
(``notebooks/pork_rasher_defects.ipynb``). 

the small, testable pieces of the
project live here so they can be checked with ``pytest``:

preparing the Roboflow dataset for training
(`fix_data_yaml`, `read_class_names`);
auditing the labels before training
(`parse_label_line`, `count_labels`, `find_images_with_class`,
`draw_yolo_boxes`);
checking for possible leakage between neighbouring video frames in the
training and test sets (`frame_number`, `nearest_frame_gaps`,
`leakage_summary`);
turning test results and timings into engineering terms
  (`summarise_test_metrics`, `per_class_table`, `images_per_second`,
  `meets_line_speed`).

Label format:
Every image has a matching ``.txt`` label file in YOLO format. Each line
describes one object as ``class_id x_centre y_centre width height``. The
four box values are fractions (0 to 1) of the image width and height, so
they do not depend on the image resolution.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

#: Maps the split names Ultralytics expects to Roboflow's folder names.
SPLIT_FOLDERS = {"train": "train", "val": "valid", "test": "test"}

#: Image file extensions tried when matching a label file to its image.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

#: Box colours (RGB) used by `draw_yolo_boxes`, one per class id (cycled).
BOX_COLOURS = (
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
)

_FRAME_PATTERN = re.compile(r"frame\D*?(\d+)", re.IGNORECASE)


# Dataset preparation


def fix_data_yaml(dataset_dir: str | Path) -> Path:
    """Rewrite a Roboflow ``data.yaml`` so every split uses an absolute path.

    Roboflow exports store the split folders as relative paths such as
    ``../train/images``. Whether those resolve correctly depends on the folder
    training is started from, so this function replaces them with absolute
    paths that always work.

    Args:
        dataset_dir: Folder created by the Roboflow download. It must contain
            ``data.yaml`` plus ``train``, ``valid`` and ``test`` folders, each
            with an ``images`` sub-folder.

    Returns:
        Path to the rewritten ``data.yaml``.

    Raises:
        FileNotFoundError: If ``data.yaml`` or any split's ``images`` folder
            is missing.
    """
    dataset_dir = Path(dataset_dir).resolve()
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"No data.yaml found in {dataset_dir}")

    with open(yaml_path, encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    for key, folder in SPLIT_FOLDERS.items():
        images_dir = dataset_dir / folder / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Missing '{key}' images folder: {images_dir}")
        config[key] = str(images_dir)
    config["path"] = str(dataset_dir)

    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    return yaml_path


def read_class_names(yaml_path: str | Path) -> list[str]:
    """Read the class names from a dataset ``data.yaml`` file.

    Args:
        yaml_path: Path to ``data.yaml``. Its ``names`` entry may be a list
            (Roboflow style) or a ``{class_id: name}`` dictionary (Ultralytics
            style).

    Returns:
        Class names ordered by class id.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has no ``names`` entry, or the dictionary ids
            are not exactly ``0 .. n-1``.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"File not found: {yaml_path}")
    with open(yaml_path, encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    names = config.get("names")
    if not names:
        raise ValueError(f"No 'names' entry in {yaml_path}")
    if isinstance(names, dict):
        ids = sorted(int(key) for key in names)
        if ids != list(range(len(ids))):
            raise ValueError(f"Class ids must be 0..n-1, got {ids}")
        return [str(names[key]) for key in sorted(names, key=int)]
    return [str(name) for name in names]


# Label auditing


def parse_label_line(line: str, n_classes: int) -> tuple[int, float, float, float, float]:
    """Parse and check one line of a YOLO label file.

    Two layouts are accepted, because Roboflow exports either one:

    * a box: ``class_id x_centre y_centre width height``;
    * a polygon: ``class_id x1 y1 x2 y2 x3 y3 ...`` (at least three points),
      which is converted to the box that encloses it.

    Args:
        line: One line of text from a label file.
        n_classes: Number of classes in the dataset. Class ids must lie in
            ``0 .. n_classes - 1``.

    Returns:
        ``(class_id, x_centre, y_centre, width, height)``, with the box values
        as fractions of the image size.

    Raises:
        ValueError: If the line has the wrong number of values, a value is not
            a number, the class id is not a whole number in range, or a
            coordinate lies outside 0 to 1.
    """
    parts = line.split()
    is_box = len(parts) == 5
    is_polygon = len(parts) >= 7 and (len(parts) - 1) % 2 == 0
    if not (is_box or is_polygon):
        raise ValueError(f"Expected 5 values (box) or a polygon, got {len(parts)}: {line!r}")

    try:
        class_value = float(parts[0])
        coords = [float(value) for value in parts[1:]]
    except ValueError as error:
        raise ValueError(f"Non-numeric value in label line: {line!r}") from error

    if not class_value.is_integer() or not 0 <= class_value < n_classes:
        raise ValueError(f"Class id {parts[0]} is not in 0..{n_classes - 1}")
    if any(not 0.0 <= value <= 1.0 for value in coords):
        raise ValueError(f"Coordinates must be between 0 and 1: {line!r}")

    if is_box:
        x_centre, y_centre, width, height = coords
    else:
        xs, ys = coords[0::2], coords[1::2]
        width, height = max(xs) - min(xs), max(ys) - min(ys)
        x_centre, y_centre = min(xs) + width / 2, min(ys) + height / 2
    return int(class_value), x_centre, y_centre, width, height


def count_labels(labels_dir: str | Path, class_names: Sequence[str]) -> dict[str, int]:
    """Count the labelled boxes of each class in one split of the dataset.

    Broken lines are counted instead of stopping the audit, so one bad file
    can be investigated rather than silently deleted.

    Args:
        labels_dir: Folder of YOLO ``.txt`` label files, e.g. ``train/labels``.
        class_names: Class names in class-id order.

    Returns:
        One entry per class name (number of boxes), plus ``"images"`` (label
        files read), ``"images with no boxes"`` and ``"invalid lines"``.

    Raises:
        FileNotFoundError: If ``labels_dir`` does not exist.
    """
    labels_dir = Path(labels_dir)
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    counts = {name: 0 for name in class_names}
    images = empty_images = invalid_lines = 0
    for label_file in sorted(labels_dir.glob("*.txt")):
        images += 1
        boxes_in_file = 0
        for line in label_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                class_id = parse_label_line(line, len(class_names))[0]
            except ValueError:
                invalid_lines += 1
                continue
            counts[class_names[class_id]] += 1
            boxes_in_file += 1
        if boxes_in_file == 0:
            empty_images += 1

    counts["images"] = images
    counts["images with no boxes"] = empty_images
    counts["invalid lines"] = invalid_lines
    return counts


def _matching_image(label_file: Path, images_dir: Path) -> Path | None:
    """Return the image that belongs to a label file, or ``None`` if missing."""
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{label_file.stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def find_images_with_class(split_dir: str | Path, class_id: int, limit: int = 4) -> list[tuple[Path, Path]]:
    """Find example images that contain at least one box of a given class.

    Args:
        split_dir: A split folder (e.g. ``valid``) containing ``images`` and
            ``labels`` sub-folders.
        class_id: The class id to look for.
        limit: Maximum number of examples to return.

    Returns:
        Up to ``limit`` ``(image_path, label_path)`` pairs, in file-name order.

    Raises:
        FileNotFoundError: If the ``labels`` folder does not exist.
        ValueError: If ``limit`` is less than 1.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    split_dir = Path(split_dir)
    labels_dir, images_dir = split_dir / "labels", split_dir / "images"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    found: list[tuple[Path, Path]] = []
    for label_file in sorted(labels_dir.glob("*.txt")):
        class_ids = set()
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                try:
                    class_ids.add(int(float(parts[0])))
                except ValueError:
                    continue
        if class_id in class_ids:
            image = _matching_image(label_file, images_dir)
            if image is not None:
                found.append((image, label_file))
                if len(found) >= limit:
                    break
    return found


def draw_yolo_boxes(
    image_path: str | Path,
    label_path: str | Path,
    class_names: Sequence[str],
    line_width: int = 3,
) -> Image.Image:
    """Draw an image's labelled boxes onto a copy of the image.

    Used to look at what each class actually marks before training. Lines
    that fail `parse_label_line` are skipped.

    Args:
        image_path: Path to the image file.
        label_path: Path to its YOLO label file.
        class_names: Class names in class-id order (used for box captions).
        line_width: Box outline thickness in pixels.

    Returns:
        A new RGB image with one coloured, captioned rectangle per box.

    Raises:
        FileNotFoundError: If the image or the label file does not exist.
    """
    label_path = Path(label_path)
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file not found: {label_path}")
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            class_id, x_centre, y_centre, box_w, box_h = parse_label_line(line, len(class_names))
        except ValueError:
            continue
        colour = BOX_COLOURS[class_id % len(BOX_COLOURS)]
        box = (
            (x_centre - box_w / 2) * width,
            (y_centre - box_h / 2) * height,
            (x_centre + box_w / 2) * width,
            (y_centre + box_h / 2) * height,
        )
        draw.rectangle(box, outline=colour, width=line_width)
        draw.text((box[0] + line_width + 2, box[1] + line_width + 1), class_names[class_id], fill=colour)
    return image


# Train/test leakage check


def frame_number(filename: str | Path) -> int | None:
    """Extract the video frame number from an image file name.

    Roboflow renames uploaded files, for example
    ``frame_with_red_8669_jpg.rf.3f2a9c.jpg``. The first number after the
    word "frame" (``8669`` here) is the image's position in the original
    video.

    Args:
        filename: File name or path.

    Returns:
        The frame number, or ``None`` if the name contains no frame number.
    """
    match = _FRAME_PATTERN.search(Path(filename).name)
    return int(match.group(1)) if match else None


def nearest_frame_gaps(test_frames: Iterable[int], train_frames: Iterable[int]) -> list[int]:
    """For each test frame, find how far away the closest training frame is.

    Neighbouring video frames look almost identical. If a test image has a
    training image only a few frames away, the model has effectively seen it
    already, so test scores can look better than real-world performance.

    The training frames are sorted once and each test frame is placed with a
    binary search, so the cost is O((n + m) log m) rather than O(n * m).

    Args:
        test_frames: Frame numbers of the test images.
        train_frames: Frame numbers of the training images.

    Returns:
        One non-negative gap per test frame, in the same order. A gap of 0
        means the same frame number appears in both sets.

    Raises:
        ValueError: If ``train_frames`` is empty.
    """
    train_sorted = sorted(set(train_frames))
    if not train_sorted:
        raise ValueError("train_frames is empty")

    gaps = []
    for frame in test_frames:
        position = bisect.bisect_left(train_sorted, frame)
        candidates = []
        if position < len(train_sorted):
            candidates.append(train_sorted[position] - frame)
        if position > 0:
            candidates.append(frame - train_sorted[position - 1])
        gaps.append(min(candidates))
    return gaps


def leakage_summary(gaps: Sequence[int], thresholds: Sequence[int] = (0, 1, 5, 30)) -> dict[str, float]:
    """Summarise how close test frames sit to training frames.

    Args:
        gaps: Output of `nearest_frame_gaps`.
        thresholds: Frame distances to report.

    Returns:
        For each threshold ``t``, the fraction (0 to 1) of test images whose
        closest training frame is at most ``t`` frames away, keyed as
        ``"within t frames"``.

    Raises:
        ValueError: If ``gaps`` is empty or any threshold is negative.
    """
    if not gaps:
        raise ValueError("gaps is empty")
    if any(threshold < 0 for threshold in thresholds):
        raise ValueError("thresholds must be non-negative")
    total = len(gaps)
    return {f"within {t} frames": sum(gap <= t for gap in gaps) / total for t in thresholds}


# Results in engineering terms


def images_per_second(ms_per_image: float) -> float:
    """Convert a processing time per image into a throughput.

    Args:
        ms_per_image: Time to process one image, in milliseconds.

    Returns:
        Images processed per second.

    Raises:
        ValueError: If ``ms_per_image`` is not positive.
    """
    if ms_per_image <= 0:
        raise ValueError(f"ms_per_image must be positive, got {ms_per_image}")
    return 1000.0 / ms_per_image


def meets_line_speed(ms_per_image: float, trays_per_minute: float) -> bool:
    """Check whether a model is fast enough to inspect every tray on a line.

    At ``trays_per_minute`` trays per minute, each tray allows a time budget
    of ``60 000 / trays_per_minute`` milliseconds.

    Args:
        ms_per_image: Measured time to process one tray image, in milliseconds.
        trays_per_minute: Assumed production-line speed.

    Returns:
        ``True`` if the model finishes each tray within its time budget.

    Raises:
        ValueError: If either argument is not positive.
    """
    if ms_per_image <= 0:
        raise ValueError(f"ms_per_image must be positive, got {ms_per_image}")
    if trays_per_minute <= 0:
        raise ValueError(f"trays_per_minute must be positive, got {trays_per_minute}")
    return ms_per_image <= 60_000.0 / trays_per_minute


def summarise_test_metrics(metrics: Any) -> dict[str, float]:
    """Extract the headline numbers from an Ultralytics validation result.

    Args:
        metrics: The object returned by ``YOLO.val()`` (an Ultralytics
            ``DetMetrics``). Only its ``box`` and ``speed`` attributes are
            used.

    Returns:
        Mean precision, mean recall, mAP50 and mAP50-95 (all 0 to 1), plus the
        validator's batched inference time in milliseconds per image.
    """
    box = metrics.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "batched inference ms/image": float(metrics.speed["inference"]),
    }


def per_class_table(metrics: Any) -> list[dict[str, Any]]:
    """Per-class precision, recall and mAP from an Ultralytics validation result.

    Classes with no labelled boxes in the evaluated split are left out,
    because their scores are undefined.

    Args:
        metrics: The object returned by ``YOLO.val()``. Uses ``box.p``,
            ``box.r``, ``box.ap50``, ``box.ap``, ``box.ap_class_index``,
            ``names`` and, if present, ``nt_per_class``.

    Returns:
        One dictionary per class with keys ``class``, ``test boxes`` (when
        available), ``precision``, ``recall``, ``mAP50`` and ``mAP50-95``.
    """
    box = metrics.box
    box_counts = getattr(metrics, "nt_per_class", None)
    rows = []
    for i, class_id in enumerate(box.ap_class_index):
        class_id = int(class_id)
        row: dict[str, Any] = {"class": metrics.names[class_id]}
        if box_counts is not None:
            row["test boxes"] = int(box_counts[class_id])
        row.update(
            {
                "precision": float(box.p[i]),
                "recall": float(box.r[i]),
                "mAP50": float(box.ap50[i]),
                "mAP50-95": float(box.ap[i]),
            }
        )
        rows.append(row)
    return rows
