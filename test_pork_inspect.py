"""This script runs tests for ``src/pork_inspect.py``.

Run from the repository root with::

python -m pytest -v tests/

The tests build a tiny fake dataset in a temporary folder, so they need no
internet connection, GPU and any real images.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pork_inspect as pi  # noqa: E402

CLASSES = ["loose-meat", "packaging-error", "twisted-meat", "unsealed", "wrinkle"]


def make_split(root: Path, split: str, labels: dict) -> None:
    """Create ``root/split/images`` and ``root/split/labels`` with blank images."""
    (root / split / "images").mkdir(parents=True)
    (root / split / "labels").mkdir(parents=True)
    for stem, text in labels.items():
        Image.new("RGB", (100, 80), "white").save(root / split / "images" / f"{stem}.jpg")
        (root / split / "labels" / f"{stem}.txt").write_text(text)


@pytest.fixture
def fake_dataset(tmp_path):
    """A three-split dataset laid out like a Roboflow YOLO export."""
    make_split(tmp_path, "train", {"frame_10_jpg.rf.a": "0 0.5 0.5 0.2 0.2\n", "frame_20_jpg.rf.b": "3 0.4 0.4 0.1 0.1\n"})
    make_split(tmp_path, "valid", {"frame_30_jpg.rf.c": "1 0.5 0.5 0.6 0.6\n"})
    make_split(tmp_path, "test", {"frame_11_jpg.rf.d": "", "frame_90_jpg.rf.e": "4 0.5 0.5 0.3 0.3\n"})
    config = {"train": "../train/images", "val": "../valid/images", "test": "../test/images", "nc": 5, "names": CLASSES}
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(config))
    return tmp_path


# Dataset preparation 


def test_fix_data_yaml_makes_split_paths_absolute(fake_dataset):
    config = yaml.safe_load(pi.fix_data_yaml(fake_dataset).read_text())
    for key, folder in [("train", "train"), ("val", "valid"), ("test", "test")]:
        assert Path(config[key]).is_absolute()
        assert Path(config[key]) == (fake_dataset / folder / "images").resolve()
    assert config["names"] == CLASSES  # other entries are kept


def test_fix_data_yaml_missing_yaml_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pi.fix_data_yaml(tmp_path)


def test_fix_data_yaml_missing_split_raises(fake_dataset):
    (fake_dataset / "test" / "images" / "frame_11_jpg.rf.d.jpg").unlink()
    (fake_dataset / "test" / "images" / "frame_90_jpg.rf.e.jpg").unlink()
    (fake_dataset / "test" / "images").rmdir()
    with pytest.raises(FileNotFoundError):
        pi.fix_data_yaml(fake_dataset)


def test_read_class_names_from_list(fake_dataset):
    assert pi.read_class_names(fake_dataset / "data.yaml") == CLASSES


def test_read_class_names_from_dict(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump({"names": {1: "b", 0: "a"}}))
    assert pi.read_class_names(path) == ["a", "b"]


def test_read_class_names_without_names_raises(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump({"nc": 2}))
    with pytest.raises(ValueError):
        pi.read_class_names(path)


# Label auditing


def test_parse_label_line_box():
    assert pi.parse_label_line("3 0.5 0.4 0.2 0.1", 5) == (3, 0.5, 0.4, 0.2, 0.1)


def test_parse_label_line_polygon_becomes_enclosing_box():
    class_id, xc, yc, w, h = pi.parse_label_line("1 0.2 0.2 0.6 0.2 0.6 0.8 0.2 0.8", 5)
    assert class_id == 1
    assert (xc, yc, w, h) == pytest.approx((0.4, 0.5, 0.4, 0.6))


@pytest.mark.parametrize(
    "line",
    [
        "3 0.5 0.4 0.2",  # too few values
        "3 0.5 0.4 0.2 0.1 0.9",  # six values: neither box nor polygon
        "5 0.5 0.5 0.1 0.1",  # class id too large
        "-1 0.5 0.5 0.1 0.1",  # negative class id
        "1.5 0.5 0.5 0.1 0.1",  # class id not a whole number
        "2 0.5 1.4 0.1 0.1",  # coordinate outside 0-1
        "2 0.5 abc 0.1 0.1",  # not a number
    ],
)
def test_parse_label_line_rejects_bad_input(line):
    with pytest.raises(ValueError):
        pi.parse_label_line(line, 5)


def test_count_labels_counts_boxes_empty_files_and_bad_lines(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.1 0.1\n4 0.5 0.5 0.3 0.3\n")
    (labels / "b.txt").write_text("")  # an image with no defects
    (labels / "c.txt").write_text("9 0.5 0.5 0.1 0.1\n\n")  # bad class id, blank line
    counts = pi.count_labels(labels, CLASSES)
    assert counts["loose-meat"] == 2
    assert counts["wrinkle"] == 1
    assert counts["unsealed"] == 0
    assert counts["images"] == 3
    assert counts["images with no boxes"] == 2  # b is empty, c has only a bad line
    assert counts["invalid lines"] == 1


def test_count_labels_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pi.count_labels(tmp_path / "nope", CLASSES)


def test_find_images_with_class(fake_dataset):
    found = pi.find_images_with_class(fake_dataset / "train", class_id=3)
    assert len(found) == 1
    image, label = found[0]
    assert image.name == "frame_20_jpg.rf.b.jpg"
    assert label.name == "frame_20_jpg.rf.b.txt"
    assert pi.find_images_with_class(fake_dataset / "train", class_id=2) == []


def test_find_images_with_class_bad_limit_raises(fake_dataset):
    with pytest.raises(ValueError):
        pi.find_images_with_class(fake_dataset / "train", class_id=0, limit=0)


def test_draw_yolo_boxes_draws_on_box_edge_only(tmp_path):
    image_path, label_path = tmp_path / "img.png", tmp_path / "img.txt"
    Image.new("RGB", (100, 100), "white").save(image_path)
    label_path.write_text("0 0.5 0.5 0.4 0.4\nnot a valid line\n")
    result = np.array(pi.draw_yolo_boxes(image_path, label_path, CLASSES, line_width=2))
    assert tuple(result[50, 30]) == pi.BOX_COLOURS[0]  # left edge of the box (x = 30 px)
    assert tuple(result[5, 5]) == (255, 255, 255)  # outside the box is untouched


def test_draw_yolo_boxes_missing_label_raises(tmp_path):
    image_path = tmp_path / "img.png"
    Image.new("RGB", (10, 10)).save(image_path)
    with pytest.raises(FileNotFoundError):
        pi.draw_yolo_boxes(image_path, tmp_path / "missing.txt", CLASSES)


# Leakage check 


@pytest.mark.parametrize(
    "name, expected",
    [
        ("frame_with_red_8669_jpg.rf.3f2a9c.jpg", 8669),
        ("Frame12.png", 12),
        ("/content/data/test/images/frame_with_red_608_jpg.rf.x1y2.jpg", 608),
        ("tray_photo_001.jpg", None),
    ],
)
def test_frame_number(name, expected):
    assert pi.frame_number(name) == expected


def test_nearest_frame_gaps_known_values():
    assert pi.nearest_frame_gaps([10, 15, 100, 0], [12, 20, 50]) == [2, 3, 50, 12]


def test_nearest_frame_gaps_same_frame_is_zero():
    assert pi.nearest_frame_gaps([7], [7, 8]) == [0]


def test_nearest_frame_gaps_empty_train_raises():
    with pytest.raises(ValueError):
        pi.nearest_frame_gaps([1, 2], [])


def test_leakage_summary_fractions():
    summary = pi.leakage_summary([0, 1, 4, 100], thresholds=(0, 5))
    assert summary == {"within 0 frames": 0.25, "within 5 frames": 0.75}


@pytest.mark.parametrize("gaps, thresholds", [([], (1,)), ([1, 2], (-1,))])
def test_leakage_summary_bad_input_raises(gaps, thresholds):
    with pytest.raises(ValueError):
        pi.leakage_summary(gaps, thresholds)


# Engineering terms


def test_images_per_second():
    assert pi.images_per_second(20.0) == pytest.approx(50.0)


@pytest.mark.parametrize("ms", [0, -5])
def test_images_per_second_rejects_non_positive(ms):
    with pytest.raises(ValueError):
        pi.images_per_second(ms)


def test_meets_line_speed():
    # 60 trays per minute gives a budget of 1000 ms per tray.
    assert pi.meets_line_speed(999.0, 60) is True
    assert pi.meets_line_speed(1000.0, 60) is True
    assert pi.meets_line_speed(1001.0, 60) is False


@pytest.mark.parametrize("ms, rate", [(0, 60), (10, 0), (-1, 60), (10, -60)])
def test_meets_line_speed_rejects_non_positive(ms, rate):
    with pytest.raises(ValueError):
        pi.meets_line_speed(ms, rate)


def fake_metrics():
    """Imitates the parts of an Ultralytics DetMetrics object the helpers use."""
    box = SimpleNamespace(
        mp=0.8,
        mr=0.7,
        map50=0.75,
        map=0.5,
        p=np.array([0.9, 0.6]),
        r=np.array([0.8, 0.5]),
        ap50=np.array([0.85, 0.55]),
        ap=np.array([0.6, 0.3]),
        ap_class_index=np.array([0, 4]),  # class 0 and class 4 have test boxes
    )
    return SimpleNamespace(
        box=box,
        names=dict(enumerate(CLASSES)),
        speed={"preprocess": 0.5, "inference": 4.2, "postprocess": 1.0},
        nt_per_class=np.array([12, 0, 0, 0, 3]),
    )


def test_summarise_test_metrics():
    summary = pi.summarise_test_metrics(fake_metrics())
    assert summary == {
        "precision": 0.8,
        "recall": 0.7,
        "mAP50": 0.75,
        "mAP50-95": 0.5,
        "batched inference ms/image": 4.2,
    }


def test_per_class_table_only_lists_classes_with_test_boxes():
    rows = pi.per_class_table(fake_metrics())
    assert [row["class"] for row in rows] == ["loose-meat", "wrinkle"]
    assert rows[1] == {
        "class": "wrinkle",
        "test boxes": 3,
        "precision": 0.6,
        "recall": 0.5,
        "mAP50": 0.55,
        "mAP50-95": 0.3,
    }
