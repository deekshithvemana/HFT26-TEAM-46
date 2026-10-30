import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.video.track import objects_from_result, summarise_tracks, track_video


class _Tensor:
    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._data


class _Boxes:
    def __init__(self, xyxy, conf, cls, track_ids) -> None:
        self.xyxy = _Tensor(np.array(xyxy, dtype=float))
        self.conf = _Tensor(np.array(conf, dtype=float))
        self.cls = _Tensor(np.array(cls, dtype=float))
        self.id = None if track_ids is None else _Tensor(np.array(track_ids, dtype=float))

    def __len__(self) -> int:
        return len(self.conf.numpy())


class _Result:
    def __init__(self, frame, boxes) -> None:
        self.orig_img = frame
        self.boxes = boxes


def test_objects_from_result_maps_coco_road_users() -> None:
    result = _Result(
        np.zeros((100, 100, 3), dtype=np.uint8),
        _Boxes(
            xyxy=[[10, 20, 40, 80]],
            conf=[0.91],
            cls=[2],
            track_ids=[7],
        ),
    )
    objects = objects_from_result(result)
    assert objects == [
        {
            "track_id": 7,
            "class": "car",
            "class_id": 2,
            "bbox": [10.0, 20.0, 40.0, 80.0],
            "centre": [25.0, 50.0],
            "conf": 0.91,
        }
    ]


def test_track_video_writes_contract_json(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    writer = __import__("cv2").VideoWriter(
        str(clip),
        __import__("cv2").VideoWriter_fourcc(*"mp4v"),
        10.0,
        (640, 360),
    )
    for _ in range(3):
        writer.write(frame)
    writer.release()

    fake_results = [
        _Result(
            frame.copy(),
            _Boxes([[50, 40, 120, 90]], [0.8], [2], [1]),
        ),
        _Result(
            frame.copy(),
            _Boxes([[60, 42, 130, 92]], [0.82], [2], [1]),
        ),
        _Result(frame.copy(), _Boxes([], [], [], [])),
    ]

    out = tmp_path / "processed"
    payload = track_video(
        clip,
        output_dir=out,
        write_annotated=False,
        predict=lambda: fake_results,
    )

    tracks_path = out / "tracks.json"
    assert tracks_path.exists()
    data = json.loads(tracks_path.read_text(encoding="utf-8"))
    assert data["fps"] == payload["fps"]
    assert "frames" in data
    assert data["frames"][0]["objects"][0]["track_id"] == 1
    assert data["frames"][0]["objects"][0]["bbox"]
    assert data["n_tracks"] == 1
    summary = summarise_tracks(data["frames"])
    assert summary[0]["class"] == "car"
