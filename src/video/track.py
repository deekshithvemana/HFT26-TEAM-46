"""P1 Task 2 — YOLO11 detection and BoT-SORT tracking.

Detects cars, motorcycles, buses, trucks, bicycles and pedestrians, then assigns
video-local track IDs. Exports tracks.json for P2/P3.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import cv2
import numpy as np

from src.video.preprocess import _tool_path, probe_video, sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKER = ROOT / "configs" / "botsort.yaml"
DEFAULT_MODEL = "yolo11n.pt"

# COCO ids used as road users in the architecture PDF.
ROAD_CLASS_IDS = (0, 1, 2, 3, 5, 7)
CLASS_NAMES = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def objects_from_result(result: Any) -> list[dict[str, Any]]:
    """Convert one Ultralytics result into schema objects."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    if boxes.id is None:
        track_ids = np.full(len(xyxy), -1, dtype=int)
    else:
        track_ids = boxes.id.cpu().numpy().astype(int)

    objects: list[dict[str, Any]] = []
    for bbox, conf, cls_id, track_id in zip(xyxy, confs, classes, track_ids):
        if int(cls_id) not in CLASS_NAMES:
            continue
        x1, y1, x2, y2 = (float(v) for v in bbox)
        objects.append(
            {
                "track_id": int(track_id),
                "class": CLASS_NAMES[int(cls_id)],
                "class_id": int(cls_id),
                "bbox": [_round(x1, 1), _round(y1, 1), _round(x2, 1), _round(y2, 1)],
                "centre": [_round((x1 + x2) / 2.0, 1), _round((y1 + y2) / 2.0, 1)],
                "conf": _round(float(conf), 4),
            }
        )
    return objects


def summarise_tracks(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    histories: dict[int, list[tuple[float, str, float]]] = defaultdict(list)
    for row in frames:
        time_s = float(row["time_s"])
        for obj in row["objects"]:
            track_id = int(obj["track_id"])
            if track_id < 0:
                continue
            histories[track_id].append((time_s, obj["class"], float(obj["conf"])))

    summaries = []
    for track_id, rows in sorted(histories.items()):
        classes = [cls for _, cls, _ in rows]
        majority = max(set(classes), key=classes.count)
        confs = [conf for _, _, conf in rows]
        summaries.append(
            {
                "track_id": track_id,
                "class": majority,
                "n_frames": len(rows),
                "t_start": _round(rows[0][0], 6),
                "t_end": _round(rows[-1][0], 6),
                "mean_conf": _round(sum(confs) / len(confs), 4),
            }
        )
    return summaries


def _annotate_frame(frame: np.ndarray, objects: list[dict[str, Any]]) -> np.ndarray:
    canvas = frame.copy()
    palette = {
        "car": (36, 180, 36),
        "truck": (40, 80, 230),
        "bus": (20, 160, 230),
        "motorcycle": (200, 80, 20),
        "bicycle": (180, 40, 180),
        "pedestrian": (40, 200, 220),
    }
    for obj in objects:
        x1, y1, x2, y2 = (int(round(v)) for v in obj["bbox"])
        color = palette.get(obj["class"], (240, 240, 240))
        label = f"ID {obj['track_id']} {obj['class']} {obj['conf']:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (15, 15, 15),
            1,
            cv2.LINE_AA,
        )
    return canvas


class _H264Writer:
    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        self.path = path
        width, height = size
        ffmpeg = _tool_path("ffmpeg")
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self.width = width
        self.height = height
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        if self.proc.stdin is None:
            raise RuntimeError("ffmpeg stdin could not be opened")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        code = self.proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path} (exit {code})")


def _iter_yolo_results(
    source: Path,
    model_name: str,
    tracker: Path,
    conf: float,
    imgsz: int,
    device: str,
) -> Iterable[Any]:
    from ultralytics import YOLO

    model = YOLO(model_name)
    yield from model.track(
        source=str(source),
        persist=True,
        tracker=str(tracker),
        classes=list(ROAD_CLASS_IDS),
        conf=conf,
        iou=0.5,
        imgsz=imgsz,
        device=device,
        stream=True,
        verbose=False,
        save=False,
    )


def track_video(
    video_path: str | Path,
    output_dir: str | Path | None = None,
    model_name: str = DEFAULT_MODEL,
    tracker_config: str | Path | None = None,
    conf: float = 0.25,
    imgsz: int = 640,
    write_annotated: bool = True,
    predict: Callable[..., Iterable[Any]] | None = None,
) -> dict[str, Any]:
    source = Path(video_path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    file_hash = sha256_file(source)
    video_id = f"{source.stem}_{file_hash[:8]}"
    out_root = Path(output_dir) if output_dir else Path("data/processed") / video_id
    out_root.mkdir(parents=True, exist_ok=True)

    existing_meta = out_root / "video_metadata.json"
    if existing_meta.exists():
        meta = json.loads(existing_meta.read_text(encoding="utf-8"))
        video_id = meta.get("video_id", video_id)
        fps = float(meta.get("fps") or 0.0) or 25.0
        start_offset_s = float(meta.get("start_offset_s") or 0.0)
        width = int(meta.get("original_width") or 0)
        height = int(meta.get("original_height") or 0)
    else:
        probed = probe_video(source)
        fps = float(probed.get("fps") or 25.0)
        start_offset_s = 0.0
        width = int(probed.get("width") or 0)
        height = int(probed.get("height") or 0)

    tracker = Path(tracker_config) if tracker_config else DEFAULT_TRACKER
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    predictor = predict or (
        lambda: _iter_yolo_results(source, model_name, tracker, conf, imgsz, device)
    )

    frames: list[dict[str, Any]] = []
    writer: _H264Writer | None = None
    annotated_path = out_root / "annotated.mp4"
    frame_index = 0
    try:
        for result in predictor():
            orig = result.orig_img
            if orig is None:
                continue
            if not width or not height:
                height, width = orig.shape[:2]
            objects = objects_from_result(result)
            time_s = frame_index / fps if fps else 0.0
            frames.append(
                {
                    "frame": frame_index,
                    "time_s": _round(time_s, 6),
                    "objects": objects,
                }
            )
            if write_annotated:
                if writer is None:
                    writer = _H264Writer(annotated_path, fps, (width, height))
                writer.write(_annotate_frame(orig, objects))
            frame_index += 1
    finally:
        if writer is not None:
            writer.close()

    payload = {
        "video_id": video_id,
        "source_path": str(source),
        "fps": _round(fps, 6),
        "start_offset_s": start_offset_s,
        "model": Path(model_name).name,
        "tracker": "botsort",
        "tracker_config": str(tracker),
        "device": device,
        "conf_threshold": conf,
        "imgsz": imgsz,
        "kind": "observation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "time_convention": (
            "time_s is seconds from video start. Incident time = time_s + start_offset_s."
        ),
        "n_frames": len(frames),
        "n_tracks": 0,
        "tracks": [],
        "frames": frames,
    }
    payload["tracks"] = summarise_tracks(frames)
    payload["n_tracks"] = len(payload["tracks"])

    tracks_path = out_root / "tracks.json"
    tracks_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["tracks_path"] = str(tracks_path.resolve())

    if write_annotated and annotated_path.exists():
        payload["annotated_path"] = str(annotated_path.resolve())

    payload["output_dir"] = str(out_root.resolve())
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO11 + BoT-SORT tracking")
    parser.add_argument("video", help="Path to the incident clip")
    parser.add_argument("--out", help="Output directory (default: data/processed/<video_id>)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tracker", default=str(DEFAULT_TRACKER))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-annotated", action="store_true")
    args = parser.parse_args()

    result = track_video(
        args.video,
        output_dir=args.out,
        model_name=args.model,
        tracker_config=args.tracker,
        conf=args.conf,
        imgsz=args.imgsz,
        write_annotated=not args.no_annotated,
    )
    summary = {
        "video_id": result["video_id"],
        "n_frames": result["n_frames"],
        "n_tracks": result["n_tracks"],
        "tracks": result["tracks"],
        "tracks_path": result.get("tracks_path"),
        "annotated_path": result.get("annotated_path"),
        "device": result["device"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
