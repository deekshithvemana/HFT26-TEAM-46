"""P1 Task 1 — video preprocessing.

Uses FFmpeg/ffprobe for container metadata and OpenCV for frame extraction.
Every saved frame is tagged with its original timestamp in seconds from video start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_FFMPEG_BIN = Path(
    r"C:\Users\HP\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-9.0.2-full_build\bin"
)


def _tool_path(name: str) -> str:
    env_key = "FFMPEG_BIN" if name == "ffmpeg" else "FFPROBE_BIN"
    override = os.environ.get(env_key)
    if override:
        return override

    found = shutil.which(name)
    if found:
        return found

    bundled = DEFAULT_FFMPEG_BIN / f"{name}.exe"
    if bundled.exists():
        return str(bundled)

    raise FileNotFoundError(
        f"{name} not found. Install FFmpeg or set {env_key}."
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, Any]:
    """Read container metadata with ffprobe; fall back to OpenCV if needed."""
    try:
        ffprobe = _tool_path("ffprobe")
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        raw = subprocess.check_output(cmd, text=True)
        payload = json.loads(raw)
        video_stream = next(
            (s for s in payload.get("streams", []) if s.get("codec_type") == "video"),
            {},
        )
        fps = _parse_rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
        duration = _to_float(payload.get("format", {}).get("duration"))
        if duration is None:
            duration = _to_float(video_stream.get("duration"))
        nb_frames = _to_int(video_stream.get("nb_frames"))
        rotation = 0
        for side_data in video_stream.get("side_data_list") or []:
            if "rotation" in side_data:
                rotation = int(float(side_data["rotation"]))
        tags = video_stream.get("tags") or {}
        if "rotate" in tags:
            rotation = int(float(tags["rotate"]))

        return {
            "probe_backend": "ffprobe",
            "codec": video_stream.get("codec_name"),
            "width": _to_int(video_stream.get("width")),
            "height": _to_int(video_stream.get("height")),
            "fps": fps,
            "duration_s": duration,
            "nb_frames": nb_frames,
            "start_time_s": _to_float(video_stream.get("start_time")) or 0.0,
            "rotation": rotation,
            "pix_fmt": video_stream.get("pix_fmt"),
            "format_name": payload.get("format", {}).get("format_name"),
        }
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError, StopIteration):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}") from None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        duration = (frame_count / fps) if fps else None
        return {
            "probe_backend": "opencv",
            "codec": None,
            "width": width,
            "height": height,
            "fps": fps or None,
            "duration_s": duration,
            "nb_frames": frame_count or None,
            "start_time_s": 0.0,
            "rotation": 0,
            "pix_fmt": None,
            "format_name": None,
        }


def _parse_rate(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        if den_f == 0:
            return None
        return float(num) / den_f
    return float(value)


def _to_float(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    return float(value)


def _to_int(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    return int(float(value))


def _enhance_dashcam(frame: np.ndarray) -> np.ndarray:
    """Light contrast stretch that helps later detection on dark dashcam footage."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.merge((clahe.apply(lightness), a_ch, b_ch))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _resize_max_width(frame: np.ndarray, max_width: int | None) -> np.ndarray:
    if not max_width or frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    size = (max_width, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _histogram(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def preprocess_video(
    video_path: str | Path,
    output_dir: str | Path | None = None,
    sample_fps: float | None = 5.0,
    max_width: int = 1280,
    enhance: bool = True,
    jpeg_quality: int = 90,
    scene_threshold: float = 0.45,
) -> dict[str, Any]:
    """Extract timestamped frames and write `video_metadata.json`.

    Times are seconds from video start. Incident-relative alignment is applied later
    using `start_offset_s` (defaults to 0 until P3 sets the incident clock).
    """
    source = Path(video_path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    file_hash = sha256_file(source)
    video_id = f"{source.stem}_{file_hash[:8]}"
    out_root = Path(output_dir) if output_dir else Path("data/processed") / video_id
    frames_dir = out_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    meta = probe_video(source)
    fps = meta["fps"] or 30.0
    start_time_s = meta["start_time_s"] or 0.0
    interval = 1.0 / sample_fps if sample_fps and sample_fps > 0 else None

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {source}")

    opencv_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if opencv_fps and abs(opencv_fps - fps) > 1.0:
        fps = opencv_fps

    frame_records: list[dict[str, Any]] = []
    scene_keyframes: list[dict[str, Any]] = []
    prev_hist: np.ndarray | None = None
    next_keep_time = 0.0
    source_index = 0
    saved_index = 0
    last_time_s = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec and pos_msec > 0:
            time_s = pos_msec / 1000.0
        else:
            time_s = source_index / fps
        last_time_s = time_s

        keep = interval is None or source_index == 0 or time_s + 1e-6 >= next_keep_time
        if keep:
            processed = _resize_max_width(frame, max_width)
            if enhance:
                processed = _enhance_dashcam(processed)

            filename = f"frame_{saved_index:06d}.jpg"
            rel_path = f"frames/{filename}"
            cv2.imwrite(
                str(frames_dir / filename),
                processed,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )

            hist = _histogram(processed)
            scene_score = 0.0
            is_scene_change = saved_index == 0
            if prev_hist is not None:
                scene_score = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                is_scene_change = scene_score >= scene_threshold
            prev_hist = hist

            record = {
                "extracted_index": saved_index,
                "source_frame": source_index,
                "time_s": round(float(time_s), 6),
                "path": rel_path,
                "width": int(processed.shape[1]),
                "height": int(processed.shape[0]),
                "kind": "observation",
                "scene_score": round(scene_score, 4),
                "is_scene_change": is_scene_change,
            }
            frame_records.append(record)
            if is_scene_change:
                scene_keyframes.append(record)
            saved_index += 1
            if interval is not None:
                next_keep_time += interval

        source_index += 1

    cap.release()

    duration_s = meta["duration_s"] if meta["duration_s"] is not None else last_time_s
    metadata = {
        "video_id": video_id,
        "source_path": str(source),
        "source_name": source.name,
        "sha256": file_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probe_backend": meta["probe_backend"],
        "codec": meta["codec"],
        "format_name": meta["format_name"],
        "pix_fmt": meta["pix_fmt"],
        "original_width": meta["width"],
        "original_height": meta["height"],
        "fps": round(float(fps), 6),
        "duration_s": round(float(duration_s or 0.0), 6),
        "source_frame_count": source_index,
        "nb_frames_header": meta["nb_frames"],
        "rotation": meta["rotation"],
        "start_offset_s": 0.0,
        "container_start_time_s": start_time_s,
        "time_convention": (
            "time_s is seconds from video start. Incident time = time_s + start_offset_s. "
            "start_offset_s stays 0 until the incident clock is locked."
        ),
        "sample_fps": sample_fps,
        "extracted_frame_count": saved_index,
        "max_width": max_width,
        "enhance_clahe": enhance,
        "kind": "observation",
        "confidence": 1.0,
        "scene_keyframe_count": len(scene_keyframes),
        "output_dir": str(out_root.resolve()),
    }

    (out_root / "video_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (out_root / "frames_index.json").write_text(
        json.dumps(
            {
                "video_id": video_id,
                "fps": metadata["fps"],
                "start_offset_s": 0.0,
                "frames": frame_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_root / "scene_keyframes.json").write_text(
        json.dumps({"video_id": video_id, "frames": scene_keyframes}, indent=2),
        encoding="utf-8",
    )
    return metadata


def write_synthetic_clip(path: Path, seconds: float = 2.0, fps: float = 10.0) -> Path:
    """Create a tiny labeled clip so preprocessing can be tested without a dashcam file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 360
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write synthetic clip to {path}")

    total = int(seconds * fps)
    for i in range(total):
        t = i / fps
        # Two colored "vehicles" that swap lanes, plus a timestamp overlay.
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (32, 32, 32)
        cv2.line(frame, (0, height // 2), (width, height // 2), (80, 80, 80), 2)
        x_red = int(40 + (width - 120) * (t / seconds))
        y_red = height // 2 - 50
        x_blue = int(width - 80 - (width - 120) * (t / seconds))
        y_blue = height // 2 + 20
        cv2.rectangle(frame, (x_red, y_red), (x_red + 60, y_red + 28), (0, 0, 220), -1)
        cv2.rectangle(frame, (x_blue, y_blue), (x_blue + 60, y_blue + 28), (220, 120, 0), -1)
        cv2.putText(
            frame,
            f"t={t:.2f}s frame={i}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 240, 240),
            2,
        )
        writer.write(frame)
    writer.release()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract timestamped frames from a video.")
    parser.add_argument("video", nargs="?", help="Path to a dashcam or other incident clip")
    parser.add_argument("--out", help="Output directory (default: data/processed/<video_id>)")
    parser.add_argument("--sample-fps", type=float, default=5.0, help="Frames to keep per second (0 = all)")
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--no-enhance", action="store_true")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a 2s test clip and preprocess it",
    )
    args = parser.parse_args()

    if args.synthetic or not args.video:
        clip = Path("data/raw/synthetic_incident.mp4")
        write_synthetic_clip(clip)
        video = clip
        print(f"Wrote synthetic clip: {clip}")
    else:
        video = Path(args.video)

    sample_fps = None if args.sample_fps == 0 else args.sample_fps
    result = preprocess_video(
        video,
        output_dir=args.out,
        sample_fps=sample_fps,
        max_width=args.max_width,
        enhance=not args.no_enhance,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
