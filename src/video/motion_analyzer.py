import json
import math
import argparse
from pathlib import Path
from collections import defaultdict


def analyze_motion(input_path, output_path):
    data = json.loads(Path(input_path).read_text())

    frames = data["frames"]
    histories = defaultdict(list)

    for frame_index, frame in enumerate(frames):
        time_s = float(frame.get("time_s", frame_index / 5.0))

        for obj in frame.get("objects", []):
            track_id = int(obj["track_id"])
            cx, cy = map(float, obj["centre"])

            histories[track_id].append({
                "time_s": time_s,
                "frame_index": frame_index,
                "x": cx,
                "y": cy,
                "class": obj["class"],
                "confidence": float(obj["conf"])
            })

    trajectories = []

    for track_id, points in sorted(histories.items()):
        points.sort(key=lambda p: p["time_s"])

        for i, point in enumerate(points):
            if i == 0:
                point.update({
                    "dx": 0.0,
                    "dy": 0.0,
                    "displacement_px": 0.0,
                    "speed_px_s": None,
                    "direction": "unknown"
                })
                continue

            previous = points[i - 1]
            dt = point["time_s"] - previous["time_s"]

            dx = point["x"] - previous["x"]
            dy = point["y"] - previous["y"]
            distance = math.hypot(dx, dy)

            speed = distance / dt if dt > 0 else None

            if distance < 1:
                direction = "stationary"
            else:
                angle = math.degrees(math.atan2(-dy, dx))

                if -22.5 <= angle < 22.5:
                    direction = "right"
                elif 22.5 <= angle < 67.5:
                    direction = "up-right"
                elif 67.5 <= angle < 112.5:
                    direction = "up"
                elif 112.5 <= angle < 157.5:
                    direction = "up-left"
                elif angle >= 157.5 or angle < -157.5:
                    direction = "left"
                elif -157.5 <= angle < -112.5:
                    direction = "down-left"
                elif -112.5 <= angle < -67.5:
                    direction = "down"
                else:
                    direction = "down-right"

            point.update({
                "dx": round(dx, 2),
                "dy": round(dy, 2),
                "displacement_px": round(distance, 2),
                "speed_px_s": round(speed, 2) if speed is not None else None,
                "direction": direction
            })

        trajectories.append({
            "track_id": track_id,
            "class": points[0]["class"],
            "observations": len(points),
            "trajectory": points
        })

    result = {
        "video_id": data.get("video_id", "unknown"),
        "measurement_note": (
            "Speed is measured in pixels per second, not km/h. "
            "Camera calibration is required for real-world speed."
        ),
        "total_tracks": len(trajectories),
        "trajectories": trajectories
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2))

    print("Motion analysis completed!")
    print("Tracks analyzed:", len(trajectories))
    print("Output:", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to tracks.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    output = args.out or str(
        Path(args.input).with_name("trajectories.json")
    )

    analyze_motion(args.input, output)
