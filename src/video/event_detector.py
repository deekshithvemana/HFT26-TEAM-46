
import json
import argparse
from pathlib import Path
from math import hypot, degrees, acos
from statistics import median


def angle_between(a, b):
    mag_a = hypot(*a)
    mag_b = hypot(*b)

    if mag_a == 0 or mag_b == 0:
        return 0.0

    dot = a[0] * b[0] + a[1] * b[1]
    cosine = max(-1, min(1, dot / (mag_a * mag_b)))

    return degrees(acos(cosine))


def smooth_trajectory(observations, radius=2):
    """Apply a local median filter to vehicle centre coordinates."""
    result = []

    for i, obs in enumerate(observations):
        start = max(0, i - radius)
        end = min(len(observations), i + radius + 1)

        window = observations[start:end]

        item = dict(obs)
        item["_smooth_x"] = median(float(p["x"]) for p in window)
        item["_smooth_y"] = median(float(p["y"]) for p in window)

        result.append(item)

    return result


def group_events(events, gap=0.5):
    """Merge repeated events of the same type for the same vehicle."""
    if not events:
        return []

    events.sort(key=lambda e: (e["track_id"], e["event_type"], e["time_s"]))

    grouped = []

    for event in events:
        if (
            grouped
            and grouped[-1]["track_id"] == event["track_id"]
            and grouped[-1]["event_type"] == event["event_type"]
            and event["time_s"] - grouped[-1]["end_time_s"] <= gap
        ):
            previous = grouped[-1]
            previous["end_time_s"] = event["time_s"]
            previous["supporting_detections"] += 1
            previous["frame_indices"].append(event["frame_index"])
        else:
            grouped.append({
                "event_type": event["event_type"],
                "track_id": event["track_id"],
                "time_s": event["time_s"],
                "end_time_s": event["time_s"],
                "frame_index": event["frame_index"],
                "frame_indices": [event["frame_index"]],
                "supporting_detections": 1,
                "confidence": "heuristic_candidate"
            })

    return grouped


def detect_events(data):
    candidates = []

    for vehicle in data.get("trajectories", []):
        track_id = vehicle.get("track_id", -1)
        observations = vehicle.get("trajectory", [])

        if not isinstance(observations, list) or len(observations) < 7:
            continue

        observations = sorted(
            observations,
            key=lambda o: float(o.get("time_s") or 0)
        )

        observations = [
            o for o in observations
            if o.get("x") is not None
            and o.get("y") is not None
            and o.get("time_s") is not None
        ]

        if len(observations) < 7:
            continue

        points = smooth_trajectory(observations, radius=2)

        # Use separated points to reduce frame-to-frame noise.
        for i in range(2, len(points) - 2):
            before = points[i - 2]
            current = points[i]
            after = points[i + 2]

            t_before = float(before["time_s"])
            t_current = float(current["time_s"])
            t_after = float(after["time_s"])

            dt_before = t_current - t_before
            dt_after = t_after - t_current

            if dt_before <= 0 or dt_after <= 0:
                continue

            vector_a = (
                current["_smooth_x"] - before["_smooth_x"],
                current["_smooth_y"] - before["_smooth_y"]
            )

            vector_b = (
                after["_smooth_x"] - current["_smooth_x"],
                after["_smooth_y"] - current["_smooth_y"]
            )

            distance_a = hypot(*vector_a)
            distance_b = hypot(*vector_b)

            # Ignore tiny movements.
            if distance_a < 8 or distance_b < 8:
                continue

            speed_before = distance_a / dt_before
            speed_after = distance_b / dt_after

            angle = angle_between(vector_a, vector_b)

            # Candidate direction change.
            if angle >= 65:
                candidates.append({
                    "event_type": "possible_direction_change",
                    "track_id": track_id,
                    "time_s": round(t_current, 3),
                    "frame_index": current.get("frame_index"),
                    "angle_change_degrees": round(angle, 2)
                })

            # Candidate deceleration.
            if (
                speed_before > 150
                and speed_after < speed_before * 0.5
            ):
                candidates.append({
                    "event_type": "possible_sudden_deceleration",
                    "track_id": track_id,
                    "time_s": round(t_current, 3),
                    "frame_index": current.get("frame_index"),
                    "speed_before_px_s": round(speed_before, 2),
                    "speed_after_px_s": round(speed_after, 2)
                })

    events = group_events(candidates)

    return {
        "video_id": data.get("video_id", "unknown"),
        "total_events": len(events),
        "events": events,
        "method": "median-smoothed multi-frame trajectory analysis",
        "note": (
            "Events are candidate motion anomalies, not confirmed "
            "collisions or proof of fault. Speeds are in pixels/second."
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to trajectories.json")
    parser.add_argument("--output", default=None)

    args = parser.parse_args()
    input_path = Path(args.input)

    with input_path.open("r") as file:
        data = json.load(file)

    result = detect_events(data)

    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent / "video_events.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as file:
        json.dump(result, file, indent=2)

    print("Event detection completed!")
    print("Vehicles analysed:", len(data.get("trajectories", [])))
    print("Grouped events:", result["total_events"])
    print("Output:", output_path)


if __name__ == "__main__":
    main()

