
import json
import argparse
from pathlib import Path
from collections import defaultdict


def calculate_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def detect_collision_candidates(data, iou_threshold=0.02, min_frames=3):
    pair_observations = defaultdict(list)

    for frame_data in data.get("frames", []):
        frame_number = frame_data.get("frame")
        time_s = frame_data.get("time_s")

        objects = frame_data.get("objects", [])

        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                obj_a = objects[i]
                obj_b = objects[j]

                id_a = obj_a.get("track_id")
                id_b = obj_b.get("track_id")

                if id_a is None or id_b is None or id_a == id_b:
                    continue

                if obj_a.get("class") not in ("car", "truck", "bus", "motorcycle"):
                    continue

                if obj_b.get("class") not in ("car", "truck", "bus", "motorcycle"):
                    continue

                overlap = calculate_iou(
                    obj_a.get("bbox", []),
                    obj_b.get("bbox", [])
                )

                if overlap >= iou_threshold:
                    pair = tuple(sorted((id_a, id_b)))

                    pair_observations[pair].append({
                        "frame_index": frame_number,
                        "time_s": time_s,
                        "iou": round(overlap, 4)
                    })

    candidates = []

    for (id_a, id_b), observations in pair_observations.items():
        observations.sort(key=lambda x: x["frame_index"])

        # Split observations into continuous groups.
        groups = []
        current_group = []

        for observation in observations:
            if not current_group:
                current_group.append(observation)
            elif observation["frame_index"] - current_group[-1]["frame_index"] <= 2:
                current_group.append(observation)
            else:
                groups.append(current_group)
                current_group = [observation]

        if current_group:
            groups.append(current_group)

        for group in groups:
            if len(group) < min_frames:
                continue

            candidates.append({
                "event_type": "possible_vehicle_collision_interaction",
                "vehicle_ids": [id_a, id_b],
                "start_time_s": group[0]["time_s"],
                "end_time_s": group[-1]["time_s"],
                "start_frame": group[0]["frame_index"],
                "end_frame": group[-1]["frame_index"],
                "supporting_frames": [
                    item["frame_index"] for item in group
                ],
                "supporting_detections": len(group),
                "peak_iou": max(item["iou"] for item in group),
                "confidence": "heuristic_candidate"
            })

    return {
        "video_id": data.get("video_id", "unknown"),
        "total_candidates": len(candidates),
        "candidates": candidates,
        "note": (
            "Bounding-box overlap identifies possible vehicle interactions. "
            "It does not independently confirm physical collision or fault."
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to tracks.json")
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    input_path = Path(args.input)

    with input_path.open() as file:
        data = json.load(file)

    result = detect_collision_candidates(data)

    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent / "collision_candidates.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as file:
        json.dump(result, file, indent=2)

    print("Collision candidate analysis completed!")
    print("Candidates found:", result["total_candidates"])
    print("Output:", output_path)

    for candidate in result["candidates"]:
        print(
            "Vehicles:", candidate["vehicle_ids"],
            "| Time:", candidate["start_time_s"],
            "-", candidate["end_time_s"],
            "| Peak IoU:", candidate["peak_iou"]
        )


if __name__ == "__main__":
    main()
