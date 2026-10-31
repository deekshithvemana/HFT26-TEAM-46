
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


def identify_participants(collision_data, motion_data):

    events = motion_data.get("events", [])
    collision_candidates = collision_data.get("candidates", [])

    vehicle_scores = defaultdict(float)
    vehicle_evidence = defaultdict(list)
    pair_results = []

    for candidate in collision_candidates:

        vehicle_ids = candidate.get("vehicle_ids", [])

        if len(vehicle_ids) < 2:
            continue

        start = float(candidate.get("start_time_s", 0))
        end = float(candidate.get("end_time_s", start))

        peak_iou = float(candidate.get("peak_iou") or 0)
        supporting = int(candidate.get("supporting_detections") or 0)

        # Find motion events near this collision candidate.
        matching_events = []

        for event in events:
            track_id = event.get("track_id")
            event_time = event.get("time_s")

            if track_id not in vehicle_ids or event_time is None:
                continue

            event_time = float(event_time)

            if start - 0.5 <= event_time <= end + 0.5:
                matching_events.append(event)

        # Transparent candidate score.
        iou_score = min(peak_iou / 0.5, 1.0) * 60
        detection_score = min(supporting / 10, 1.0) * 25
        motion_score = min(len(matching_events) / 4, 1.0) * 15

        total_score = round(
            iou_score + detection_score + motion_score, 2
        )

        pair_results.append({
            "vehicle_ids": vehicle_ids,
            "start_time_s": start,
            "end_time_s": end,
            "peak_iou": peak_iou,
            "supporting_detections": supporting,
            "matching_motion_events": len(matching_events),
            "candidate_score": total_score
        })

        # Give both vehicles evidence from the interaction.
        for vehicle_id in vehicle_ids:

            vehicle_scores[vehicle_id] += total_score / len(vehicle_ids)

            vehicle_evidence[vehicle_id].append({
                "evidence_type": "collision_candidate",
                "other_vehicles": [
                    v for v in vehicle_ids if v != vehicle_id
                ],
                "time_window": [start, end],
                "peak_iou": peak_iou,
                "supporting_detections": supporting
            })

        # Add motion evidence to the corresponding vehicle.
        for event in matching_events:

            vehicle_id = event["track_id"]
            vehicle_scores[vehicle_id] += 5

            vehicle_evidence[vehicle_id].append({
                "evidence_type": "motion_event",
                "event_type": event.get("event_type"),
                "time_s": event.get("time_s"),
                "frame_index": event.get("frame_index")
            })

    ranked_vehicles = []

    for vehicle_id, score in vehicle_scores.items():

        ranked_vehicles.append({
            "track_id": vehicle_id,
            "involvement_score": round(score, 2),
            "evidence_count": len(vehicle_evidence[vehicle_id]),
            "evidence": vehicle_evidence[vehicle_id]
        })

    ranked_vehicles.sort(
        key=lambda item: item["involvement_score"],
        reverse=True
    )

    return {
        "video_id": collision_data.get("video_id", "unknown"),
        "total_collision_candidates": len(pair_results),
        "potentially_involved_vehicles": ranked_vehicles,
        "collision_pair_analysis": pair_results,
        "note": (
            "Vehicles are ranked using motion and bounding-box "
            "interaction evidence. These scores are heuristic and "
            "do not establish collision, causation, or fault."
        )
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tracks-dir",
        default="data/processed/accident_65ea913a"
    )

    parser.add_argument(
        "--output",
        default=None
    )

    args = parser.parse_args()

    directory = Path(args.tracks_dir)

    collision_data = load_json(
        directory / "collision_candidates.json"
    )

    motion_data = load_json(
        directory / "video_events.json"
    )

    result = identify_participants(
        collision_data,
        motion_data
    )

    output_path = (
        Path(args.output)
        if args.output
        else directory / "participant_candidates.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as file:
        json.dump(result, file, indent=2)

    print("\nParticipant identification completed!")
    print(
        "Collision candidates:",
        result["total_collision_candidates"]
    )

    print("\nRanked vehicles:")

    for vehicle in result["potentially_involved_vehicles"]:
        print(
            f"Track {vehicle['track_id']} | "
            f"Score: {vehicle['involvement_score']} | "
            f"Evidence: {vehicle['evidence_count']}"
        )

    print("\nOutput:", output_path)


if __name__ == "__main__":
    main()
