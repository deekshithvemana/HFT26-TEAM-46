
import json
import cv2
import argparse
from pathlib import Path


def extract_keyframes(tracks_dir, output_dir):

    tracks_dir = Path(tracks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata and collision candidates
    with open(tracks_dir / "tracks.json") as f:
        track_data = json.load(f)

    with open(tracks_dir / "collision_candidates.json") as f:
        collision_data = json.load(f)

    video_path = track_data["source_path"]
    fps = float(track_data["fps"])

    # Main vehicles identified by participant analysis
    target_vehicles = {2, 4}

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    selected_frames = set()

    for candidate in collision_data.get("candidates", []):

        vehicle_ids = set(candidate.get("vehicle_ids", []))

        # Prioritise candidates involving the main vehicles
        if not vehicle_ids.intersection(target_vehicles):
            continue

        supporting_frames = candidate.get("supporting_frames", [])

        if not supporting_frames:
            continue

        start = min(supporting_frames)
        middle = supporting_frames[len(supporting_frames) // 2]
        end = max(supporting_frames)

        # Before, during and after the interaction
        selected_frames.update([
            max(0, start - 3),
            start,
            middle,
            end,
            min(total_frames - 1, end + 3)
        ])

    manifest = []

    for frame_index in sorted(selected_frames):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()

        if not success:
            print(f"Could not read frame {frame_index}")
            continue

        time_s = frame_index / fps

        # Add frame information
        label = f"Frame {frame_index} | {time_s:.2f}s"

        cv2.putText(
            frame,
            label,
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        filename = f"frame_{frame_index:04d}.jpg"
        output_path = output_dir / filename

        cv2.imwrite(str(output_path), frame)

        manifest.append({
            "frame_index": frame_index,
            "time_s": round(time_s, 3),
            "filename": filename
        })

    cap.release()

    manifest_path = output_dir / "keyframes.json"

    with open(manifest_path, "w") as f:
        json.dump({
            "video_id": track_data.get("video_id", "unknown"),
            "source_video": video_path,
            "fps": fps,
            "total_keyframes": len(manifest),
            "keyframes": manifest
        }, f, indent=2)

    print("\nKeyframe extraction completed!")
    print("Frames extracted:", len(manifest))
    print("Output directory:", output_dir)
    print("Manifest:", manifest_path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tracks-dir",
        default="data/processed/accident_65ea913a"
    )

    parser.add_argument(
        "--output-dir",
        default="data/processed/accident_65ea913a/keyframes"
    )

    args = parser.parse_args()

    extract_keyframes(args.tracks_dir, args.output_dir)


if __name__ == "__main__":
    main()
