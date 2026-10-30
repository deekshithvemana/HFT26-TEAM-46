import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.video.preprocess import preprocess_video, write_synthetic_clip


def test_preprocess_synthetic(tmp_path: Path) -> None:
    clip = write_synthetic_clip(tmp_path / "clip.mp4", seconds=1.0, fps=10.0)
    out = tmp_path / "processed"
    meta = preprocess_video(clip, output_dir=out, sample_fps=5.0, enhance=False)

    assert meta["extracted_frame_count"] >= 5
    assert (out / "video_metadata.json").exists()
    index = json.loads((out / "frames_index.json").read_text(encoding="utf-8"))
    times = [row["time_s"] for row in index["frames"]]
    assert times[0] == 0.0
    assert times[-1] >= 0.8
    first = out / index["frames"][0]["path"]
    assert first.exists()
