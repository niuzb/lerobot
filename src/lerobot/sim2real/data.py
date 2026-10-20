"""Read source trajectories without rewriting source data or metadata."""

import json
from pathlib import Path

import numpy as np

from .config import JOINTS, TASK, TRAIN, VALIDATION, digest, write_json


class SourceData:
    """Audited real trajectories with a fixed episode-level train/validation split."""

    def __init__(self, config):
        """Read parquet directly; no Hub download, metadata rewrite or recording."""
        import pyarrow.parquet as pq

        self.root = Path(config["dataset_root"])
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info["fps"] != 30 or self.info["total_episodes"] != 88 or self.info["total_frames"] != 27632:
            raise ValueError("Expected the audited 88-episode / 27632-frame source dataset")
        for name in ("action", "observation.state"):
            if self.info["features"][name]["names"] != [f"{joint}.pos" for joint in JOINTS]:
                raise ValueError(f"Unexpected joint order for {name}")
        self.paths = sorted((self.root / "data").rglob("*.parquet"))
        self.table = pq.read_table(self.paths).sort_by("index")
        if self.table.num_rows != 27632:
            raise ValueError("Incomplete source data")
        self.ep = np.asarray(self.table["episode_index"])
        self.state = np.asarray(self.table["observation.state"].to_pylist(), dtype=np.float32)
        self.action = np.asarray(self.table["action"].to_pylist(), dtype=np.float32)
        if not np.isfinite(self.state).all() or not np.isfinite(self.action).all():
            raise ValueError("Source contains non-finite values")
        if set(self.ep.tolist()) != set(TRAIN + VALIDATION):
            raise ValueError("Unexpected source episode IDs")

    def episode(self, index):
        """Return aligned state and leader-target arrays from one complete episode."""
        indices = np.flatnonzero(self.ep == index)
        if not len(indices):
            raise ValueError(f"No episode {index}")
        rows = self.table.take(indices).to_pydict()
        expected = np.arange(len(indices)) / 30
        if not np.allclose(rows["timestamp"], expected, atol=1e-4):
            raise ValueError("Source timestamps do not match 30 Hz")
        return self.state[indices], self.action[indices]

    def manifest(self):
        """Record split, hashes and source action semantics."""
        files = sorted(
            p for folder in ("meta", "data", "videos") for p in (self.root / folder).rglob("*") if p.is_file()
        )
        return {
            "root": str(self.root),
            "episodes": 88,
            "frames": 27632,
            "train_episodes": TRAIN,
            "validation_episodes": VALIDATION,
            "source_sha256": {str(p.relative_to(self.root)): digest(p) for p in files},
            "action_semantics": "Recorded processed leader target; not guaranteed to be driver sent target",
        }


def export_episode(dataset, trajectory):
    """Only pre-action observations paired with the action applied next are exported."""
    for row, frames in zip(trajectory["rows"], trajectory["frames"], strict=True):
        dataset.add_frame(
            {
                "observation.state": np.asarray(row["state"], dtype=np.float32),
                "action": np.asarray(row["sent_action"], dtype=np.float32),
                "observation.images.front": frames["front"],
                "observation.images.side": frames["side"],
                "task": TASK,
            }
        )
    dataset.save_episode()


def new_dataset(path, repo_id):
    """Create an independent LeRobot v3 dataset with dual RGB video streams."""
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        name: {"dtype": "float32", "shape": (6,), "names": [f"{joint}.pos" for joint in JOINTS]}
        for name in ("observation.state", "action")
    }
    for camera in ("front", "side"):
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=path,
        fps=30,
        robot_type="so101_follower",
        features=features,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18),
        image_writer_threads=8,
    )


def save_attempt(out, index, trajectory):
    """Keep failed attempts as evidence; they never become successful BC labels."""
    import cv2

    write_json(out / f"attempt_{index:04d}.json", {k: v for k, v in trajectory.items() if k != "frames"})
    if not trajectory["frames"]:
        return  # IK can fail before the first physical tick; the JSON preserves why.
    writer = cv2.VideoWriter(
        str(out / f"attempt_{index:04d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 480)
    )
    if not writer.isOpened():
        raise RuntimeError("Unable to create evidence video")
    try:
        for frames in trajectory["frames"]:
            writer.write(
                cv2.cvtColor(np.concatenate([frames["front"], frames["side"]], axis=1), cv2.COLOR_RGB2BGR)
            )
    finally:
        writer.release()
