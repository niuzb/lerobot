"""Read-only recording audit. Exit nonzero for invalid or suspicious frozen states.

Usage: python check_recorded_dataset.py PATH_TO_LOCAL_DATASET
This checks numerical consistency, not visual task success or camera placement.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit_frames(frame: pd.DataFrame, fps: float, freeze_seconds: float = 2.0) -> dict:
    """Flag non-finite values and frozen feedback paired with varying commands."""
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(freeze_seconds) or freeze_seconds <= 0:
        raise ValueError("fps and freeze_seconds must be finite and positive")
    issues = []
    if frame.empty:
        issues.append({"reason": "No recorded frames"})
    for episode, rows in frame.groupby("episode_index", sort=True):
        rows = rows.sort_values("frame_index")
        state = np.stack(rows["observation.state"]).astype(float)
        action = np.stack(rows["action"]).astype(float)
        if state.ndim != 2 or state.shape != action.shape:
            raise ValueError("Expected matching two-dimensional state/action arrays")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            issues.append({"episode": int(episode), "reason": "Non-finite state/action"})
            continue
        # Exact repeats detect cached poses without treating small real movement as frozen.
        boundaries = np.r_[0, np.flatnonzero(np.any(np.diff(state, axis=0) != 0, axis=1)) + 1, len(state)]
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            duration = (end - start - 1) / fps
            action_span = np.ptp(action[start:end], axis=0)
            if duration >= freeze_seconds and np.any(action_span > 5.0):
                issues.append(
                    {
                        "episode": int(episode),
                        "start_frame": int(rows.iloc[start].frame_index),
                        "frames": int(end - start),
                        "seconds": float(duration),
                        "reason": "Exactly frozen state while action changes by >5 units",
                        "action_span": action_span.tolist(),
                    }
                )
    return {
        "frames": len(frame),
        "episodes": int(frame.episode_index.nunique()),
        "passed": not issues,
        "issues": issues,
    }


def main() -> None:
    """Audit a finalized local dataset without modifying it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--freeze-seconds", type=float, default=2.0)
    args = parser.parse_args()
    info = json.loads((args.root / "meta/info.json").read_text(encoding="utf-8"))
    paths = sorted((args.root / "data").rglob("*.parquet"))
    if not paths:
        parser.error("No data parquet files found; finish/finalize recording first")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    report = audit_frames(frame, info["fps"], args.freeze_seconds)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
