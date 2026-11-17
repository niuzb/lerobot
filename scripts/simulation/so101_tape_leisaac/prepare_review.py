"""Create browser-playable paired-camera clips for review, without training."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from common import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--framing-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rendering = json.loads((args.render / "render_manifest.json").read_text())
    framing = json.loads(args.framing_audit.read_text())
    if not rendering["passed"] or rendering["stride"] != 1 or not framing["geometric_framing_passed"]:
        raise ValueError("Require full-rate rendering and complete object framing")
    if rendering["camera_config_sha256"] != framing["camera_config_sha256"]:
        raise ValueError("Framing audit and rendered cameras differ")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg must be installed and available on PATH")
    args.output.mkdir(parents=True, exist_ok=False)
    rows, episodes = [], []
    for number, episode in enumerate(rendering["episodes"], 1):
        prefix = f"attempt_{episode['attempt']:04d}"
        source = args.render / prefix / "preview.mp4"
        if sha256(source) != episode["files_sha256"]["preview.mp4"]:
            raise ValueError("Rendered video changed")
        output = args.output / f"episode_{number:02d}_dual_camera.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-i",
                str(source),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
        arrays = np.load(args.physics / f"{prefix}.npz")
        carry = np.flatnonzero(arrays["stage"] == 4)
        selected = {
            0: "start",
            int(carry[0]) if len(carry) else episode["frames"] // 2: "carry",
            episode["frames"] - 1: "released",
        }
        cap = cv2.VideoCapture(str(output))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or abs(fps - 30) > 0.001:
            raise ValueError("Review video is not 30 FPS")
        frames, tiles, border_fractions = 0, [], []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != (480, 1280, 3):
                raise ValueError("Review video dimensions changed")
            if frames % 30 == 0:
                side = frame[:, 640:]
                border = np.concatenate(
                    (
                        side[:4].reshape(-1, 3),
                        side[-4:].reshape(-1, 3),
                        side[:, :4].reshape(-1, 3),
                        side[:, -4:].reshape(-1, 3),
                    )
                )
                border_fractions.append(float(np.mean(border.max(axis=1) < 5)))
            if frames in selected:
                tile = np.full((268, 640, 3), 240, np.uint8)
                tile[28:] = cv2.resize(frame, (640, 240), interpolation=cv2.INTER_AREA)
                cv2.putText(
                    tile,
                    f"Video {number} | {selected[frames]} | {frames / fps:.1f}s",
                    (12, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (30, 30, 30),
                    1,
                    cv2.LINE_AA,
                )
                tiles.append(tile)
            frames += 1
        cap.release()
        if frames != episode["frames"] or len(tiles) != 3:
            raise ValueError("Review video has missing frames")
        rows.append(np.concatenate(tiles, axis=1))
        episodes.append(
            {
                "video": output.name,
                "attempt": episode["attempt"],
                "seed": episode["seed"],
                "frames": frames,
                "fps": fps,
                "duration_s": frames / fps,
                "codec": "H264/yuv420p",
                "external_black_border_fraction": float(np.mean(border_fractions)),
                "sha256": sha256(output),
            }
        )
        print(json.dumps({"video_ready": output.name, "frames": frames}), flush=True)
    cv2.imwrite(
        str(args.output / "contact_sheet.jpg"), np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    write_json(
        args.output / "review_manifest.json",
        {
            "passed": True,
            "videos": episodes,
            "camera_config_sha256": rendering["camera_config_sha256"],
            "physical_source": str(args.physics),
            "render_source": str(args.render),
            "human_review_pending": True,
            "training_requested": False,
            "gpu_release_requested": False,
            "calibration_passed": False,
        },
    )


if __name__ == "__main__":
    main()
