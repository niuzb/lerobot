"""Local operator-controlled pose capture; startup and preview are read-only."""

import argparse
import json
import queue
import secrets
import threading
import time
from collections import deque
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from .board import chessboard_points
from .config import JOINTS, write_json
from .pose_control import DirectPoseControl, joint_vector


class PoseSession:
    """Own two read-only cameras and serialize all arm access on one thread."""

    def __init__(self, root, square_mm):
        """Prepare an empty capture session without opening devices."""
        self.root = root
        self.board = {"inner_corners": [9, 6], "square_m": square_mm / 1000}
        chessboard_points(self.board)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.commands = queue.Queue(maxsize=32)
        self.frames = {}
        self.samples = []
        self.history = deque(maxlen=60)
        self.side_reference = None
        self.fault = None
        self.sent_count = 0
        self.torque_owned = False
        self.state = {"armed": False, "following": False, "fault": None, "counts": {"fit": 0, "check": 0}}

    def connect(self):
        """Read existing motor calibration and open cameras without enabling torque."""
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        self.follower = SO101Follower(
            SO101FollowerConfig(
                port="COM5",
                id="zihao_follower_arm",
                use_degrees=True,
                cameras={},
                max_target_velocity=3.0,
                max_tracking_error=3.0,
                target_ramp_max_dt=0.1,
            )
        )
        self.leader = SO101Leader(SO101LeaderConfig(port="COM6", id="zihao_leader_arm", use_degrees=True))
        self.cameras = {}
        preflight = {"motion_commands_sent": False, "board": self.board, "devices": {}}
        for name, device in (("leader", self.leader), ("follower", self.follower)):
            device.bus.connect()
            if not device.is_calibrated:
                raise ValueError(f"{name} calibration mismatch")
            torque = device.bus.sync_read("Torque_Enable", normalize=False)
            mode = device.bus.sync_read("Operating_Mode", normalize=False)
            if any(torque.values()) or any(mode.values()):
                raise ValueError(f"{name}: expected torque off and position mode before startup")
            preflight["devices"][name] = {
                "torque": torque,
                "mode": mode,
                "calibration": {n: vars(c) for n, c in device.calibration.items()},
            }
        extent = [
            (self.follower.calibration[n].range_max - self.follower.calibration[n].range_min) * 180 / 4095
            for n in JOINTS[:5]
        ]
        self.control = DirectPoseControl([-v for v in extent] + [0], extent + [100])
        for name, index in (("front", 0), ("side", 2)):
            for attempt in range(3):
                camera = OpenCVCamera(
                    OpenCVCameraConfig(
                        index_or_path=index, width=640, height=480, fps=30, fourcc="MJPG", backend=700
                    )
                )
                self.cameras[name] = camera
                try:
                    camera.connect()
                    break
                except (OSError, RuntimeError, ConnectionError):
                    if camera.is_connected:
                        camera.disconnect()
                    if attempt == 2:
                        raise
                    time.sleep(2)
        write_json(self.root / "preflight.json", preflight)

    def preview(self):
        """Keep only the latest camera images and detected board corners."""
        while not self.stop.is_set():
            for name, camera in self.cameras.items():
                try:
                    camera.read_latest(max_age_ms=200)
                    # Pair the pixels with their actual driver timestamp, not the detection time.
                    with camera.frame_lock:
                        rgb = camera.latest_frame.copy()
                        timestamp = time.monotonic() - (time.perf_counter() - camera.latest_timestamp)
                    found, corners = cv2.findChessboardCornersSB(
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (9, 6)
                    )
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    if found:
                        cv2.drawChessboardCorners(bgr, (9, 6), corners, True)
                    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ok:
                        raise RuntimeError("JPEG encoding failed")
                    row = {
                        "found": bool(found),
                        "time": timestamp,
                        "rgb": rgb,
                        "points": corners.reshape(-1, 2) if found else None,
                        "jpeg": encoded.tobytes(),
                    }
                except Exception as exc:
                    row = {"found": False, "error": str(exc), "time": time.monotonic()}
                with self.lock:
                    self.frames[name] = row
            self.stop.wait(0.2)

    def capture_ready(self, now):
        """Require fresh complete views and stationary measured joints."""
        with self.lock:
            good_images = len(self.frames) == 2 and all(
                r.get("found") and now - r["time"] < 0.8 for r in self.frames.values()
            )
            oldest_image = min((r["time"] for r in self.frames.values()), default=now)
        recent = [p for t, p in self.history if t >= min(now - 0.8, oldest_image - 0.05)]
        still = len(recent) >= 15 and np.max(np.ptp(recent, axis=0)) < 0.4
        covered = bool(self.history) and self.history[0][0] <= oldest_image
        return bool(good_images and still and covered and self.control.token is None and self.fault is None)

    def capture(self, split, follower):
        """Save observed poses and scale, without claiming known world coordinates."""
        if split not in ("fit", "check") or not self.capture_ready(time.monotonic()):
            raise ValueError("请点击停止跟随、停稳 1 秒，并让两路完整看到棋盘格")
        with self.lock:
            frames = {n: dict(r) for n, r in self.frames.items()}
        for old in self.samples:
            if np.max(np.abs(np.array(old["state"][:5]) - follower[:5])) < 5:
                raise ValueError("姿态与已有记录太接近，请换一个姿态（至少一个关节相差 5°）")
        side = frames["side"]["points"]
        if self.side_reference is not None:
            drift = min(
                np.median(np.linalg.norm(side - self.side_reference, axis=1)),
                np.median(np.linalg.norm(side[::-1] - self.side_reference, axis=1)),
            )
            if drift > 2:
                raise ValueError("外部画面的棋盘位置变化超过 2 像素，请检查板和相机是否松动")
        destination = self.root / f"pose_{len(self.samples):03d}_{split}"
        destination.mkdir(exist_ok=False)
        result = {
            "board": self.board,
            "motion_commands_sent": self.sent_count > 0,
            "source": "direct_leader_angle_teleop",
            "world_registration_validated": False,
            "joint_reference_truth": None,
            "cameras": {},
        }
        for name, row in frames.items():
            path = destination / f"{name}.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(row["rgb"], cv2.COLOR_RGB2BGR)):
                raise RuntimeError("Image save failed")
            state_time, image_state = min(self.history, key=lambda item: abs(item[0] - row["time"]))
            result["cameras"][name] = {
                "view_id": f"{destination.name}_{name}",
                "image": str(path.resolve()),
                "captured_at_unix": time.time() - (time.monotonic() - row["time"]),
                "robot_state": image_state.tolist(),
                "image_state_time_offset_s": state_time - row["time"],
                "board_detected": True,
                "image_points_px": row["points"].tolist(),
                "world_points_m": None,
                "split": split,
                "image_age_at_save_s": time.monotonic() - row["time"],
            }
        write_json(destination / "capture.json", result)
        self.side_reference = side.copy() if self.side_reference is None else self.side_reference
        self.samples.append({"split": split, "state": follower.tolist(), "path": str(destination)})
        write_json(self.root / "samples.json", {"samples": self.samples, "calibration_complete": False})
        return {"message": f"已保存第 {len(self.samples)} 个姿态（{split}）"}

    def handle(self, payload, leader, follower, received):
        """Apply a fresh explicit operator request from the local page."""
        op = payload.get("op")
        if time.monotonic() - received > 0.5:
            raise ValueError("操作请求已过期，请重试")
        if self.fault and op not in ("release", "shutdown", "pause"):
            raise ValueError(self.fault)
        if op == "inspect_gripper":
            if self.torque_owned or self.control.armed:
                raise ValueError("请先托稳从臂并释放电机，再读取量尺参考值")
            calibration = self.follower.calibration["gripper"]
            raw = self.follower.bus.read("Present_Position", "gripper", normalize=False)
            span = calibration.range_max - calibration.range_min
            if span <= 0:
                raise ValueError("夹爪标定范围无效")
            percent = 100 * (raw - calibration.range_min) / span
            if self.follower.bus.apply_drive_mode and calibration.drive_mode:
                percent = 100 - percent
            saturated = raw < calibration.range_min or raw > calibration.range_max
            return {
                "message": (
                    f"夹爪原始值 {raw}；未截断开度 {percent:.2f}%；"
                    f"标定范围 {calibration.range_min}–{calibration.range_max}。"
                    + ("已超出标定范围，不能当作 0%/100% 端点。" if saturated else "当前在标定范围内。")
                ),
                "raw_position": raw,
                "unclipped_percent": percent,
                "outside_calibrated_range": saturated,
                "read_at_unix": time.time(),
            }
        if op == "arm":
            if self.control.armed:
                raise ValueError("已经处于保持状态")
            self.control.arm(follower)
            # Seed exact raw feedback BEFORE torque enable; no EEPROM/configuration writes.
            try:
                raw = self.follower.bus.sync_read("Present_Position", normalize=False)
                if set(raw) != set(JOINTS) or any(
                    not self.follower.calibration[n].range_min
                    <= raw[n]
                    <= self.follower.calibration[n].range_max
                    for n in JOINTS
                ):
                    raise ValueError("Raw feedback is outside calibrated limits")
                self.follower.bus.sync_write("Goal_Position", raw, normalize=False)
                if self.follower.bus.sync_read("Goal_Position", normalize=False) != raw:
                    raise RuntimeError("Current-position goal verification failed; torque remains off")
                self.torque_owned = True
                self.follower.bus.enable_torque()
                if not all(self.follower.bus.sync_read("Torque_Enable", normalize=False).values()):
                    raise RuntimeError("Torque enable verification failed")
            except Exception as exc:
                self.latch_fault(exc)
                raise
            self.follower._ramp_target = None
            self.follower._ramp_time = None
            return {"message": "已保持当前位置"}
        if op == "begin":
            if not self.control.armed:
                self.handle({"op": "arm"}, leader, follower, received)
            return {"token": self.control.begin(leader, follower, time.monotonic())}
        if op == "heartbeat":
            self.control.heartbeat(payload.get("token"), received, time.monotonic())
        elif op == "pause":
            if follower is not None:
                self.control.pause(follower)
            else:
                self.control.token = None
            return {"message": "已停止跟随，保持当前位置"}
        elif op == "capture":
            return self.capture(payload.get("split"), follower)
        elif op == "release":
            self.control.token = None
            if self.torque_owned:
                try:
                    self.follower.bus.disable_torque()
                    if any(self.follower.bus.sync_read("Torque_Enable", normalize=False).values()):
                        raise RuntimeError("Torque release verification failed")
                except Exception as exc:
                    self.latch_fault(exc)
                    raise
                self.torque_owned = False
            self.control.disarm()
            return {"message": "电机已释放，恢复只读"}
        elif op == "shutdown":
            if self.torque_owned:
                raise ValueError("请先托住从臂并点击释放电机，再关闭服务")
            self.stop.set()
            return {"message": "服务已关闭，照片保存在本地"}
        elif op != "heartbeat":
            raise ValueError("Unknown operation")
        return {"message": "ok"}

    def latch_fault(self, exc):
        """Block new movement until restart; explicit torque release remains available."""
        self.fault = str(exc)
        self.control.token = None
        self.history.clear()

    def tick(self, log):
        """Read once, process operator requests and send at most one ramped goal."""
        leader = follower = None
        if self.fault is None:
            try:
                action = self.leader.get_action()
                observation = self.follower.get_observation()
                leader = joint_vector([action[f"{n}.pos"] for n in JOINTS])
                follower = joint_vector([observation[f"{n}.pos"] for n in JOINTS])
                self.history.append((time.monotonic(), follower.copy()))
            except Exception as exc:
                self.latch_fault(exc)
        # Explicit release and shutdown remain available even after a feedback failure.
        for _ in range(8):
            try:
                received, payload, future = self.commands.get_nowait()
            except queue.Empty:
                break
            if not future.set_running_or_notify_cancel():
                continue
            try:
                response = self.handle(payload, leader, follower, received)
                future.set_result(response)
            except Exception as exc:
                future.set_exception(exc)
                continue
            if payload.get("op") != "heartbeat":
                log.write(
                    json.dumps(
                        {"time": time.time(), "operator_command": payload.get("op"), "result": response}
                    )
                    + "\n"
                )
                log.flush()
        if self.fault is None and not self.stop.is_set():
            target = self.control.target(leader, follower, time.monotonic())
            if target is not None:
                sent = self.follower.send_action(
                    {f"{n}.pos": float(v) for n, v in zip(JOINTS, target, strict=True)}
                )
                self.sent_count += 1
                log.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "leader": leader.tolist(),
                            "observed": follower.tolist(),
                            "requested": target.tolist(),
                            "sent": sent,
                            "following": self.control.token is not None,
                        }
                    )
                    + "\n"
                )
                log.flush()
        fit = [s["state"] for s in self.samples if s["split"] == "fit"]
        state = {
            "armed": self.control.armed,
            "torque_owned": self.torque_owned,
            "following": self.control.token is not None,
            "fault": self.fault,
            "follower": follower.tolist() if follower is not None else None,
            "leader": leader.tolist() if leader is not None else None,
            "capture_ready": self.capture_ready(time.monotonic()),
            "sent_count": self.sent_count,
            "counts": {s: sum(r["split"] == s for r in self.samples) for s in ("fit", "check")},
            "fit_span": np.ptp(fit, axis=0).tolist() if fit else [0] * 6,
        }
        with self.lock:
            self.state = state

    def run(self):
        """Run the serial control loop at 30 Hz, independently of camera detection."""
        with (self.root / "control.jsonl").open("x", encoding="utf-8") as log:
            while not self.stop.is_set():
                start = time.monotonic()
                try:
                    self.tick(log)
                except Exception as exc:
                    self.latch_fault(exc)
                    with self.lock:
                        self.state.update(fault=self.fault, following=False, capture_ready=False)
                self.stop.wait(max(0.0, 1 / 30 - (time.monotonic() - start)))

    def close(self):
        """Close devices without dropping an arm that may still be held by its servos."""
        self.stop.set()
        errors = []
        for camera in getattr(self, "cameras", {}).values():
            if camera.is_connected:
                try:
                    camera.disconnect()
                except Exception as exc:
                    errors.append(str(exc))
        for device in (getattr(self, "leader", None), getattr(self, "follower", None)):
            if device is not None and device.bus.is_connected:
                try:
                    device.bus.disconnect(disable_torque=False)
                except Exception as exc:
                    errors.append(str(exc))
        write_json(
            self.root / "closed.json",
            {
                "torque_left_enabled": self.torque_owned,
                "sent_count": self.sent_count,
                "fault": self.fault,
                "cleanup_errors": errors,
                "calibration_complete": False,
            },
        )


class LocalPoseRequestHandler(BaseHTTPRequestHandler):
    """Treat a browser cancelling a response as a closed client, not a server fault."""

    def handle(self):
        """Also tolerate disconnects while parsing requests or flushing responses."""
        try:
            super().handle()
        except ConnectionError:
            self.close_connection = True

    def respond(self, code, body, kind="application/json; charset=utf-8"):
        """Write one response, quietly ending it if the browser has disconnected."""
        if isinstance(body, dict):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)
        except ConnectionError:
            self.close_connection = True
            return False
        return True


def main():
    """Serve the loopback-only operator page, initially in read-only mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="New directory; defaults to a timestamped calibration run"
    )
    parser.add_argument("--square-mm", type=float, required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("outputs/calibration") / time.strftime("%Y%m%d/pose_capture_%H%M%S")
    cv2.setNumThreads(2)
    chessboard_points({"inner_corners": [9, 6], "square_m": args.square_mm / 1000})
    args.output.mkdir(parents=True, exist_ok=False)
    session = PoseSession(args.output.resolve(), args.square_mm)
    key = secrets.token_urlsafe(32)
    html_path = Path(__file__).with_suffix(".html")
    if not html_path.is_file():
        raise FileNotFoundError(html_path)
    hosts = {f"127.0.0.1:{args.port}", f"localhost:{args.port}"}

    class Handler(LocalPoseRequestHandler):
        def allowed(self):
            return self.headers.get("Host") in hosts and self.headers.get("Sec-Fetch-Site") != "cross-site"

        def do_GET(self):
            if not self.allowed():
                return self.respond(403, {"message": "Local origin required"})
            path = self.path.split("?")[0]
            if path == "/":
                html = html_path.read_text(encoding="utf-8").replace("__SESSION_KEY__", key)
                return self.respond(200, html.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/status":
                with session.lock:
                    data = dict(session.state)
                    data["cameras"] = {
                        n: {k: v for k, v in r.items() if k in ("found", "error", "time")}
                        for n, r in session.frames.items()
                    }
                return self.respond(200, data)
            if path in ("/image/front", "/image/side"):
                with session.lock:
                    jpeg = session.frames.get(path.rsplit("/", 1)[1], {}).get("jpeg", b"")
                return self.respond(200 if jpeg else 503, jpeg, "image/jpeg")
            self.respond(404, {"message": "Not found"})

        def do_POST(self):
            if (
                not self.allowed()
                or self.headers.get("X-Pose-Session") != key
                or self.headers.get("Origin") not in {f"http://{h}" for h in hosts}
            ):
                return self.respond(403, {"message": "Local session and origin required"})
            if self.path != "/command":
                return self.respond(404, {"message": "Not found"})
            future = Future()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024:
                    raise ValueError("Invalid request length")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected object")
                session.commands.put_nowait((time.monotonic(), payload, future))
                result = future.result(timeout=2)
                self.respond(200, result)
            except Exception as exc:
                future.cancel()
                self.respond(422, {"message": str(exc) or "操作超时"})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threads = []
    try:
        session.connect()
        for work in (session.preview, session.run):
            thread = threading.Thread(target=work, daemon=True)
            thread.start()
            threads.append(thread)
        server.timeout = 0.5
        print(f"POSE_CAPTURE_READY http://127.0.0.1:{args.port} READ_ONLY", flush=True)
        print(f"CAPTURE_DIRECTORY {args.output.resolve()}", flush=True)
        while not session.stop.is_set():
            server.handle_request()
    finally:
        session.stop.set()
        for thread in threads:
            thread.join(timeout=3)
        session.close()
        server.server_close()


if __name__ == "__main__":
    main()
