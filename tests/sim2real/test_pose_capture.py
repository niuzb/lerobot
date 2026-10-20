"""No hardware: direct following, expiry, fault handling and capture provenance."""

import io
import json
import re
import socket
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from lerobot.sim2real.config import JOINTS
from lerobot.sim2real.pose_capture import LocalPoseRequestHandler, PoseSession, main
from lerobot.sim2real.pose_control import DirectPoseControl


def controller():
    return DirectPoseControl([-180] * 5 + [0], [180] * 5 + [100])


class DirectFollowTests(unittest.TestCase):
    def test_absolute_angles_and_gripper_follow_without_relative_offset(self):
        c = controller()
        leader = np.array([-80, -100, 90, 80, 0, 2.0])
        follower = np.array([4, -71, 20, 99, 13, 30.0])
        self.assertIsNone(c.target(leader, follower, 0))
        with self.assertRaises(ValueError):
            c.begin(leader, follower, 0)
        c.arm(follower)
        c.begin(leader, follower, 0)
        np.testing.assert_array_equal(c.target(leader, follower, 0.01), leader)
        np.testing.assert_array_equal(c.target(leader + 5, follower, 0.02), leader + 5)
        c.pause(follower + 1)
        c.begin(leader + 90, follower + 1, 0.03)
        np.testing.assert_array_equal(c.target(leader + 90, follower + 1, 0.04), leader + 90)

    def test_expiry_and_stalled_loop_hold_measured_pose_cannot_revive_old_token(self):
        for delay in (0.21, 0.81):
            c = controller()
            c.arm([0] * 6)
            token = c.begin([0] * 6, [0] * 6, 1)
            np.testing.assert_array_equal(c.target([10] * 6, [2] * 6, 1 + delay), [2] * 6)
            with self.assertRaises(ValueError):
                c.heartbeat(token, 1 + delay, 1 + delay)

    def test_stale_heartbeat_joint_bounds_and_continuous_follow_beyond_twenty_seconds(self):
        c = controller()
        c.arm([179] * 5 + [30])
        token = c.begin([0] * 6, [179] * 5 + [30], 0)
        self.assertEqual(c.target([190] * 6, [179] * 5 + [30], 0.01)[0], 180)
        with self.assertRaises(ValueError):
            c.heartbeat(token, -1, 0.1)
        for i in range(1, 250):
            t = i / 10
            c.heartbeat(token, t, t)
            c.target([0] * 6, [179] * 5 + [30], t)
        self.assertIsNotNone(c.token)
        with self.assertRaises(ValueError):
            DirectPoseControl([1] * 6, [0] * 6)
        with self.assertRaises(ValueError):
            c.arm([float("nan")] * 6)


class FakeBus:
    def __init__(self):
        self.events = []
        self.goal = None
        self.torque = 0
        self.bad_goal = False

    def sync_read(self, name, **kwargs):
        if name == "Present_Position":
            return dict.fromkeys(JOINTS, 2000)
        if name == "Goal_Position":
            return {} if self.bad_goal else self.goal.copy()
        if name == "Torque_Enable":
            return dict.fromkeys(JOINTS, self.torque)
        raise AssertionError(name)

    def sync_write(self, name, value, **kwargs):
        self.events.append(name)
        self.goal = value.copy()

    def enable_torque(self):
        self.events.append("enable")
        self.torque = 1

    def disable_torque(self):
        self.events.append("disable")
        self.torque = 0


class FakeArm:
    def __init__(self):
        self.bus = FakeBus()
        self.calibration = {n: SimpleNamespace(range_min=0, range_max=4095) for n in JOINTS}
        self.fail = False
        self.sent = []

    def get_action(self):
        if self.fail:
            raise RuntimeError("feedback unavailable")
        return dict.fromkeys([f"{n}.pos" for n in JOINTS], 0)

    get_observation = get_action

    def send_action(self, target):
        self.sent.append(target)
        return target


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.s = PoseSession(Path(self.temp.name), 17)
        self.s.control = controller()
        self.s.leader, self.s.follower = FakeArm(), FakeArm()
        self.log = io.StringIO()

    def command(self, op, **kwargs):
        f = Future()
        self.s.commands.put((time.monotonic(), {"op": op, **kwargs}, f))
        self.s.tick(self.log)
        return f.result()

    def test_readonly_never_writes_and_cancelled_arm_never_executes(self):
        for _ in range(5):
            self.s.tick(self.log)
        future = Future()
        future.cancel()
        self.s.commands.put((time.monotonic(), {"op": "arm"}, future))
        self.s.tick(self.log)
        self.assertEqual(self.s.follower.bus.events, [])
        self.assertEqual(self.s.follower.sent, [])
        self.assertEqual(self.s.leader.bus.events, [])

    def test_raw_gripper_inspection_exposes_clipped_opening_without_writes(self):
        self.s.follower.calibration["gripper"] = SimpleNamespace(range_min=2030, range_max=2700, drive_mode=0)
        bus = self.s.follower.bus
        bus.apply_drive_mode = True
        bus.read = Mock(return_value=2732)
        result = self.command("inspect_gripper")
        self.assertAlmostEqual(result["unclipped_percent"], 104.77611940298507)
        self.assertTrue(result["outside_calibrated_range"])
        bus.read.assert_called_once_with("Present_Position", "gripper", normalize=False)
        self.assertEqual(bus.events, [])
        self.assertEqual(self.s.follower.sent, [])
        self.assertEqual(self.s.sent_count, 0)

    def test_raw_inspection_refused_while_armed(self):
        self.s.control.arm([0] * 6)
        with self.assertRaisesRegex(ValueError, "释放电机"):
            self.s.handle({"op": "inspect_gripper"}, [0] * 6, [0] * 6, time.monotonic())

    def test_seed_goal_before_torque_and_release_still_works_after_feedback_failure(self):
        self.command("begin")
        self.assertEqual(self.s.follower.bus.events, ["Goal_Position", "enable"])
        self.assertEqual(self.s.follower.sent[-1], dict.fromkeys([f"{n}.pos" for n in JOINTS], 0.0))
        self.command("begin")
        self.s.leader.fail = True
        before = len(self.s.follower.sent)
        self.s.tick(self.log)
        self.assertEqual(len(self.s.follower.sent), before)
        self.assertIsNotNone(self.s.fault)
        with self.assertRaises(ValueError):
            self.command("begin")
        self.command("release")
        self.assertFalse(self.s.torque_owned)
        self.assertFalse(self.s.control.armed)
        self.assertEqual(self.s.follower.bus.events[-1], "disable")
        self.command("shutdown")
        self.assertTrue(self.s.stop.is_set())

    def test_bad_goal_readback_prevents_torque_enable(self):
        self.s.follower.bus.bad_goal = True
        with self.assertRaisesRegex(RuntimeError, "verification failed"):
            self.command("arm")
        self.assertNotIn("enable", self.s.follower.bus.events)
        self.assertEqual(self.s.follower.sent, [])

    def test_capture_saves_measured_scale_and_observation_without_invented_world_truth(self):
        now = time.monotonic()
        points = np.indices((6, 9)).reshape(2, -1).T.astype(float)
        for i in range(30):
            self.s.history.append((now - 0.7 + i * 0.024, np.zeros(6)))
        for name in ("front", "side"):
            self.s.frames[name] = {
                "found": True,
                "time": now - 0.1,
                "points": points.copy(),
                "rgb": np.zeros((48, 64, 3), dtype=np.uint8),
            }
        self.s.capture("fit", np.zeros(6))
        row = json.loads((self.s.root / "pose_000_fit/capture.json").read_text(encoding="utf-8"))
        self.assertEqual(row["board"]["square_m"], 0.017)
        self.assertIsNone(row["joint_reference_truth"])
        self.assertFalse(row["world_registration_validated"])
        self.assertIsNone(row["cameras"]["front"]["world_points_m"])
        self.assertLess(abs(row["cameras"]["front"]["image_state_time_offset_s"]), 0.025)
        with self.assertRaises(ValueError):
            self.s.capture("check", np.zeros(6))
        self.s.frames["side"]["points"] += 10
        with self.assertRaisesRegex(ValueError, "2 像素"):
            self.s.capture("check", np.array([10, 0, 0, 0, 0, 0]))
        self.s.frames["front"]["time"] -= 2
        self.assertFalse(self.s.capture_ready(time.monotonic()))

    def test_local_http_page_start_stop_and_origin_guard_using_fake_hardware_only(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        sessions = []
        errors = []

        def fake_connect(s):
            s.control = controller()
            s.leader, s.follower = FakeArm(), FakeArm()
            s.cameras = {}
            s.leader.bus.is_connected = s.follower.bus.is_connected = False
            sessions.append(s)

        def serve():
            try:
                main()
            except Exception as exc:
                errors.append(exc)

        output = self.s.root / "http_test"
        argv = ["pose_capture", "--output", str(output), "--square-mm", "17", "--port", str(port)]
        base = f"http://127.0.0.1:{port}"
        with patch("sys.argv", argv), patch.object(PoseSession, "connect", fake_connect):
            server = threading.Thread(target=serve, daemon=True)
            server.start()
            try:
                for _ in range(60):
                    try:
                        with urlopen(base, timeout=0.2) as response:
                            html = response.read().decode("utf-8")
                        break
                    except (URLError, TimeoutError):
                        time.sleep(0.025)
                else:
                    self.fail(f"Fake server did not start: {errors}")
                self.assertIn("开始跟随", html)
                self.assertNotIn("按住这里", html)
                self.assertEqual(sessions[0].follower.bus.events, [])
                key = re.search(r"const key='([^']+)'", html).group(1)

                def post(op, origin=base):
                    request = Request(
                        base + "/command",
                        data=json.dumps({"op": op}).encode(),
                        headers={"Origin": origin, "Content-Type": "application/json", "X-Pose-Session": key},
                    )
                    with urlopen(request, timeout=2) as result:
                        return json.loads(result.read())

                with self.assertRaises(HTTPError) as denied:
                    post("begin", "http://example.test")
                self.assertEqual(denied.exception.code, 403)
                self.assertEqual(sessions[0].follower.bus.events, [])
                self.assertIn("token", post("begin"))
                self.assertEqual(sessions[0].follower.bus.torque, 1)
                post("pause")
                self.assertIsNone(sessions[0].control.token)
                post("release")
                self.assertEqual(sessions[0].follower.bus.torque, 0)
                post("shutdown")
            finally:
                for s in sessions:
                    s.stop.set()
                server.join(timeout=3)
            self.assertFalse(server.is_alive())
            self.assertEqual(errors, [])


class BrowserDisconnectTests(unittest.TestCase):
    def handler(self, error):
        handler = LocalPoseRequestHandler.__new__(LocalPoseRequestHandler)
        handler.close_connection = False
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        handler.wfile = SimpleNamespace(write=Mock(side_effect=error))
        return handler

    def test_browser_abort_reset_and_broken_pipe_quietly_end_response(self):
        for error in (
            ConnectionAbortedError(10053, "Client aborted"),
            ConnectionResetError(),
            BrokenPipeError(),
        ):
            with self.subTest(error=type(error).__name__):
                handler = self.handler(error)
                self.assertFalse(handler.respond(200, {"capture_ready": True}))
                self.assertTrue(handler.close_connection)

    def test_disconnect_while_reading_or_flushing_is_also_a_closed_client(self):
        handler = self.handler(None)
        with patch("http.server.BaseHTTPRequestHandler.handle", side_effect=ConnectionResetError()):
            handler.handle()
        self.assertTrue(handler.close_connection)

    def test_unrelated_server_errors_remain_visible(self):
        handler = self.handler(RuntimeError("Real server bug"))
        with self.assertRaisesRegex(RuntimeError, "Real server bug"):
            handler.respond(200, b"test")


if __name__ == "__main__":
    unittest.main()
