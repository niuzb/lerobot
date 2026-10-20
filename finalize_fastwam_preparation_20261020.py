"""Enable the queued GPU job only after its assets and CPU data checks succeed."""

import json
import os
import time
from pathlib import Path

ROOT = Path("/root/code/lerobot/outputs/fastwam_20261020")
deadline = time.monotonic() + 24 * 3600
while time.monotonic() < deadline:
    asset_file = ROOT / "preparation_status.json"
    assets = json.loads(asset_file.read_text()) if asset_file.exists() else {}
    data_file = ROOT / "data_preflight.json"
    data = json.loads(data_file.read_text()) if data_file.exists() else {}
    audit_file = ROOT / "base_header_audit.json"
    audit = json.loads(audit_file.read_text()) if audit_file.exists() else {}
    tests_file = ROOT / "cpu_tests.json"
    tests = json.loads(tests_file.read_text()) if tests_file.exists() else {}
    if assets.get("completed") and data.get("passed") and audit.get("passed") and tests.get("passed"):
        (ROOT / "READY").write_text(
            "Verified pinned assets, checkpoint coverage, CPU data and regression tests.\n"
        )
        print("FASTWAM_PREPARATION_READY", flush=True)
        break
    if assets.get("stage") == "failed":
        (ROOT / "PREPARATION_FAILED").write_text("Asset preparation failed; inspect preparation_status.json")
        raise SystemExit(1)
    for stem, done in (("prepare", assets.get("completed")), ("data_preflight", data.get("passed"))):
        pid_file = ROOT / (stem + ".pid")
        if pid_file.exists() and not done:
            try:
                os.kill(int(pid_file.read_text()), 0)
            except ProcessLookupError:
                (ROOT / "PREPARATION_FAILED").write_text(stem + " process exited before verification passed")
                raise SystemExit(1) from None
    time.sleep(30)
else:
    (ROOT / "PREPARATION_FAILED").write_text("Preparation exceeded the 24-hour deadline")
    raise SystemExit(1)
