import json
import sys
from types import SimpleNamespace

import pytest

from examples.dreamzero import launch_libero_distributed as launcher
from examples.dreamzero.libero_full_utils import write_json
from examples.dreamzero.train_libero_distributed import make_config


def test_multi_rank_preserves_global_batch():
    job = {"accumulation": 16, "grad_clip": 1.0, "lr": 1e-5,
           "weight_decay": 1e-5, "warmup_updates": 1000}
    for world_size in (2, 4, 8):
        config = make_config(job, world_size)
        assert config["gradient_accumulation_steps"] * world_size == 16
    with pytest.raises(ValueError):
        make_config(job, 3)
    job["parameter_cpu_offload"] = False
    config = make_config(job, 2)
    assert "offload_param" not in config["zero_optimization"]
    assert config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert config["train_batch_size"] == 16
    job["micro_batch"] = 2
    config = make_config(job, 2)
    assert config["train_micro_batch_size_per_gpu"] == 2
    assert config["gradient_accumulation_steps"] == 4
    assert config["train_batch_size"] == 16


def test_failed_reload_never_authorizes_training(tmp_path, monkeypatch):
    write_json(tmp_path / "smoke_complete.json", {"update": 4})
    monkeypatch.setattr(sys, "argv", ["launch", "--root", str(tmp_path), "--reference", str(tmp_path)])
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(launcher.subprocess, "run", fail)
    with pytest.raises(SystemExit, match="7"):
        launcher.main()
    assert len(calls) == 1
    assert "--smoke" in calls[0] and "--resume" in calls[0]
    assert not (tmp_path / "PREFLIGHT_PASSED.json").exists()


def test_preflight_requires_both_rank_restore_and_continuation(tmp_path, monkeypatch):
    write_json(tmp_path / "smoke_complete.json", {"update": 4})
    monkeypatch.setattr(sys, "argv", ["launch", "--root", str(tmp_path), "--reference", str(tmp_path),
                                     "--preflight-only"])

    def restore(command, **kwargs):
        for rank in (0, 1):
            write_json(tmp_path / f"reload_rank{rank}.json", {"passed": True, "update": 4})
        write_json(tmp_path / "smoke_complete.json", {"update": 5, "initial_update": 4})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", restore)
    launcher.main()
    assert (tmp_path / "PREFLIGHT_PASSED.json").exists()
    assert json.loads((tmp_path / "stage.json").read_text())["stage"] == "preflight_passed"


def test_pilot_stops_at_requested_budget_and_requests_small_evaluation(tmp_path, monkeypatch):
    write_json(tmp_path / "PREFLIGHT_PASSED.json", {"passed": True})
    write_json(tmp_path / "latest.json", {"tag": "update_000004", "update": 4})
    monkeypatch.setattr(sys, "argv", ["launch", "--root", str(tmp_path), "--reference", str(tmp_path),
                                     "--max-updates", "1000"])
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", execute)
    launcher.main()
    assert calls[0][calls[0].index("--max-updates") + 1] == "1000"
    assert "--selection-only" in calls[1]
    assert calls[1][calls[1].index("--update") + 1] == "1000"
    assert len(calls) == 2
    assert json.loads((tmp_path / "stage.json").read_text())["stage"] == "segment_completed"
