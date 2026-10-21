"""Config/discovery tests that also run without the DreamZero model extra."""

import json
import subprocess
import sys
from pathlib import Path

import draccus
import pytest

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies import make_policy_config


def test_dreamzero_config_is_discoverable_without_loading_weights():
    cfg = make_policy_config("dreamzero", device="cpu")
    assert cfg.type == "dreamzero"
    assert cfg.wan_revision and cfg.image_encoder_revision and cfg.tokenizer_revision


@pytest.mark.parametrize("filename", ["so101_wan21_14b.json", "so101_wan22_5b.json"])
def test_example_is_a_valid_native_training_config(filename):
    root = Path(__file__).resolve().parents[3]
    path = root / "examples/dreamzero" / filename
    with draccus.config_type("json"):
        cfg = draccus.parse(TrainPipelineConfig, path, args=[])
    assert cfg.policy.type == "dreamzero" and cfg.policy.chunk_size == 24
    assert cfg.steps == 10 and not cfg.policy.push_to_hub
    assert json.loads(path.read_text())["policy"]["init_from"]


def test_missing_optional_dependency_reports_dreamzero_extra():
    code = """
import sys
sys.modules["diffusers"] = None
from lerobot.policies import make_policy_config, get_policy_class
cfg = make_policy_config("dreamzero", device="cpu")
cls = get_policy_class("dreamzero")
try:
    cls(cfg)
except ImportError as error:
    assert "lerobot[dreamzero]" in str(error), str(error)
else:
    raise AssertionError("Missing dependency was not rejected")
assert "lerobot.policies.dreamzero.wan.causal_wan" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
