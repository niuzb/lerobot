"""Run the real training entry point with first-run JEPA checks; never controls hardware."""
import json
import math
import os
from pathlib import Path

import torch
import lerobot.scripts.lerobot_train as trainer

original_make = trainer.make_policy
original_update = trainer.update_policy
state = {"gradients": {}, "losses": []}


def checked_make(*args, **kwargs):
    policy = original_make(*args, **kwargs)
    state["policy"] = policy
    for name, param in policy.named_parameters():
        group = next((g for prefix, g in [
            ("model.qwen.", "qwen"),
            ("model.action_model.", "action"),
            ("model.video_predictor.", "world_predictor"),
        ] if name.startswith(prefix)), None)
        if group and param.requires_grad:
            def hook(grad, group=group):
                if not state["gradients"].get(group):
                    state["gradients"][group] = bool(torch.isfinite(grad).all() and grad.abs().max() > 0)
            param.register_hook(hook)
    return policy


def checked_update(*args, **kwargs):
    state["batch"] = kwargs["batch"] if "batch" in kwargs else args[2]
    metrics, output = original_update(*args, **kwargs)
    assert all(math.isfinite(float(output[k])) for k in ("action_loss", "wm_loss", "loss")), output
    assert output["wm_loss"] > 0, output
    state["losses"].append(output)
    return metrics, output


def main():
    trainer.make_policy = checked_make
    trainer.update_policy = checked_update
    trainer.main()
    assert all(state["gradients"].get(g) for g in ("qwen", "action", "world_predictor")), state["gradients"]
    policy = state["policy"]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = policy.predict_action_chunk(state["batch"])
    assert prediction.shape[-1] == 6 and torch.isfinite(prediction).all(), prediction.shape
    report = {
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "gradients": state["gradients"], "losses": state["losses"],
        "prediction_shape": list(prediction.shape),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    path = Path(os.environ["JEPA_AUDIT_PREFIX"] + f'.rank{report["rank"]}.json')
    path.write_text(json.dumps(report, indent=2))
    print("JEPA_PREFLIGHT_PASSED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
