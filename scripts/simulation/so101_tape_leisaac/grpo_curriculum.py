"""Audited reset contexts and an outcome-driven backwards task curriculum."""

import numpy as np
from common import SuccessTracker

STAGES = ("release", "transport", "lift", "approach", "full")
ALL_STAGES = ("open_hold", *STAGES)


def tracker_from_prefix(config, physics, initial_position, tick):
    """Reconstruct historical prerequisites from actual source physics, not constants."""
    tracker = SuccessTracker(config, initial_position)
    for index in range(tick + 1):
        succeeded = tracker.update(
            physics["tape"][index, :3],
            physics["tape"][index, 3:7],
            physics["tape"][index, 7:],
            physics["state"][index, 5],
            *(physics["forces"][index] > 1e-6),
            physics["holder"][index, :3],
            physics["holder"][index, 3:7],
        )
        if succeeded:
            raise ValueError("A curriculum prefix must not already satisfy terminal success")
    # A new rollout must independently accumulate all terminal stable ticks.
    tracker.stable = 0
    return tracker


def validate_anchor(config, physics, initial_position, stage, tick):
    if stage not in ALL_STAGES[:-1] or not 0 <= tick < len(physics["state"]):
        raise ValueError("Invalid curriculum anchor")
    tracker = tracker_from_prefix(config, physics, initial_position, tick)
    flags = (tracker.grasped, tracker.lifted, tracker.carried)
    expected = {
        "open_hold": (True, True, True),
        "release": (True, True, True),
        "transport": (True, True, False),
        "lift": (True, False, False),
        "approach": (False, False, False),
    }[stage]
    if flags != expected:
        raise ValueError(f"{stage} anchor has the wrong recorded prerequisite history: {flags}")
    held = bool(np.all(physics["forces"][tick, :2] > 1e-6))
    if stage not in ("approach", "open_hold") and not held:
        raise ValueError("A held-object curriculum must start with recorded bilateral contact")
    if physics["state"][tick, 5] >= config["generation"]["open_percent"] - 5 and held:
        raise ValueError("Held-object anchor is already commanded open")
    if stage == "open_hold":
        if held or physics["state"][tick, 5] >= config["generation"]["open_percent"] - 5:
            raise ValueError("Open-hold initialization must still require a new opening action")
        tracker.update(
            physics["tape"][tick, :3],
            physics["tape"][tick, 3:7],
            physics["tape"][tick, 7:],
            config["generation"]["open_percent"],
            *(physics["forces"][tick] > 1e-6),
            physics["holder"][tick, :3],
            physics["holder"][tick, 3:7],
        )
        if tracker.stable != 1:
            raise ValueError("Open-hold anchor must already have valid physical containment and support")
    return {"grasped": flags[0], "lifted": flags[1], "carried": flags[2], "recorded_held": held}


class CurriculumSchedule:
    """Advance on measured suffix success; halt rather than hide a failed stage."""

    def __init__(self, min_groups=4, max_groups=10, threshold=0.25, stages=None):
        self.min_groups = min_groups
        self.max_groups = max_groups
        self.threshold = threshold
        self.stages = tuple(STAGES if stages is None else stages)
        if (
            not self.stages
            or self.stages[-1] != "full"
            or len(set(self.stages)) != len(self.stages)
            or any(stage not in ALL_STAGES for stage in self.stages)
        ):
            raise ValueError("Invalid curriculum stage sequence")
        self.index = 0
        self.groups = []
        self.completed_stages = []

    @property
    def stage(self):
        return self.stages[self.index]

    def state_dict(self):
        return {
            "min_groups": self.min_groups,
            "max_groups": self.max_groups,
            "threshold": self.threshold,
            "stages": list(self.stages),
            "index": self.index,
            "groups": [list(group) for group in self.groups],
            "completed_stages": [dict(stage) for stage in self.completed_stages],
        }

    @classmethod
    def from_state_dict(cls, state):
        instance = cls(**{key: state[key] for key in ["min_groups", "max_groups", "threshold", "stages"]})
        if not 0 <= state["index"] < len(instance.stages):
            raise ValueError("Invalid restored curriculum stage")
        if any(len(group) != 8 or any(reward not in (0, 1) for reward in group) for group in state["groups"]):
            raise ValueError("Invalid restored group outcomes")
        instance.index = state["index"]
        instance.groups = [list(group) for group in state["groups"]]
        instance.completed_stages = list(state["completed_stages"])
        if instance.stage != "full" and len(instance.groups) >= instance.max_groups:
            raise ValueError("A stalled stage requires an explicitly revised curriculum")
        return instance

    def observe(self, rewards):
        if len(rewards) != 8 or any(x not in (0, 1) for x in rewards):
            raise ValueError("A curriculum group requires eight binary live outcomes")
        self.groups.append(list(rewards))
        recent = sum(sum(g) for g in self.groups[-2:]) / (8 * min(len(self.groups), 2))
        result = {"stage": self.stage, "groups": len(self.groups), "recent_success_rate": recent}
        if self.stage == "full":
            return {**result, "decision": "continue_full_task"}
        if len(self.groups) >= self.min_groups and recent >= self.threshold:
            self.completed_stages.append({**result, "decision": "advanced"})
            self.index += 1
            self.groups = []
            return {**result, "decision": "advance", "next_stage": self.stage}
        if len(self.groups) >= self.max_groups:
            return {**result, "decision": "stalled"}
        return {**result, "decision": "continue"}
