# SO-101 tape task in LeIsaac

Experiment tools for collecting tape-to-pen-holder trajectories, rendering and exporting LeRobot datasets,
training SmolVLA with simulated or mixed real/simulated data, and running online Flow-SDE GRPO with a task curriculum.

See the [runbook](../../../docs/simulation/SO101_TAPE_LEISAAC_RUNBOOK.md) for script roles, required scene and
asset inputs, isolated simulator dependencies, commands, verification artifacts, and historical experiment notes.

Run scripts by path from the repository root, for example:

```bash
uv run scripts/simulation/so101_tape_leisaac/prepare_spatial.py --help
```

Keep this directory intact when creating a run's `code/` snapshot: sibling imports and subprocess entry points
depend on the complete tree. Isaac Sim uses its own Python environment and launcher; the simulator, scene files,
assets, datasets, and model checkpoints are not bundled here. The mixed-training and GRPO coordinators retain
experiment-specific paths and must be adapted before use on another machine.

The vendored FlowVLA-RL code retains its [MIT license](./flowvla_rl/LICENSE), pinned revision, and
[compatibility notes](./flowvla_rl/UPSTREAM.md).
