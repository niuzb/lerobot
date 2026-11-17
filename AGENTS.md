This file provides guidance to AI agents when working with code in this repository.

## Documentation Index

- [User guides](./docs/guides/) include the [SO-101 quickstart and training guide](./docs/guides/AGENT_GUIDE.md) and the [Chinese project overview](./docs/guides/readme_zh.md).
- [Architecture](./docs/architecture/) covers the [CLI commands](./docs/architecture/cli/README.md), [rollout internals](./docs/architecture/LEROBOT_ROLLOUT_ARCHITECTURE.md), and [training internals](./docs/architecture/train.md).
- [Dataset operations](./docs/datasets/) cover recording recovery and dataset cleaning.
- [Policy runbooks and experiment records](./docs/policies/) are grouped by [ACT](./docs/policies/act/), [FastWAM](./docs/policies/fastwam/), [Pi 0.5](./docs/policies/pi05/), [SmolVLA](./docs/policies/smolvla/), and [VLA-JEPA](./docs/policies/vla-jepa/).
- [Simulation](./docs/simulation/) covers Isaac Sim setup, VLA-JEPA simulation, the SO-101 real-to-sim-to-real workflow, and the [LeIsaac tape task, mixed training, and GRPO runbook](./docs/simulation/SO101_TAPE_LEISAAC_RUNBOOK.md).
- [Change summaries](./docs/changes/) record past repository changes.
- [Community guidelines](./docs/CONTRIBUTING.md) include the [code of conduct](./docs/CODE_OF_CONDUCT.md), [security policy](./docs/SECURITY.md), and [AI usage policy](./docs/community/AI_POLICY.md).
- [Hugging Face documentation sources](./docs/source/) ship to the public site. See the [build and preview instructions](./docs/README.md).

Add repository-wide documentation to the matching directory under `docs/`. Add a link here when introducing a new documentation category.

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
DEVICE=cuda make test-end-to-end                      # All E2E tests
pre-commit run --all-files                           # Lint + format (ruff, typos, bandit, etc.)
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **[`scripts/`](./scripts/)** — Repository tools and experiment entry points for [datasets](./scripts/datasets/),
  [teleoperation](./scripts/teleop/), [diagnostics](./scripts/diagnostics/), [simulation](./scripts/simulation/), and
  [policies](./scripts/policies/). Run documented commands from the repository root.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files** include `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, and `README.md` (project overview).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Imports**: prefer top-level imports; relative (`from .sibling import X`) across sibling files within a module, absolute (`from lerobot.module import X`) across modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`, see `pyproject.toml`). Guard optional imports with `TYPE_CHECKING or _foo_available` at module top + a `require_package(...)` check at use time. Reuse the `_foo_available` flags in `utils/import_utils.py`; don't call `is_package_available`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
