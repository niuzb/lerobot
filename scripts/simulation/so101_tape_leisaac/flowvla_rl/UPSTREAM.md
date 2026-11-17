# Third-party SDE implementation

The `sde.py` and `smolvla_sde.py` are from Teng Zhu's
[FlowVLA-RL](https://github.com/BlackMirean/FlowVLA-RL), MIT license,
commit `50e73afc07212f48bed370ea4c88a3131d6f914a`.
The accompanying LICENSE applies to those files.

Local compatibility patch: `smolvla_sde.py` omits `fill_kv_cache=True` because
this LeRobot snapshot fills its prefix cache through `use_cache=True` and its
forward signature does not accept that older keyword. The transition math is unchanged.

Repository integration also normalizes import ordering, type-annotation spelling and
formatting to the local Ruff configuration; these changes do not alter the SDE equations.

Our LeIsaac adaptation uses standardized group advantages, asymmetric PPO clipping
0.2/0.28 and zero KL, rather than that repository's default centered-only advantage
and reference KL. All 50x32 stochastic latent coordinates enter the density ratio;
only the first 30x6 generated controls are executed. This keeps a joint latent-path
likelihood, rather than silently dropping policy-dependent unexecuted coordinates.
The controller's fixed clipping map is part of the simulator transition; sampled
latent states are never replaced by clipped motor commands for density scoring.

This continuous Flow-SDE adaptation preserves the existing SmolVLA expert and is
not a reproduction of the discrete-action-token architecture in arXiv:2603.22876.
