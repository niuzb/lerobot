# LIBERO-only full DreamZero training

The previous A100 worker was **4434805**, attached to the **lianqing CPU devbox
dbx_37944**. Check current availability before running: the historical worker ID
does not establish that a GPU is still allocated. Inspect workers through the master:

```bash
NO_COLOR=1 TERM=dumb bytedcli merlin cpu-devbox instances exec \
  --sid dbx_37944 --command 'NO_COLOR=1 TERM=dumb mlx worker list'
```

The existing FastWAM asset/environment root is
`/root/code/lerobot/outputs/fastwam_libero_20261021`. The pinned LIBERO dataset,
original Wan DiT shards and CLIP were prepared under
`/root/code/lerobot/outputs/dreamzero_libero_20261024`. The fixed 40-task,
50-initial-states-per-task evaluation manifest is
`/root/code/lerobot/outputs/dreamzero_libero_eval_20261025/manifest.json`.

Prepare a new run in an independent environment with the local LIBERO source on
`PYTHONPATH` and the existing `LIBERO_CONFIG_PATH`:

```bash
uv run python -m examples.dreamzero.prepare_libero_full \
  --root outputs/dreamzero_libero_full \
  --previous-root /root/code/lerobot/outputs/dreamzero_libero_20261024 \
  --reference-root /root/code/lerobot/outputs/fastwam_libero_20261021 \
  --protocol-manifest /root/code/lerobot/outputs/dreamzero_libero_eval_20261025/manifest.json
uv run python -m examples.dreamzero.launch_libero_full --root outputs/dreamzero_libero_full
```

The launcher runs four updates, exits that process, restores the complete state in
a new process and checks one further update. Production resumes update 4 and ends
at update 20,000, so the verification-only fifth update does not add to the formal
budget. `SMOKE_PASSED.json` gates production. Each phase has a separate log and
`stage` records the current phase. `failure.json` preserves a Python failure;
`first_optimizer_allocation.json` records the projected first Adam-state allocation.

## Memory profiles

Both profiles keep the full DiT trainable, BF16 model weights, FP32 master weights,
batch size 1, 16 accumulation microsteps, and the existing dataset/evaluation split.
The default is `zero3_cpu_adam` with `offload_optimizer.ratio=0.5`. Parameters stay
on the GPU; FP32 master/Adam partitions are split between CPU and GPU. A subgroup
limit of 100 million elements bounds temporary conversion buffers; the measured
CPU fraction is logged because DeepSpeed assigns whole subgroups, not exact bytes.

The alternative `zero3_adamw8bit` uses a supplied bitsandbytes `AdamW8bit` optimizer
on GPU with CPU offload disabled. Large optimizer partitions use uint8 moment
storage and FP32 quantization scales; small partitions may retain FP32 moments.
This changes optimizer numerics and needs its own validation run.

These are **separate profiles**, not two switches to combine. DeepSpeed 0.19.8
requires `DeepSpeedCPUAdam` for partial offload and creates a normal GPU AdamW
backup. This implementation does not patch those internals to inject an 8-bit
hybrid optimizer. An 8-bit profile with a nonzero offload ratio fails immediately.

Install runtime dependencies in the Linux CUDA training environment (the base
DreamZero dependencies and pinned data/assets must already be prepared):

```bash
uv pip install 'deepspeed==0.19.8'
# Only for the 8-bit profile. Its installed version is recorded and enforced on resume.
uv pip install bitsandbytes
```

Run the four-update and independent reload preflight before production:

```bash
uv run python -m examples.dreamzero.launch_libero_full \
  --root outputs/dreamzero_libero_full \
  --optimizer-mode zero3_cpu_adam --offload-ratio 0.5 --preflight-only

# Use a separately prepared directory to compare the 8-bit profile.
uv run python -m examples.dreamzero.launch_libero_full \
  --root outputs/dreamzero_libero_full_8bit \
  --optimizer-mode zero3_adamw8bit --preflight-only
```

The profile is persisted in `runtime.json`. Existing checkpoints seal it; switching
profile after an update requires a new prepared directory. A prior initialization
failure with no complete checkpoint can reuse its prepared data directory. After
preflight, resume production explicitly from the four-update checkpoint:

```bash
uv run torchrun --standalone --nproc_per_node=1 \
  --module examples.dreamzero.train_libero_full \
  --root outputs/dreamzero_libero_full \
  --resume-from outputs/dreamzero_libero_full/checkpoints/update_000004
```

`optimizer_audit.json` reports real master/state bytes per device and verifies
state precision after the first update. `first_optimizer_allocation.json` budgets
the lazily created state separately for CPU and GPU. Existing 90 GiB host and
72 GiB GPU limits still apply; no measured full-model peak is claimed until a
CUDA preflight has passed.

Checkpoints store BF16 parameter partitions, FP32 masters, and **both** CPU/GPU
optimizer states in bounded chunks with checksums. Format 2 also preserves all
8-bit moment tensors, quantization maps/scales, and their original devices/dtypes.
RNG, sampler, both optimizer learning rates, scheduler, runtime versions and update
counters are restored alongside them. Only single-rank BF16 training without
NVMe swapping is supported. Format 1 checkpoints need the former ZeRO-2 runner;
they are not silently converted to a new optimizer layout.

Local contract tests and optional native CUDA update/save/reload checks:

```bash
uv run pytest tests/policies/dreamzero/test_libero_full_runtime.py \
  tests/policies/dreamzero/test_libero_full_utils.py -q
```

CUDA checks are skipped when CUDA or the selected optional optimizer is unavailable.
Passing the CPU contracts does not establish A100 compatibility or memory usage.

The 40 selection episodes are labeled separately from the remaining 1,960 final
episodes. Final evaluation runs in another process after the training optimizer
has exited. Every failure stops the launcher, and no entry point releases a worker
or uploads to the Hub. Keep the independent existing resource-retention shell alive.
