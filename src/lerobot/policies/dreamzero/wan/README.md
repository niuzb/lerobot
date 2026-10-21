# DreamZero / Wan source provenance

Adapted from the local DreamZero checkout, commit `ab790c198fbce33503358efbbd4187ce9a89adf3`.
Source: `groot/vla/model/dreamzero/modules/` and `n1_5/modules/action_encoder.py`.

NVIDIA and Alibaba copyrights and the Apache-2.0 license are retained.
The original checkout is not a runtime dependency. Changes: relative imports,
PyTorch SDPA, removal of server/TensorRT/debug-mask code, CPU/dtype support,
blockwise clean-context attention, and unpacking the non-checkpointed block result.
The CLIP path retains flattened multi-head attention semantics. Unused CLIP text
models and Sora download hooks were removed; the separately loaded CLIP file may
contain unused `textual.*` weights. Explicit CUDA constructors and forced UniPC
compilation were removed. Core training/inference orchestration is implemented in
the native policy, retaining the upstream teacher-forcing and observed-video KV
cache update rules. Full-sized checkpoint execution has not been GPU-validated.

Original source SHA-256:

- `wan_video_vae.py`: `1f65f8fd915c1e6703da9d320e764df107dbb988b12623a7e087554281c84edc`
- `wan_video_text_encoder.py`: `8dfddba064cbe5ce12b2481c84c0ac0e1ecf2dd5484be8f026ce3d34b357a5b9`
- `wan_video_image_encoder.py`: `e405a886a8e1d48c1e1981770a3e1b4d102c173734168af722d91b400acfec70`
- `flow_match_scheduler.py`: `6c3f720e7520bf6e138a0e78732c7b7283b109c589dd2e231c6a67ff9f663477`
- `flow_unipc_multistep_scheduler.py`: `8748ff51dc050833ae75e621088787a9d1eaf28d8c964dd7639528c29395765e`
- `wan2_1_submodule.py`: `8a038f5647d420a818ac92740ade5cdc5984925e5a32b6d3c9e767c93dd2ea51`
- `wan_video_dit_action_casual_chunk.py`: `efc5120f73cbce2e00b67d78e8bf71a561e943a909791f039b435abc27ccd605`
