# Goal: boot the vLLM TPU stack (tpu-inference, JAX) on a fresh H100 node — the JAX arms of GPU Figure 2

## Assignment and status

Run the vLLM TPU stack (PyPI `vllm-tpu` + `tpu-inference`, JAX/XLA) on one NVIDIA H100 of
`brent-llm-d-asia-southeast1-h100`, changing only what the GPU forces. Then measure, as in paper
Figure 2 and the CUDA vLLM campaign, how long a new instance takes to serve Gemma-4 31B on a node
scaled up from zero, split into stages, for each weight/compile-cache delivery arm. Compare it with
CUDA vLLM on the same nodes and with the TPU figure. "JAX on GPU boots slower and the cache does
not close the gap" is an acceptable honest outcome.

Workspace `~/hetero-wt/GpuJax` (jj workspace `gpu-jax`, bookmark `research/gpu-jax`, parent
`research/gpu-autoscale`). Status on 2026-10-22:

- Done (11:13 UTC): image `gpu_jax/image/IMAGE` (sha256:913c3f40, 4.1 GB); environment check
  (`results/gpu-jax/envcheck.json`); attention tests (15 pass, `results/gpu-jax/kernel-test.txt`);
  Gemma-4 31B served end to end via Ray with the CUDA campaign's greedy output; driver
  `scripts/gpu_jax_autoscale.py` (reuses `scripts/gpu_autoscale.py`), every arm smoke-tested
  (`results/gpu-jax/smoke-22f57e8`); campaign `results/gpu-jax/campaign-f1f2336`: 5 arms x 3 good
  runs on fresh spot nodes, figure + summary; GPU pools at 0, Ray head deleted.
- Environment additions found on the H100: `JAX_PLATFORMS=cuda,cpu`
  (tpu-inference reshapes weights on a CPU mesh), `LD_LIBRARY_PATH=/usr/local/nvidia/lib64` in the
  pod (driver location on GKE), `TRITON_DISABLE_LINE_INFO=1` (PTX line info made the JAX
  persistent-cache key depend on the Ray working_dir path).

## Decisions already made — implement, do not relitigate

- **Environment**: stock PyPI releases only. `vllm-tpu==0.32.0`, `tpu-inference==0.32.0` (pins
  `jax==jaxlib==0.11.2`, `libtpu==0.0.48`, `flax==0.12.10`, `tokamax==0.0.17.dev20261015`),
  `jax-cuda13-plugin[with-cuda]==jax-cuda13-pjrt==0.11.2`, `jax-triton==0.4.1`, `triton==3.8.0`,
  CPU-only `torch==2.10.0`, `ray==2.56.1` (matches the local client), Python 3.12. Resolve with
  `uv pip compile` (`--python-platform x86_64-manylinux_2_36`, PyTorch CPU index, prerelease
  allowed only where explicit). Node driver 580.178 (CUDA 13). Any version change means re-lock,
  rebuild and record the new image digest. `libtpu` stays installed: vLLM's TPU platform plugin
  detects it, and `JAX_PLATFORMS=cuda,cpu` keeps JAX off it.
- **Image**: built without docker (Cloud Build is denied). `gpu_jax/image/build_image.sh` runs in a
  CPU pod, installs the lock into `/opt/venv` on `python:3.12-slim-bookworm` and appends it as one
  layer with crane: `asia-southeast1-docker.pkg.dev/sysres-disagg-ml/gpu-jax/vllm-jax-cuda:0.32.0-jax0.11.2-cu13`.
  The image contains no fork code.
- **Fork code ships through Ray, as on TPU**: a Ray head pod on the CPU `default-pool`, GPU pods
  join it as workers, and the engine starts as a Ray job whose `working_dir` is `gpu_jax/`. Patches
  are applied by `sitecustomize.py` post-import hooks (`HETERO_JAX_GPU=1`) in every process.
  Installed files are never edited.
- **Attention stays ragged paged attention** (vLLM's paging contract; the tpu-inference runner,
  KV cache manager and cache layout are unchanged). Only the Pallas/Mosaic TPU kernel is replaced
  by vLLM's Triton unified attention (ejKernel 0.0.80.3's JAX port, Apache-2.0, vendored in
  `rpa_triton.py`, launched with jax-triton) plus one XLA scatter for the KV write. CUDA vLLM
  forced this same algorithm (`TRITON_ATTN`) for Gemma-4 because `global_head_dim` is 512, while
  JAX's cuDNN attention caps head_dim at 256 on Hopper.
- **Nodes**: spot pool `gpu-pool-spot` (`a3-highgpu-1g`, one H100 80 GB). On-demand `gpu-pool`
  fails VM creation with `INTERNAL_ERROR`. The user cleared use of this cluster ("no one is using
  that: you can do what you need").

## Existing evidence to reuse first

- CUDA vLLM campaign `results/gpu-autoscale/campaign-549bd16` (bookmark `research/gpu-autoscale`,
  15 good runs, n=3 per arm, vLLM v0.23.0 cu129, TP1). Median time to ready: raw 456.7 s, ckpt
  144.9 s, ckpt-cc 105.3 s, runai 126.6 s, runai-cc 85.1 s. GKE submit → container start 284 s
  median (n=12): VM create + boot 119 s, GPU driver → scheduled 42 s, 12.1 GB image pull 124 s.
- Reusable machinery: `scripts/gpu_autoscale.py` (run / adopt / report, polled kubelet logs,
  verified cache export), `scripts/gpu_autoscale/{pod_run.sh,fetch.py}`,
  `scripts/gpu_autoscale_figure.py`, and skill `hetero-gpu-autoscale-h100`.
- GCS (asia-southeast1): `gs://hongtaozhang-gpu-model-cache-asia-southeast1/gemma-4-31b/`
  (62.6 GB), `.../compile-cache/`, and the gcsfuse binary under `.../tools/`. AR pull-through
  cache `dockerhub-remote`.
- TPU Figure 2 (`~/tpuflex-overleaf/background.tex`, `scripts/autoscale_figure.py`): 4x4 GKE
  scale-up median 242.3 s; engine boot raw 779.4 s, ckpt + GCS XLA cache 284.9 s, peer 50.2 s.
- Gemma-4 31B text config: 60 layers (50 sliding with window 1024, head_dim 256, 16 KV heads;
  10 full with head_dim 512, 4 KV heads, `attention_k_eq_v`), 32 query heads, hidden size 5376.

## Question and hypothesis

Does the TPU stack's boot-time profile carry over when the same JAX/XLA code targets a GPU?
Hypothesis: download and load_model match CUDA vLLM, since the bytes and the GCS path are the
same. Without a cache, XLA compile + warm-up dominates, because tpu-inference precompiles every
token and request bucket and GPU XLA compiles of a 60-layer model take tens of seconds each. A
persistent JAX compilation cache plus the Triton cache, copied from a peer, recovers most of that,
as on TPU. Falsified if compile is a minor share without a cache, or if the cache does not shorten
it.

## Arms

`raw` (lazy GCS FUSE mount), `ckpt` (checkpoint downloaded to tmpfs first), `ckpt-cc` (+ JAX
persistent compilation cache and `~/.triton/cache` copied from a peer before boot). Add a Run:ai
streamer arm only if tpu-inference's JAX model path supports `--load-format runai_streamer`
(it pins `runai-model-streamer[gcs]`); check, don't assume. Engine settings are those of the CUDA
campaign (`--max-model-len 8192 --max-num-seqs 512 --max-num-batched-tokens 2048`, memory
utilisation 0.92) unless one is infeasible on tpu-inference. Document any deviation, its reason
and its effect on compile time.

## Non-negotiable rules

- **Correctness before timing**: unit-test the attention against tpu-inference's
  `ref_ragged_paged_attention` on an H100 (prefill, decode, mixed batch, padded sequences,
  sliding window, head_dim 256 and 512, KV write). Every timed run must pass the same greedy
  chat-completion check as the CUDA campaign.
- **Fresh node per timed run** (scale up from zero), n=3 good runs per arm, and the same stage
  decomposition as the CUDA figure. Map tpu-inference log anchors to stages explicitly and keep
  the raw logs. Setup steps (gcsfuse install, Ray join) are excluded and reported separately.
- **Isolation**: always `--kubeconfig ~/.kube/brent-h100.yaml`, namespace `gpu-jax`. Never touch
  the global kube context, the TPU cluster, `rcm_exec` or other workspaces.
- **Cost**: scale GPU nodes back to 0 after every session of runs and after failures. Delete the
  `ar-token` secret when the image build is done.
- **Provenance**: every run record carries the image digest, the lock hash, the `gpu_jax/` commit
  id and the patch list that was applied (`hetero_jax_gpu.patches.APPLIED`).
- Commit on `research/gpu-jax` only; do not push or publish without asking.

## Bounded execution sequence

1. **Environment check on H100**: versions match the lock, `jax.devices()` shows CUDA,
   a jax-triton add-kernel smoke test passes, and `import vllm, tpu_inference` resolves
   `TpuPlatform` with the patches applied.
2. **Kernel test** against the reference, as listed above, on an H100.
3. **End-to-end serve**: Ray head + one GPU worker. Serve Gemma-4 31B as a Ray job, pass the
   greedy check, and record the stage timeline. Fix each TPU assumption that surfaces with a
   patch in `patches.py` (keyed by module, one purpose each).
4. **Driver**: add JAX arms to `scripts/gpu_autoscale.py`, reusing run/adopt/report and the
   verified cache export. Smoke-test each arm once.
5. **Campaign**: n=3 per arm on fresh spot nodes.
6. **Figure and report**: JAX rows next to the CUDA rows (same panels and colours as
   `gpu_autoscale_figure.py`), a per-stage medians table against CUDA vLLM and TPU, and threats
   to validity.

## Deliverables and completion criteria

- `gpu_jax/` (env lock, build script, patches, kernel, tests), the image digest, and the driver arms.
- `results/gpu-jax/<campaign>/`: `runs.jsonl`, per-run logs and k8s records, figure (SVG + PDF),
  summary JSON.
- A short report in chat: per-arm stage medians vs CUDA vLLM and TPU, what the cache buys, every
  deviation from the CUDA campaign settings, and threats to validity. Update the Google Doc only
  when asked.

Done when every arm has 3 good runs on fresh nodes that passed the greedy check, the figure and
summary are rendered, the evidence is committed on `research/gpu-jax`, and the GPU pools are back
at 0 nodes.
