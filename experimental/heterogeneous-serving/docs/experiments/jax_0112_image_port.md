# Port of hetero/ to the jax 0.11.2 image

OLD: vLLM 0ba2aa35a, tpu-inference v0.26.0, jax 0.10.2, libtpu 0.0.43, py3.12.
NEW: vLLM 273a04964 (`0.32.0rc1.dev176+g273a04964`), tpu-inference 672546d8, jax/jaxlib 0.11.2, flax 0.12.10,
libtpu 0.0.48, ray 2.59.0, py3.12. Mirror env: `~/hetero-wt/JaxImage-mirror/src/{vllm,tpu_inference}-{old,new}`
(outside the worktree, so Ray job uploads stay small) + `.venv-image`
(CPU; `JAX_PLATFORMS=cpu`). Paths below: `V-o/V-n` = vLLM old/new, `T-o/T-n` = tpu_inference old/new.

## Broken items (all fixed on `infra/jax-0112-image`)

| # | hetero | upstream change (OLD -> NEW) | fix |
|---|---|---|---|
| 1 | scheduler.py:278 `("waiting","skipped_waiting")` | `skipped_waiting` removed; `kv_holding_waiting` queue (V-n scheduler.py:227) and `deferred_waiting` set (:230) | iterate `("waiting","kv_holding_waiting")`; `deferred_waiting` is stats-only, untouched |
| 2 | scheduler.py:317 `_preempt_request(request, timestamp)` | V-n scheduler.py:1571 adds `drop_stale_output`, called with it at :836 (TypeError on any KV-pressure preemption) | signature `(request, timestamp, *args, **kwargs)`, forwarded |
| 3 | engine_actor.py:1171,1397 `scheduler._skip_zero_block_ids` | attribute and drain logic removed (V-o scheduler.py:308,1196); drain now `_get_new_block_ids_to_zero` (V-n :1563) | HeteroScheduler owns `_skip_zero_block_ids` (scheduler.py:43) and overrides `_get_new_block_ids_to_zero` (:305) with the old contract (filter, then clear; untouched when zeroing not needed) |
| 4 | kv_transport.py:130 `spec.storage_block_size` | removed from KVCacheSpec; physical page = `num_states` (V-n kv_cache_interface.py:228, T-n runner/kv_cache_manager.py:991) | `int(spec.num_states)` (== block_size for attention specs); test fake updated |
| 5 | exec_bank.py `install_runner`: closure cell `jitted_model_fn` | step jit built lazily per KV-sharding key: T-n models/common/model_loader.py `wrapped_model_fn`:505 -> `_resolve_step_fn`:500 -> `_get_step_fn`:405 -> `_wrap_with_jit`:382 (cache `_step_fns`:403); `step_fn_no_options` twin only for continue_decode (default off) | `Installation.install_step_fn_wrapper` replaces the `_wrap_with_jit` cell so the options-carrying jit is banked on creation (second KV-sharding set counted in `unbanked_step_fns`, reported); loaded tables applied lazily; `restore()` unwraps cached entries. `WorkerBank` uses `known()`/`set_capture`/`load_table` |
| 6 | exec_bank.py JAX_CONFIG_KEYS | `jax_exec_time_optimization_effort`, `jax_memory_fitting_effort` no longer exist in jax 0.11.2 | removed the two keys (identity hash of other keys unchanged) |
| 7 | preread.py:68 `device_table` | T-n runner/tpu_runner.py:986 `_create_2d_mesh`: with `sharding_config.device_indexes` builds an unsorted `Mesh`, no `make_optimized_mesh` | `device_table(..., device_indexes=)` refuses that case (never guess); worker passes it (tpu_worker.py:415) |
| 8 | tpu_worker.py:523, preread.py:592, tests | upstream `_get_nnx_model` now `nnx.eval_shape(..., graph_updates=False)` (T-n model_loader.py:195,271) | same kwarg in all hetero mirrors |
| 9 | restore_graph.py `cheap_tracer_sharding_errors` | jax 0.11.2 `Tracer.sharding` no longer builds the origin message | kept (flag stays): property swapped for the block with identical message/type, behaviour-identical; test rewritten to assert no `_origin_msg` walk and exact restore |
| 10 | scheduler.py:667-684 continuation refusal | SamplingParams gained `watermarking`, `trace_decode_token_ids`, `prompt_logprob_token_ids`, `prompt_logprob_start`, `stream_interval` | non-default `watermarking`/`trace_decode_token_ids`/`prompt_logprob_token_ids` refused like the other untransferable state |

Comment-only drift refreshed: tpu_worker.py:977 (weight_utils 742/769 -> 759/786), preread.py:589, prespawn.py:9,291,321.

## Compatibility table (unchanged / changed-compatible)

| hetero file:line | upstream symbol | OLD | NEW | class |
|---|---|---|---|---|
| tpu_worker.py init_device/determine_available_memory/initialize_from_config/compile_or_warm_up_model/load_model | `TPUWorker` methods | T-o worker/tpu_worker.py | T-n :311,:512,:711,:670,:667 (same signatures; `BASE_JAX_PORT`->`BASE_TPU_PROCESS_PORT` only in DP/PP paths hetero does not use) | unchanged |
| tpu_worker.py:512-558, preread.py:583-605 | `model_loader._get_nnx_model(model_class, vllm_config, rng, mesh, pooler, is_draft_model)`, `get_model_loader`, `LoadableWithIterator`, `apply_qwix_on_abstract_model` | model_loader.py | model_loader.py:120-127,:283-285 | unchanged |
| tpu_worker.py:980-983 hooks | `weight_utils.jax_array_from_reshaped_torch`, `assign_and_shard_param`; `unquantized.py:42` | :742,:769 | :759,:786 | unchanged (bodies identical) |
| tpu_worker.py:1061 | `DefaultModelLoader.get_all_weights/load_weights` | default_loader.py | same entry points, `*args/**kwargs` forwarded | changed-compatible |
| tpu_worker.py:167-195 | `kv_cache_manager.create_kv_caches` module global | kv_cache_manager.py | :53 (called :992) | unchanged |
| kv_transport.py, tpu_worker.py omp_kv_* | `runner.kv_caches` flat list, `layer_name_to_kvcache_index`, `kv_cache_config` | tpu_runner.py:844 | :844 | changed-compatible (slot sharing allocator rewritten; real Gemma4 layout unverified) |
| tpu_worker.py:889-972 | `CachedRequestState(NewRequestData)` | input_batch.py:26 | :27 (new fields defaulted) | unchanged |
| tpu_worker.py:844-886 | `runner.requests`, `_pre_async_results`, `_modify_prev_results`, `uses_mrope` | tpu_runner.py | :838,:1491,:1042 | unchanged |
| exec_bank.py | `compute_logits_fn`, `sample`, `compute_and_gather_logprobs`, `_select_from_array_fn`, `_substitute_placeholder_token_fn`, `structured_decode_fn`, `_run_compilation`, `_flush_compilations` | compilation_manager.py:126,:203 | model_loader.py:448; sampling.py:327,421,432; tpu_runner.py:61-64,561,839,2385; structured_decoding_manager.py:33; compilation_manager.py:150-252 (adds `compile_only`, `_sync_hosts` barriers) | changed-compatible |
| compile_overlap.py:132-200 | `compiler._cache_read`, `backend_compile_and_load`, `compilation_cache.get_executable_and_time/_get_cache/extract_executable_and_time/decompress_executable`, `stages.Lowered.compile`, jax.monitoring listeners | jax 0.10.2 | jax 0.11.2 (signatures probed live; `get_executable_and_time` gains `host_callbacks=()`, already handled) | changed-compatible |
| exec_bank.py:134,279,426-455 | `xla_client.get_topology_for_devices`, `jit._jit_info`, `Compiled._executable/_params.const_args/_unloaded_executable.device_list/xla_executable.local_devices`, `serialize_executable.serialize/deserialize_and_load(execution_devices=)` | | all present on 0.11.2 | unchanged |
| scripts/overlap_replay.py:23-98, overlap_source_probe.py | `compiler._get_cache_key`, `compile_or_get_cached(..., executable_devices)`, `xla_bridge._backends`, `topologies.get_topology_desc/get_attached_topology`, `distributed.global_state.process_id` | | present | unchanged |
| weight_snapshot.py:234,358 | `jax.Array._npy_value`, `_single_device_array_to_np_array_did_copy` | | jax array.py:612 | unchanged |
| weight_snapshot.py:316,478 | `runner.state` leaf paths | gemma4 `qkv_proj` | `q_proj/k_proj/v_proj` | changed-compatible (old snapshots refused by context/path check: regenerate) |
| engine_actor.py:423-470 | `EngineArgs` kwargs, `additional_config` keys, `create_engine_config`, `LLMEngine(...)`, in-proc `engine_core` | arg_utils.py | same | unchanged (LLMEngine path source-read only) |
| engine_actor.py:457-459 | `parallel_config.worker_cls`, `scheduler_config.scheduler_cls` post-creation overrides | | T-n tpu_platform.py:466,:527 | unchanged for TP-only (DP>1 would install DPScheduler; hetero overwrites it: layouts are TP only) |
| engine_actor.py:1712 / prespawn.py:323-337 | `Executor.get_class`, `RayDistributedExecutorV2(MambaPoolSyncExecutorMixin, RayExecutorV2)`, `ray_executor_v2.get_bundles_sorted_by_node` patch, `_get_actor_resource_kwargs` | ray_executor_v2.py:255 | :276 (T-n override :138) | changed-compatible |
| engine_actor.py:1454 | `PauseState`, `scheduler.pause_state`, `batch_queue` | | interface.py:24 | unchanged |
| scheduler.py (class) | `AsyncScheduler`: `schedule(throttle_prefills=False)`, `update_from_output`, `add_request`, `RequestQueue.prepend_request/remove_requests`, `create_request_queue`, `Request.num_in_flight_tokens/num_output_placeholders/external_req_id` | | present | unchanged / changed-compatible |
| scheduler.py:409,635; engine_actor.py:575 | `len(scheduler.waiting)` | skipped_waiting separate | kv_holding_waiting separate | unchanged semantics |
| env/config | `VLLM_XLA_CACHE_PATH` (T-n compilation_manager.py:91), `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`, `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION` (deprecation log), `VLLM_ENABLE_V1_MULTIPROCESSING`, `JAX_COMPILATION_CACHE_DIR`, `JAX_PERSISTENT_CACHE_*` | | present | unchanged |
| native_backend_patch.json | | | owned by NativePatch | out of scope |

## Verification

- Control `.venv` (jax 0.11.2, no vllm/tpu_inference): baseline 8 failed / 9 errors (pre-existing: `results/` data files absent in worktree, test_restore_graph tracer test written for jax 0.10, test_restore_tail_worker collection). After: 7 failed (failures that remain are the same missing-data tests) / 9 errors.
- `.venv-image` (JAX_PLATFORMS=cpu): all `hetero.*` modules import; real-base-class subclassing works; new `tests/test_scheduler_native.py` (5 tests, real vLLM Scheduler: fence/unfence resume, waiting-queue restore, preemption with `drop_stale_output`, skip-zero contract) passes; test_exec_bank, test_portable_bank, test_restore_graph, test_restore_tail_worker (against real TPUWorker/model_loader, patched only inside setUpModule/tearDownModule), test_kv_transport, test_first_switch, test_preread, test_weight_snapshot pass file by file.
- Whole-suite run in `.venv-image` still has order-dependent failures (test_preread parent-backend, restore_graph placement, restore_tail_worker): `tests/test_batch_size_optimizer.py` (imports admission_control/ortools) initialises a 1-device CPU backend first; each file passes alone. Other image-only failures are missing packages (`matplotlib`, `envoy`) and missing `results/` files, not port issues.

## Not verifiable without a TPU (cluster smoke targets)

1. `omp_kv_layout` / export / import on real Gemma4: slot-sharing allocator (T-n kv_cache_manager.py:806-1021) may change array count/index map; `_omp_kv_num_blocks` can differ from `kv_caches[i].shape[0]`.
2. exec bank on the real step jit: lazily built `_wrap_with_jit` banking, `unbanked_step_fns == 0`, relocation, AOT flow with `compile_only` and the new `_sync_hosts` barriers (every rank must reach `compile_step`/`flush_pre_warmup` equally under bank read).
3. `jax.make_mesh`/`create_device_mesh` equality 0.10.2 vs 0.11.2 on a v5e subslice (preread `device_table`).
4. LLMEngine construction, in-process engine_core, RayExecutorV2 prespawn with CPU-only actors (empty accelerator ids), model-runner V1/V2 choice on TPU, DP>1.
5. Gemma4 weight snapshots must be regenerated (state tree `q/k/v_proj`); fast_graph on the real 31B graph (`graph_updates=False`, 0.11.2 Tracer).
6. Scripts' hard-coded head pod names (`tpu-ray-cluster-vllm-tpu-head-*`) are cluster-specific.
