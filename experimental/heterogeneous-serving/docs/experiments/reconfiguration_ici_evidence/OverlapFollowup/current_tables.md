# Reconfiguration results — audited tables

Inputs: `/home/hongtao/hetero-wt/FastSnapshot/results/restore/FastSnapshot`, `/home/hongtao/hetero-wt/OverlapCompile/results/overlap/OverlapCompile`; throughput table `/home/hongtao/hetero-research/results/reconfig/throughput_table_measured.json`.
Rows marked `NOT ACCEPTED` fail an Auditor gate (see script docstring); they are kept so negative data is not lost.

Acceptance is scoped to the row, not full reconfiguration. Warm cache names do not establish valid controls. Cold/reused snapshot labels describe destination pages; reused live sources may be PJRT-cached. Fingerprint validation is outside timers and is not in-path checksum verification.

Audit restore: `/home/hongtao/hetero-wt/AuditFollowup/results/audit/followup_restore_claims.json`; source manifest matched=True; scope=SUPPORTED_S_INDEPENDENT_RESTORE_CORRECTNESS_COMPONENT_TIMING_ONLY.

Snapshot includes publication. snapshot_device_placement includes chip grouping, device_put, all per-chip block_until_ready, array reconstruction and final readiness, but excludes earlier manifest verification, mmap preparation, generation vote and full load_model tail. Fingerprints/token qualification are outside those timers. Per-worker extrema are not full reconfiguration wall time.

All eight workers restore manifest v2 checksum_kind=none. checksums_requested=true but verified_checksums=false and verified_bytes=0. Independent outside-timer fingerprints establish byte correctness; no in-path checksum throughput is measured here.

Read-only local reanalysis; no cluster calls or reruns.

Independent fingerprint chain (outside snapshot/restore timers): cold files == live source == independently restored live weights; 8 workers, 32896 shards, 251587795712 bytes; audit inputs matched=True.

Audit checksum: `/home/hongtao/hetero-wt/AuditFollowup/results/audit/followup_checksum_claims.json`; source manifest matched=True; scope=SUPPORTED_SYNTHETIC_WEAK_REDUCTION_COST_ONLY.

Four chips, 1,028 real TP8 shard shapes per chip, 31,448,474,464 synthetic source bytes; two repetitions over the same generated arrays, not live model weights or independent experiment repeats.

snapshot_s includes completion and atomic publication. Inspect each worker phase_semantics for overlapping stages. Cold/reused describe destination pages; host-cache and did_copy evidence distinguish source DMA.

snapshot_read_tmpfs measures mmap construction, not full physical file reads. Device placement includes grouping, JAX puts, reassembly and completion, not pure DMA. load_model additionally includes CPU model/graph construction; initialization and sampling/gather compilation are separate startup costs. v2 checksum_kind=none performs no byte checksum in the restore timer; validation-only full SHA and W0 token gates are separate.

## Synthetic helper components — NOT full snapshot/restore

Pinned DMA excludes copied ndarray extraction and shared-file publication; ordinary collector includes ndarray acquisition. Weak sum/XOR is noncryptographic and position-insensitive. SHA256 of its output vector is provenance, not source SHA256. No extrapolation from synthetic arrays to live model snapshots or physical bandwidth ceilings.

| component | repeat | bytes | host-consumed s (DMA: ready-only) | kernel-ready incl enqueue s | readback s | GB/s | timing boundary | component verdict |
|---|---|---|---|---|---|---|---|---|
| async_native_collector | 0 | 268435456 | 0.059191 | — | — | 4.535100582164916 | source generation/readiness, destination allocation/faulting, compilation and verification excluded; all D2H completions included; ndarray acquisition included; file publication excluded | accepted |
| async_native_collector | 1 | 268435456 | 0.058529 | — | — | 4.58636120506156 | source generation/readiness, destination allocation/faulting, compilation and verification excluded; all D2H completions included; ndarray acquisition included; file publication excluded | accepted |
| preallocated_pinned_donation | 0 | 268435456 | 0.019522 | — | — | 13.750731570171503 | source generation/readiness, destination allocation/faulting, compilation and verification excluded; all D2H completions included; ndarray acquisition/copy and final shared-file publication excluded | accepted |
| preallocated_pinned_donation | 1 | 268435456 | 0.018833 | — | — | 14.253227716444476 | source generation/readiness, destination allocation/faulting, compilation and verification excluded; all D2H completions included; ndarray acquisition/copy and final shared-file publication excluded | accepted |
| async_native_collector | 0 | 1073741824 | 0.095154 | — | — | 11.284276985290344 | source generation/readiness, destination allocation/faulting, compilation and verification excluded; all D2H completions included; ndarray acquisition included; file publication excluded | accepted |
| async_native_collector | 1 | 1073741824 | 0.094517 | — | — | 11.360270156520745 | source generation/readiness, destination allocation/faulting, compilation and verification excluded; all D2H completions included; ndarray acquisition included; file publication excluded | accepted |
| preallocated_pinned_donation | 0 | 1073741824 | 0.021094 | — | — | 50.90278522759417 | source generation/readiness, destination allocation/faulting, compilation and verification excluded; all D2H completions included; ndarray acquisition/copy and final shared-file publication excluded | accepted |
| preallocated_pinned_donation | 1 | 1073741824 | 0.019041 | — | — | 56.39195980947486 | source generation/readiness, destination allocation/faulting, compilation and verification excluded; all D2H completions included; ndarray acquisition/copy and final shared-file publication excluded | accepted |
| async_native_collector | 0 | 1073741824 | 0.233316 | — | — | 4.602092957957572 | source generation/readiness, destination allocation/faulting, compilation and verification excluded; all D2H completions included; ndarray acquisition included; file publication excluded | accepted |
| async_native_collector | 1 | 1073741824 | 0.237507 | — | — | 4.5208922111568555 | source generation/readiness, destination allocation/faulting, compilation and verification excluded; all D2H completions included; ndarray acquisition included; file publication excluded | accepted |
| preallocated_pinned_donation | 0 | 1073741824 | 0.074250 | — | — | 14.461214137290266 | source generation/readiness, destination allocation/faulting, compilation and verification excluded; all D2H completions included; ndarray acquisition/copy and final shared-file publication excluded | accepted |
| preallocated_pinned_donation | 1 | 1073741824 | 0.073357 | — | — | 14.63713631282727 | source generation/readiness, destination allocation/faulting, compilation and verification excluded; all D2H completions included; ndarray acquisition/copy and final shared-file publication excluded | accepted |
| async_native_collector | 0 | 4294967296 | 0.364191 | — | — | 11.793170648367791 | source generation/readiness, destination allocation/faulting, compilation and verification excluded; all D2H completions included; ndarray acquisition included; file publication excluded | accepted |
| async_native_collector | 1 | 4294967296 | 0.365282 | — | — | 11.757940515765176 | source generation/readiness, destination allocation/faulting, compilation and verification excluded; all D2H completions included; ndarray acquisition included; file publication excluded | accepted |
| preallocated_pinned_donation | 0 | 4294967296 | 0.075960 | — | — | 56.54232379189189 | source generation/readiness, destination allocation/faulting, compilation and verification excluded; all D2H completions included; ndarray acquisition/copy and final shared-file publication excluded | accepted |
| preallocated_pinned_donation | 1 | 4294967296 | 0.073678 | — | — | 58.29359695303055 | source generation/readiness, destination allocation/faulting, compilation and verification excluded; all D2H completions included; ndarray acquisition/copy and final shared-file publication excluded | accepted |
| synthetic weak sum/XOR reduction | 0 | 31448474464 | 0.732827 | 0.27257891598856077 | 0.4602483339840546 | None | Kernel-ready wall includes Python enqueue and waits for all 4,112 outputs. The following ndarray conversion/list materialization and readback is separately timed and required for host consumption. The excluded compile interval includes both generators and reducers together, not checksum-only compile cost. Analytic comparison and output-digest hashing occur after the readback timer. | accepted |
| synthetic weak sum/XOR reduction | 1 | 31448474464 | 0.734976 | 0.26567890599835664 | 0.4692966930451803 | None | Kernel-ready wall includes Python enqueue and waits for all 4,112 outputs. The following ndarray conversion/list materialization and readback is separately timed and required for host consumption. The excluded compile interval includes both generators and reducers together, not checksum-only compile cost. Analytic comparison and output-digest hashing occur after the readback timer. | accepted |

## FastSnapshot component bandwidth — `bandwidth-corrected.json` (submission raysubmit_yg9HDtkEvZ3B4ZSd, measurements.ok=False, failed_ops=['host_copy/model/chips4'])

| op | host | bytes | elapsed s (median, min–max) | GB/s | gate verdict |
|---|---|---|---|---|---|
| d2h/model/chip0/serial_asarray | gke-tpu-8770cb66-8v5c | 7862118616 | 2.284 | 3.44 | accepted |
| d2h/model/chip0/async_all_asarray | gke-tpu-8770cb66-8v5c | 7862118616 | 1.553 | 5.06 | accepted |
| d2h/model/chip0/device_get_tree | gke-tpu-8770cb66-8v5c | 7862118616 | 1.398 | 5.63 | accepted |
| d2h/model/chip0/async_did_copy | gke-tpu-8770cb66-8v5c | 7862118616 | 1.375 | 5.72 | accepted |
| d2h/model/chip0/pinned_host | gke-tpu-8770cb66-8v5c | 7862118616 | 3.427 | 2.29 | accepted |
| d2h/model/chip0/serial_contiguous | gke-tpu-8770cb66-8v5c | 7862118616 | 2.265 | 3.47 | accepted |
| d2h/model/chip0/async_contiguous | gke-tpu-8770cb66-8v5c | 7862118616 | 1.495 | 5.26 | accepted |
| d2h/model/chip0/pool8_contiguous | gke-tpu-8770cb66-8v5c | 7862118616 | 0.629 | 12.50 | accepted |
| d2h/model/chips4/serial_asarray | gke-tpu-8770cb66-8v5c | 31448474464 | 8.886 | 3.54 | accepted |
| d2h/model/chips4/async_all_asarray | gke-tpu-8770cb66-8v5c | 31448474464 | 2.893 | 10.87 | accepted |
| d2h/model/chips4/device_get_tree | gke-tpu-8770cb66-8v5c | 31448474464 | 2.632 | 11.95 | accepted |
| d2h/model/chips4/chip_threads_asarray | gke-tpu-8770cb66-8v5c | 31448474464 | 2.691 | 11.69 | accepted |
| d2h/model/chips4/chip_threads_did_copy | gke-tpu-8770cb66-8v5c | 31448474464 | 3.286 | 9.57 | accepted |
| d2h/model/chips4/async_did_copy | gke-tpu-8770cb66-8v5c | 31448474464 | 2.680 | 11.73 | accepted |
| d2h/model/chips4/pinned_host | gke-tpu-8770cb66-8v5c | 31448474464 | 8.340 | 3.77 | accepted |
| d2h/model/chips4/serial_contiguous | gke-tpu-8770cb66-8v5c | 31448474464 | 9.180 | 3.43 | accepted |
| d2h/model/chips4/async_contiguous | gke-tpu-8770cb66-8v5c | 31448474464 | 2.556 | 12.30 | accepted |
| d2h/model/chips4/chip_threads_contiguous | gke-tpu-8770cb66-8v5c | 31448474464 | 2.662 | 11.81 | accepted |
| d2h/model/chips4/pool8_contiguous | gke-tpu-8770cb66-8v5c | 31448474464 | 2.452 | 12.83 | accepted |
| d2h/model/chip1/serial_asarray | gke-tpu-8770cb66-8v5c | 7862118616 | 2.032 | 3.87 | accepted |
| d2h/model/chip1/async_did_copy | gke-tpu-8770cb66-8v5c | 7862118616 | 1.311 | 6.00 | accepted |
| d2h/model/chip1/serial_contiguous | gke-tpu-8770cb66-8v5c | 7862118616 | 2.098 | 3.75 | accepted |
| d2h/model/chip2/serial_asarray | gke-tpu-8770cb66-8v5c | 7862118616 | 1.989 | 3.95 | accepted |
| d2h/model/chip2/async_did_copy | gke-tpu-8770cb66-8v5c | 7862118616 | 1.351 | 5.82 | accepted |
| d2h/model/chip2/serial_contiguous | gke-tpu-8770cb66-8v5c | 7862118616 | 2.281 | 3.45 | accepted |
| d2h/model/chip3/serial_asarray | gke-tpu-8770cb66-8v5c | 7862118616 | 2.147 | 3.66 | accepted |
| d2h/model/chip3/async_did_copy | gke-tpu-8770cb66-8v5c | 7862118616 | 1.330 | 5.91 | accepted |
| d2h/model/chip3/serial_contiguous | gke-tpu-8770cb66-8v5c | 7862118616 | 2.217 | 3.55 | accepted |
| d2h/large_268435456/chip0/serial_asarray | gke-tpu-8770cb66-8v5c | 268435456 | 0.059 | 4.51 | accepted |
| d2h/large_268435456/chip0/device_get_tree | gke-tpu-8770cb66-8v5c | 268435456 | 0.058 | 4.60 | accepted |
| d2h/large_268435456/chip0/async_did_copy | gke-tpu-8770cb66-8v5c | 268435456 | 0.062 | 4.30 | accepted |
| d2h/large_268435456/chip0/pinned_host | gke-tpu-8770cb66-8v5c | 268435456 | 0.204 | 1.32 | accepted |
| d2h/large_268435456/chip0/serial_contiguous | gke-tpu-8770cb66-8v5c | 268435456 | 0.061 | 4.42 | accepted |
| d2h/large_268435456/chip0/async_contiguous | gke-tpu-8770cb66-8v5c | 268435456 | 0.061 | 4.39 | accepted |
| d2h/large_268435456/chips4/serial_asarray | gke-tpu-8770cb66-8v5c | 1073741824 | 0.262 | 4.10 | accepted |
| d2h/large_268435456/chips4/device_get_tree | gke-tpu-8770cb66-8v5c | 1073741824 | 0.082 | 13.03 | accepted |
| d2h/large_268435456/chips4/chip_threads_asarray | gke-tpu-8770cb66-8v5c | 1073741824 | 0.093 | 11.60 | accepted |
| d2h/large_268435456/chips4/async_did_copy | gke-tpu-8770cb66-8v5c | 1073741824 | 0.090 | 12.00 | accepted |
| d2h/large_268435456/chips4/pinned_host | gke-tpu-8770cb66-8v5c | 1073741824 | 0.422 | 2.54 | accepted |
| d2h/large_268435456/chips4/serial_contiguous | gke-tpu-8770cb66-8v5c | 1073741824 | 0.271 | 3.96 | accepted |
| d2h/large_268435456/chips4/async_contiguous | gke-tpu-8770cb66-8v5c | 1073741824 | 0.092 | 11.68 | accepted |
| d2h/large_268435456/chips4/chip_threads_contiguous | gke-tpu-8770cb66-8v5c | 1073741824 | 0.099 | 10.84 | accepted |
| d2h/large_1073741824/chip0/serial_asarray | gke-tpu-8770cb66-8v5c | 1073741824 | 0.248 | 4.33 | accepted |
| d2h/large_1073741824/chip0/device_get_tree | gke-tpu-8770cb66-8v5c | 1073741824 | 0.239 | 4.49 | accepted |
| d2h/large_1073741824/chip0/async_did_copy | gke-tpu-8770cb66-8v5c | 1073741824 | 0.240 | 4.48 | accepted |
| d2h/large_1073741824/chip0/pinned_host | gke-tpu-8770cb66-8v5c | 1073741824 | 0.702 | 1.53 | accepted |
| d2h/large_1073741824/chip0/serial_contiguous | gke-tpu-8770cb66-8v5c | 1073741824 | 0.245 | 4.38 | accepted |
| d2h/large_1073741824/chip0/async_contiguous | gke-tpu-8770cb66-8v5c | 1073741824 | 0.238 | 4.51 | accepted |
| d2h/large_1073741824/chips4/serial_asarray | gke-tpu-8770cb66-8v5c | 4294967296 | 0.979 | 4.39 | accepted |
| d2h/large_1073741824/chips4/device_get_tree | gke-tpu-8770cb66-8v5c | 4294967296 | 0.352 | 12.22 | accepted |
| d2h/large_1073741824/chips4/chip_threads_asarray | gke-tpu-8770cb66-8v5c | 4294967296 | 0.358 | 11.99 | accepted |
| d2h/large_1073741824/chips4/async_did_copy | gke-tpu-8770cb66-8v5c | 4294967296 | 0.342 | 12.57 | accepted |
| d2h/large_1073741824/chips4/pinned_host | gke-tpu-8770cb66-8v5c | 4294967296 | 1.299 | 3.31 | accepted |
| d2h/large_1073741824/chips4/serial_contiguous | gke-tpu-8770cb66-8v5c | 4294967296 | 0.961 | 4.47 | accepted |
| d2h/large_1073741824/chips4/async_contiguous | gke-tpu-8770cb66-8v5c | 4294967296 | 0.319 | 13.46 | accepted |
| d2h/large_1073741824/chips4/chip_threads_contiguous | gke-tpu-8770cb66-8v5c | 4294967296 | 0.347 | 12.36 | accepted |
| d2h/large_4294967296/chip0/serial_asarray | gke-tpu-8770cb66-8v5c | 4294967296 | 1.975 | 2.17 | accepted |
| d2h/large_4294967296/chip0/device_get_tree | gke-tpu-8770cb66-8v5c | 4294967296 | 1.974 | 2.18 | accepted |
| d2h/large_4294967296/chip0/async_did_copy | gke-tpu-8770cb66-8v5c | 4294967296 | 1.906 | 2.25 | accepted |
| d2h/large_4294967296/chip0/pinned_host | gke-tpu-8770cb66-8v5c | 4294967296 | 1.459 | 2.94 | accepted |
| d2h/large_4294967296/chip0/serial_contiguous | gke-tpu-8770cb66-8v5c | 4294967296 | 1.936 | 2.22 | accepted |
| d2h/large_4294967296/chip0/async_contiguous | gke-tpu-8770cb66-8v5c | 4294967296 | 1.972 | 2.18 | accepted |
| d2h/large_4294967296/chips4/serial_asarray | gke-tpu-8770cb66-8v5c | 17179869184 | 7.517 | 2.29 | accepted |
| d2h/large_4294967296/chips4/device_get_tree | gke-tpu-8770cb66-8v5c | 17179869184 | 2.559 | 6.71 | accepted |
| d2h/large_4294967296/chips4/chip_threads_asarray | gke-tpu-8770cb66-8v5c | 17179869184 | 2.520 | 6.82 | accepted |
| d2h/large_4294967296/chips4/async_did_copy | gke-tpu-8770cb66-8v5c | 17179869184 | 2.649 | 6.49 | accepted |
| d2h/large_4294967296/chips4/pinned_host | gke-tpu-8770cb66-8v5c | 17179869184 | 3.826 | 4.49 | accepted |
| d2h/large_4294967296/chips4/serial_contiguous | gke-tpu-8770cb66-8v5c | 17179869184 | 7.708 | 2.23 | accepted |
| d2h/large_4294967296/chips4/async_contiguous | gke-tpu-8770cb66-8v5c | 17179869184 | 2.624 | 6.55 | accepted |
| d2h/large_4294967296/chips4/chip_threads_contiguous | gke-tpu-8770cb66-8v5c | 17179869184 | 2.658 | 6.46 | accepted |
| h2d/model/chip0/put_all_ready_end/anon | gke-tpu-8770cb66-8v5c | 7862118616 | 0.583 | 13.49 | accepted |
| h2d/model/chip0/tree_put_per_chip/anon | gke-tpu-8770cb66-8v5c | 7862118616 | 0.734 | 10.71 | accepted |
| h2d/model/chips4/put_all_ready_end/anon | gke-tpu-8770cb66-8v5c | 31448474464 | 1.690 | 18.60 | accepted |
| h2d/model/chips4/tree_put_per_chip/anon | gke-tpu-8770cb66-8v5c | 31448474464 | 1.821 | 17.27 | accepted |
| h2d/model/chips4/chip_threads_put/anon | gke-tpu-8770cb66-8v5c | 31448474464 | 1.373 | 22.90 | accepted |
| h2d/model/chip0/put_all_ready_end/mmap | gke-tpu-8770cb66-8v5c | 7862118616 | 0.597 | 13.17 | accepted |
| h2d/model/chip0/tree_put_per_chip/mmap | gke-tpu-8770cb66-8v5c | 7862118616 | 0.660 | 11.92 | accepted |
| h2d/model/chips4/put_all_ready_end/mmap | gke-tpu-8770cb66-8v5c | 31448474464 | 1.610 | 19.54 | accepted |
| h2d/model/chips4/tree_put_per_chip/mmap | gke-tpu-8770cb66-8v5c | 31448474464 | 2.168 | 14.50 | accepted |
| h2d/model/chips4/chip_threads_put/mmap | gke-tpu-8770cb66-8v5c | 31448474464 | 1.408 | 22.33 | accepted |
| h2d/model/chip1/put_all_ready_end/anon | gke-tpu-8770cb66-8v5c | 7862118616 | 0.570 | 13.79 | accepted |
| h2d/model/chip2/put_all_ready_end/anon | gke-tpu-8770cb66-8v5c | 7862118616 | 0.578 | 13.61 | accepted |
| h2d/model/chip3/put_all_ready_end/anon | gke-tpu-8770cb66-8v5c | 7862118616 | 0.577 | 13.62 | accepted |
| h2d/large_268435456/chip0/put_all_ready_end/anon | gke-tpu-8770cb66-8v5c | 268435456 | 0.047 | 5.76 | accepted |
| h2d/large_268435456/chips4/put_all_ready_end/anon | gke-tpu-8770cb66-8v5c | 1073741824 | 0.065 | 16.62 | accepted |
| h2d/large_268435456/chips4/chip_threads_put/anon | gke-tpu-8770cb66-8v5c | 1073741824 | 0.065 | 16.41 | accepted |
| h2d/large_1073741824/chip0/put_all_ready_end/anon | gke-tpu-8770cb66-8v5c | 1073741824 | 0.182 | 5.90 | accepted |
| h2d/large_1073741824/chips4/put_all_ready_end/anon | gke-tpu-8770cb66-8v5c | 4294967296 | 0.252 | 17.03 | accepted |
| h2d/large_1073741824/chips4/chip_threads_put/anon | gke-tpu-8770cb66-8v5c | 4294967296 | 0.253 | 16.99 | accepted |
| h2d/large_4294967296/chip0/put_all_ready_end/anon | gke-tpu-8770cb66-8v5c | 4294967296 | 1.476 | 2.91 | accepted |
| h2d/large_4294967296/chips4/put_all_ready_end/anon | gke-tpu-8770cb66-8v5c | 17179869184 | 1.931 | 8.90 | accepted |
| h2d/large_4294967296/chips4/chip_threads_put/anon | gke-tpu-8770cb66-8v5c | 17179869184 | 1.908 | 9.00 | accepted |
| host_copy/model/chips4/first_touch/flat_contiguous_source/t1/none | gke-tpu-8770cb66-8v5c | 31448474464 | 19.601 | 1.60 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/flat_contiguous_source/t1/sha256 | gke-tpu-8770cb66-8v5c | 31448474464 | 38.420 | 0.82 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/flat_contiguous_source/t1/zlib_crc32 | gke-tpu-8770cb66-8v5c | 31448474464 | 27.648 | 1.14 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/flat_contiguous_source/t1/xxhash | gke-tpu-8770cb66-8v5c | 31448474464 | 20.968 | 1.50 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/flat_contiguous_source/t1/crc32c | gke-tpu-8770cb66-8v5c | 31448474464 | — | — | NOT ACCEPTED: cell_ok, durable (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/flat_contiguous_source/t4/none | gke-tpu-8770cb66-8v5c | 31448474464 | 5.057 | 6.22 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/flat_contiguous_source/t4/sha256 | gke-tpu-8770cb66-8v5c | 31448474464 | 9.839 | 3.20 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/flat_contiguous_source/t4/zlib_crc32 | gke-tpu-8770cb66-8v5c | 31448474464 | 7.035 | 4.47 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/flat_contiguous_source/t4/xxhash | gke-tpu-8770cb66-8v5c | 31448474464 | 5.380 | 5.85 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/flat_contiguous_source/t4/crc32c | gke-tpu-8770cb66-8v5c | 31448474464 | — | — | NOT ACCEPTED: cell_ok, durable (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/copyto_strided/t1/none | gke-tpu-8770cb66-8v5c | 31448474464 | 19.832 | 1.59 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/copyto_strided/t4/none | gke-tpu-8770cb66-8v5c | 31448474464 | 5.062 | 6.21 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t1/none | gke-tpu-8770cb66-8v5c | 31448474464 | 2.329 | 13.50 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t1/sha256 | gke-tpu-8770cb66-8v5c | 31448474464 | 21.187 | 1.48 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t1/zlib_crc32 | gke-tpu-8770cb66-8v5c | 31448474464 | 10.292 | 3.06 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t1/xxhash | gke-tpu-8770cb66-8v5c | 31448474464 | 3.812 | 8.25 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t1/crc32c | gke-tpu-8770cb66-8v5c | 31448474464 | — | — | NOT ACCEPTED: cell_ok, durable, no_noop (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t4/none | gke-tpu-8770cb66-8v5c | 31448474464 | 0.770 | 40.83 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t4/sha256 | gke-tpu-8770cb66-8v5c | 31448474464 | 5.359 | 5.87 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t4/zlib_crc32 | gke-tpu-8770cb66-8v5c | 31448474464 | 2.647 | 11.88 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t4/xxhash | gke-tpu-8770cb66-8v5c | 31448474464 | 1.177 | 26.71 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t4/crc32c | gke-tpu-8770cb66-8v5c | 31448474464 | — | — | NOT ACCEPTED: cell_ok, durable, no_noop (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/copyto_strided/t1/none | gke-tpu-8770cb66-8v5c | 31448474464 | 2.499 | 12.59 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/copyto_strided/t4/none | gke-tpu-8770cb66-8v5c | 31448474464 | 0.816 | 38.53 | accepted (host copy only; not a snapshot time) |

## FastSnapshot component bandwidth — `bandwidth-first-run.json` (submission raysubmit_3rjDgzZcCENJzF2Y, measurements.ok=False, failed_ops=['d2h/model/chip0/serial_asarray', 'd2h/model/chip0/async_all_asarray', 'd2h/model/chip0/device_get_tree', 'd2h/model/chip0/async_did_copy', 'd2h/model/chip0/pinned_host', 'd2h/model/chips4/serial_asarray', 'd2h/model/chips4/async_all_asarray', 'd2h/model/chips4/device_get_tree', 'd2h/model/chips4/chip_threads_asarray', 'd2h/model/chips4/chip_threads_did_copy', 'd2h/model/chips4/async_did_copy', 'd2h/model/chips4/pinned_host', 'd2h/model/chip1/serial_asarray', 'd2h/model/chip1/async_did_copy', 'd2h/model/chip2/serial_asarray', 'd2h/model/chip2/async_did_copy', 'd2h/model/chip3/serial_asarray', 'd2h/model/chip3/async_did_copy', 'host_copy/model/chips4'])

| op | host | bytes | elapsed s (median, min–max) | GB/s | gate verdict |
|---|---|---|---|---|---|
| d2h/model/chip0/serial_asarray | gke-tpu-8770cb66-8v5c | 7862118616 | 2.316 | 3.39 | NOT ACCEPTED: op_ok, d2h_complete |
| d2h/model/chip0/async_all_asarray | gke-tpu-8770cb66-8v5c | 7862118616 | 1.578 | 4.98 | NOT ACCEPTED: op_ok, d2h_complete |
| d2h/model/chip0/device_get_tree | gke-tpu-8770cb66-8v5c | 7862118616 | 1.316 | 5.98 | NOT ACCEPTED: op_ok, d2h_complete |
| d2h/model/chip0/async_did_copy | gke-tpu-8770cb66-8v5c | 7862118616 | 1.270 | 6.19 | NOT ACCEPTED: op_ok, d2h_complete |
| d2h/model/chip0/pinned_host | gke-tpu-8770cb66-8v5c | 7862118616 | 3.365 | 2.34 | NOT ACCEPTED: op_ok, d2h_complete |
| d2h/model/chips4/serial_asarray | gke-tpu-8770cb66-8v5c | 31448474464 | 8.633 | 3.64 | NOT ACCEPTED: op_ok, d2h_complete |
| d2h/model/chips4/async_all_asarray | gke-tpu-8770cb66-8v5c | 31448474464 | 2.592 | 12.13 | NOT ACCEPTED: op_ok, d2h_complete |
| d2h/model/chips4/device_get_tree | gke-tpu-8770cb66-8v5c | 31448474464 | 2.295 | 13.70 | NOT ACCEPTED: op_ok, d2h_complete |
| d2h/model/chips4/chip_threads_asarray | gke-tpu-8770cb66-8v5c | 31448474464 | 2.674 | 11.76 | NOT ACCEPTED: op_ok, d2h_complete |
| d2h/model/chips4/chip_threads_did_copy | gke-tpu-8770cb66-8v5c | 31448474464 | 2.924 | 10.76 | NOT ACCEPTED: op_ok, d2h_complete |
| d2h/model/chips4/async_did_copy | gke-tpu-8770cb66-8v5c | 31448474464 | 2.278 | 13.80 | NOT ACCEPTED: op_ok, d2h_complete |
| d2h/model/chips4/pinned_host | gke-tpu-8770cb66-8v5c | 31448474464 | 17.253 | 1.82 | NOT ACCEPTED: op_ok, d2h_complete |
| d2h/model/chip1/serial_asarray | gke-tpu-8770cb66-8v5c | 7862118616 | 2.073 | 3.79 | NOT ACCEPTED: op_ok, d2h_complete |
| d2h/model/chip1/async_did_copy | gke-tpu-8770cb66-8v5c | 7862118616 | 1.234 | 6.37 | NOT ACCEPTED: op_ok, d2h_complete |
| d2h/model/chip2/serial_asarray | gke-tpu-8770cb66-8v5c | 7862118616 | 2.107 | 3.73 | NOT ACCEPTED: op_ok, d2h_complete |
| d2h/model/chip2/async_did_copy | gke-tpu-8770cb66-8v5c | 7862118616 | 1.267 | 6.21 | NOT ACCEPTED: op_ok, d2h_complete |
| d2h/model/chip3/serial_asarray | gke-tpu-8770cb66-8v5c | 7862118616 | 2.046 | 3.84 | NOT ACCEPTED: op_ok, d2h_complete |
| d2h/model/chip3/async_did_copy | gke-tpu-8770cb66-8v5c | 7862118616 | 1.308 | 6.01 | NOT ACCEPTED: op_ok, d2h_complete |
| d2h/large_268435456/chip0/serial_asarray | gke-tpu-8770cb66-8v5c | 268435456 | 0.059 | 4.57 | accepted |
| d2h/large_268435456/chip0/device_get_tree | gke-tpu-8770cb66-8v5c | 268435456 | 0.065 | 4.15 | accepted |
| d2h/large_268435456/chip0/async_did_copy | gke-tpu-8770cb66-8v5c | 268435456 | 0.064 | 4.22 | accepted |
| d2h/large_268435456/chip0/pinned_host | gke-tpu-8770cb66-8v5c | 268435456 | 0.122 | 2.19 | accepted |
| d2h/large_268435456/chips4/serial_asarray | gke-tpu-8770cb66-8v5c | 1073741824 | 0.246 | 4.37 | accepted |
| d2h/large_268435456/chips4/device_get_tree | gke-tpu-8770cb66-8v5c | 1073741824 | 0.084 | 12.77 | accepted |
| d2h/large_268435456/chips4/chip_threads_asarray | gke-tpu-8770cb66-8v5c | 1073741824 | 0.092 | 11.69 | accepted |
| d2h/large_268435456/chips4/async_did_copy | gke-tpu-8770cb66-8v5c | 1073741824 | 0.082 | 13.13 | accepted |
| d2h/large_268435456/chips4/pinned_host | gke-tpu-8770cb66-8v5c | 1073741824 | 0.268 | 4.01 | accepted |
| d2h/large_1073741824/chip0/serial_asarray | gke-tpu-8770cb66-8v5c | 1073741824 | 0.240 | 4.48 | accepted |
| d2h/large_1073741824/chip0/device_get_tree | gke-tpu-8770cb66-8v5c | 1073741824 | 0.242 | 4.45 | accepted |
| d2h/large_1073741824/chip0/async_did_copy | gke-tpu-8770cb66-8v5c | 1073741824 | 0.242 | 4.44 | accepted |
| d2h/large_1073741824/chip0/pinned_host | gke-tpu-8770cb66-8v5c | 1073741824 | 0.369 | 2.91 | accepted |
| d2h/large_1073741824/chips4/serial_asarray | gke-tpu-8770cb66-8v5c | 4294967296 | 0.954 | 4.50 | accepted |
| d2h/large_1073741824/chips4/device_get_tree | gke-tpu-8770cb66-8v5c | 4294967296 | 0.324 | 13.27 | accepted |
| d2h/large_1073741824/chips4/chip_threads_asarray | gke-tpu-8770cb66-8v5c | 4294967296 | 0.355 | 12.10 | accepted |
| d2h/large_1073741824/chips4/async_did_copy | gke-tpu-8770cb66-8v5c | 4294967296 | 0.321 | 13.37 | accepted |
| d2h/large_1073741824/chips4/pinned_host | gke-tpu-8770cb66-8v5c | 4294967296 | 1.062 | 4.04 | accepted |
| d2h/large_4294967296/chip0/serial_asarray | gke-tpu-8770cb66-8v5c | 4294967296 | 2.025 | 2.12 | accepted |
| d2h/large_4294967296/chip0/device_get_tree | gke-tpu-8770cb66-8v5c | 4294967296 | 1.984 | 2.16 | accepted |
| d2h/large_4294967296/chip0/async_did_copy | gke-tpu-8770cb66-8v5c | 4294967296 | 2.010 | 2.14 | accepted |
| d2h/large_4294967296/chip0/pinned_host | gke-tpu-8770cb66-8v5c | 4294967296 | 1.426 | 3.01 | accepted |
| d2h/large_4294967296/chips4/serial_asarray | gke-tpu-8770cb66-8v5c | 17179869184 | 8.224 | 2.09 | accepted |
| d2h/large_4294967296/chips4/device_get_tree | gke-tpu-8770cb66-8v5c | 17179869184 | 7.413 | 2.32 | accepted |
| d2h/large_4294967296/chips4/chip_threads_asarray | gke-tpu-8770cb66-8v5c | 17179869184 | 10.409 | 1.65 | accepted |
| d2h/large_4294967296/chips4/async_did_copy | gke-tpu-8770cb66-8v5c | 17179869184 | 2.659 | 6.46 | accepted |
| d2h/large_4294967296/chips4/pinned_host | gke-tpu-8770cb66-8v5c | 17179869184 | 3.944 | 4.36 | accepted |
| h2d/model/chip0/put_all_ready_end/anon | gke-tpu-8770cb66-8v5c | 7862118616 | 0.640 | 12.28 | accepted |
| h2d/model/chip0/tree_put_per_chip/anon | gke-tpu-8770cb66-8v5c | 7862118616 | 0.681 | 11.55 | accepted |
| h2d/model/chips4/put_all_ready_end/anon | gke-tpu-8770cb66-8v5c | 31448474464 | 1.564 | 20.11 | accepted |
| h2d/model/chips4/tree_put_per_chip/anon | gke-tpu-8770cb66-8v5c | 31448474464 | 2.310 | 13.61 | accepted |
| h2d/model/chips4/chip_threads_put/anon | gke-tpu-8770cb66-8v5c | 31448474464 | 1.392 | 22.59 | accepted |
| h2d/model/chip0/put_all_ready_end/mmap | gke-tpu-8770cb66-8v5c | 7862118616 | 0.570 | 13.80 | accepted |
| h2d/model/chip0/tree_put_per_chip/mmap | gke-tpu-8770cb66-8v5c | 7862118616 | 0.699 | 11.24 | accepted |
| h2d/model/chips4/put_all_ready_end/mmap | gke-tpu-8770cb66-8v5c | 31448474464 | 1.581 | 19.89 | accepted |
| h2d/model/chips4/tree_put_per_chip/mmap | gke-tpu-8770cb66-8v5c | 31448474464 | 2.326 | 13.52 | accepted |
| h2d/model/chips4/chip_threads_put/mmap | gke-tpu-8770cb66-8v5c | 31448474464 | 1.403 | 22.42 | accepted |
| h2d/large_268435456/chip0/put_all_ready_end/anon | gke-tpu-8770cb66-8v5c | 268435456 | 0.047 | 5.76 | accepted |
| h2d/large_268435456/chips4/put_all_ready_end/anon | gke-tpu-8770cb66-8v5c | 1073741824 | 0.067 | 16.04 | accepted |
| h2d/large_268435456/chips4/chip_threads_put/anon | gke-tpu-8770cb66-8v5c | 1073741824 | 0.067 | 16.02 | accepted |
| h2d/large_1073741824/chip0/put_all_ready_end/anon | gke-tpu-8770cb66-8v5c | 1073741824 | 0.174 | 6.17 | accepted |
| h2d/large_1073741824/chips4/put_all_ready_end/anon | gke-tpu-8770cb66-8v5c | 4294967296 | 0.258 | 16.62 | accepted |
| h2d/large_1073741824/chips4/chip_threads_put/anon | gke-tpu-8770cb66-8v5c | 4294967296 | 0.255 | 16.83 | accepted |
| h2d/large_4294967296/chip0/put_all_ready_end/anon | gke-tpu-8770cb66-8v5c | 4294967296 | 1.463 | 2.94 | accepted |
| h2d/large_4294967296/chips4/put_all_ready_end/anon | gke-tpu-8770cb66-8v5c | 17179869184 | 1.854 | 9.27 | accepted |
| h2d/large_4294967296/chips4/chip_threads_put/anon | gke-tpu-8770cb66-8v5c | 17179869184 | 1.946 | 8.83 | accepted |

## FastSnapshot component bandwidth — `bandwidth-host-copy.json` (submission raysubmit_CfWN2G7k4YKJN63g, measurements.ok=True, failed_ops=[])

| op | host | bytes | elapsed s (median, min–max) | GB/s | gate verdict |
|---|---|---|---|---|---|
| host_copy/model/chips4/first_touch/flat_contiguous_source/t1/none | gke-tpu-8770cb66-8v5c | 31448474464 | 19.501 | 1.61 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/flat_contiguous_source/t1/sha256 | gke-tpu-8770cb66-8v5c | 31448474464 | 38.441 | 0.82 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/flat_contiguous_source/t1/zlib_crc32 | gke-tpu-8770cb66-8v5c | 31448474464 | 27.512 | 1.14 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/flat_contiguous_source/t1/xxhash | gke-tpu-8770cb66-8v5c | 31448474464 | 20.982 | 1.50 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/flat_contiguous_source/t1/crc32c | gke-tpu-8770cb66-8v5c | 31448474464 | 40.012 | 0.79 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/flat_contiguous_source/t4/none | gke-tpu-8770cb66-8v5c | 31448474464 | 5.031 | 6.25 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/flat_contiguous_source/t4/sha256 | gke-tpu-8770cb66-8v5c | 31448474464 | 9.697 | 3.24 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/flat_contiguous_source/t4/zlib_crc32 | gke-tpu-8770cb66-8v5c | 31448474464 | 6.979 | 4.51 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/flat_contiguous_source/t4/xxhash | gke-tpu-8770cb66-8v5c | 31448474464 | 5.553 | 5.66 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/flat_contiguous_source/t4/crc32c | gke-tpu-8770cb66-8v5c | 31448474464 | 24.250 | 1.30 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/copyto_strided/t1/none | gke-tpu-8770cb66-8v5c | 31448474464 | 19.818 | 1.59 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/first_touch/copyto_strided/t4/none | gke-tpu-8770cb66-8v5c | 31448474464 | 5.115 | 6.15 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t1/none | gke-tpu-8770cb66-8v5c | 31448474464 | 2.303 | 13.66 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t1/sha256 | gke-tpu-8770cb66-8v5c | 31448474464 | 21.075 | 1.49 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t1/zlib_crc32 | gke-tpu-8770cb66-8v5c | 31448474464 | 10.290 | 3.06 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t1/xxhash | gke-tpu-8770cb66-8v5c | 31448474464 | 3.722 | 8.45 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t1/crc32c | gke-tpu-8770cb66-8v5c | 31448474464 | 22.785 | 1.38 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t4/none | gke-tpu-8770cb66-8v5c | 31448474464 | 0.766 | 41.07 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t4/sha256 | gke-tpu-8770cb66-8v5c | 31448474464 | 5.337 | 5.89 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t4/zlib_crc32 | gke-tpu-8770cb66-8v5c | 31448474464 | 2.629 | 11.96 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t4/xxhash | gke-tpu-8770cb66-8v5c | 31448474464 | 1.152 | 27.29 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/flat_contiguous_source/t4/crc32c | gke-tpu-8770cb66-8v5c | 31448474464 | 19.295 | 1.63 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/copyto_strided/t1/none | gke-tpu-8770cb66-8v5c | 31448474464 | 2.479 | 12.68 | accepted (host copy only; not a snapshot time) |
| host_copy/model/chips4/reused/copyto_strided/t4/none | gke-tpu-8770cb66-8v5c | 31448474464 | 0.806 | 39.00 | accepted (host copy only; not a snapshot time) |

## Snapshot write time per worker

### `S-cold.json` (cold, submission raysubmit_fVXpJbxkRCVZkcUy)

| engine | rank | bytes | snapshot_s | GB/s | parts (max per-chip active; overlapping) | baseline_s | source | scope | checksum kind | cold snapshot proof | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| small_a | 0 | 31448474464 | 14.44 | 2.18 | metadata_s=0.06, enqueue_s=0.44, transfer_s=14.27, transfer_wait_s=3.98, allocation_s=6.99, copy_s=1.08, checksum_s=0.00, flush_s=2.89, publish_s=0.11 | 51.5 | cold (Python cache) | cold-destination snapshot publication | none | True | accepted |
| small_a | 1 | 31448474464 | 14.28 | 2.20 | metadata_s=0.06, enqueue_s=0.55, transfer_s=14.11, transfer_wait_s=4.81, allocation_s=6.58, copy_s=1.07, checksum_s=0.00, flush_s=2.67, publish_s=0.11 | 50.7 | cold (Python cache) | cold-destination snapshot publication | none | True | accepted |
| small_b | 0 | 31448474464 | 14.79 | 2.13 | metadata_s=0.06, enqueue_s=0.40, transfer_s=14.62, transfer_wait_s=4.35, allocation_s=6.83, copy_s=1.13, checksum_s=0.00, flush_s=3.50, publish_s=0.11 | 51.0 | cold (Python cache) | cold-destination snapshot publication | none | True | accepted |
| small_b | 1 | 31448474464 | 14.48 | 2.17 | metadata_s=0.06, enqueue_s=0.48, transfer_s=14.30, transfer_wait_s=4.16, allocation_s=7.10, copy_s=0.98, checksum_s=0.00, flush_s=2.93, publish_s=0.11 | 50.7 | cold (Python cache) | cold-destination snapshot publication | none | True | accepted |
| small_c | 0 | 31448474464 | 14.92 | 2.11 | metadata_s=0.07, enqueue_s=0.61, transfer_s=14.74, transfer_wait_s=6.23, allocation_s=6.67, copy_s=0.89, checksum_s=0.00, flush_s=2.79, publish_s=0.11 | 51.0 | cold (Python cache) | cold-destination snapshot publication | none | True | accepted |
| small_c | 1 | 31448474464 | 14.32 | 2.20 | metadata_s=0.07, enqueue_s=0.84, transfer_s=14.14, transfer_wait_s=4.73, allocation_s=6.82, copy_s=1.02, checksum_s=0.00, flush_s=3.13, publish_s=0.11 | 50.4 | cold (Python cache) | cold-destination snapshot publication | none | True | accepted |
| small_d | 0 | 31448474464 | 14.00 | 2.25 | metadata_s=0.07, enqueue_s=0.70, transfer_s=13.82, transfer_wait_s=4.22, allocation_s=6.75, copy_s=1.04, checksum_s=0.00, flush_s=2.85, publish_s=0.11 | 51.3 | cold (Python cache) | cold-destination snapshot publication | none | True | accepted |
| small_d | 1 | 31448474464 | 14.26 | 2.21 | metadata_s=0.07, enqueue_s=0.52, transfer_s=14.08, transfer_wait_s=4.58, allocation_s=6.54, copy_s=1.04, checksum_s=0.00, flush_s=2.85, publish_s=0.11 | 50.9 | cold (Python cache) | cold-destination snapshot publication | none | True | accepted |

Accepted workers: 8/8; snapshot_s 14.38 (14.00–14.92, n=8); metadata_s 0.07 (0.06–0.07, n=8); enqueue_s 0.53 (0.40–0.84, n=8); transfer_s 14.20 (13.82–14.74, n=8); transfer_wait_s 4.47 (3.98–6.23, n=8); allocation_s 6.78 (6.54–7.10, n=8); copy_s 1.04 (0.89–1.13, n=8); checksum_s 0.00 (0.00–0.00, n=8); flush_s 2.87 (2.67–3.50, n=8); publish_s 0.11 (0.11–0.11, n=8)

### `S-reused.json` (reused, submission raysubmit_mWRMgN65RxDsBQ69)

| engine | rank | bytes | snapshot_s | GB/s | parts (max per-chip active; overlapping) | baseline_s | source | scope | checksum kind | cold snapshot proof | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| small_a | 0 | 31448474464 | 3.65 | 8.61 | metadata_s=0.04, enqueue_s=0.01, transfer_s=3.50, transfer_wait_s=0.01, allocation_s=1.68, copy_s=0.82, checksum_s=0.00, flush_s=1.05, publish_s=0.12 | 51.5 | possibly warm | reused-destination publication; PJRT source possibly cached, not cold D2H | none | False | accepted |
| small_a | 1 | 31448474464 | 3.56 | 8.84 | metadata_s=0.04, enqueue_s=0.01, transfer_s=3.40, transfer_wait_s=0.01, allocation_s=1.59, copy_s=0.82, checksum_s=0.00, flush_s=0.96, publish_s=0.12 | 50.7 | possibly warm | reused-destination publication; PJRT source possibly cached, not cold D2H | none | False | accepted |
| small_b | 0 | 31448474464 | 3.58 | 8.78 | metadata_s=0.04, enqueue_s=0.01, transfer_s=3.43, transfer_wait_s=0.01, allocation_s=1.61, copy_s=0.83, checksum_s=0.00, flush_s=1.02, publish_s=0.12 | 51.0 | possibly warm | reused-destination publication; PJRT source possibly cached, not cold D2H | none | False | accepted |
| small_b | 1 | 31448474464 | 3.61 | 8.70 | metadata_s=0.04, enqueue_s=0.01, transfer_s=3.46, transfer_wait_s=0.01, allocation_s=1.67, copy_s=0.82, checksum_s=0.00, flush_s=1.01, publish_s=0.12 | 50.7 | possibly warm | reused-destination publication; PJRT source possibly cached, not cold D2H | none | False | accepted |
| small_c | 0 | 31448474464 | 3.60 | 8.73 | metadata_s=0.04, enqueue_s=0.01, transfer_s=3.45, transfer_wait_s=0.01, allocation_s=1.63, copy_s=0.82, checksum_s=0.00, flush_s=1.01, publish_s=0.12 | 51.0 | possibly warm | reused-destination publication; PJRT source possibly cached, not cold D2H | none | False | accepted |
| small_c | 1 | 31448474464 | 3.56 | 8.84 | metadata_s=0.03, enqueue_s=0.01, transfer_s=3.41, transfer_wait_s=0.01, allocation_s=1.57, copy_s=0.83, checksum_s=0.00, flush_s=1.01, publish_s=0.11 | 50.4 | possibly warm | reused-destination publication; PJRT source possibly cached, not cold D2H | none | False | accepted |
| small_d | 0 | 31448474464 | 3.63 | 8.66 | metadata_s=0.04, enqueue_s=0.01, transfer_s=3.47, transfer_wait_s=0.01, allocation_s=1.62, copy_s=0.83, checksum_s=0.00, flush_s=1.02, publish_s=0.12 | 51.3 | possibly warm | reused-destination publication; PJRT source possibly cached, not cold D2H | none | False | accepted |
| small_d | 1 | 31448474464 | 3.61 | 8.72 | metadata_s=0.04, enqueue_s=0.01, transfer_s=3.45, transfer_wait_s=0.01, allocation_s=1.62, copy_s=0.82, checksum_s=0.00, flush_s=1.02, publish_s=0.12 | 50.9 | possibly warm | reused-destination publication; PJRT source possibly cached, not cold D2H | none | False | accepted |

Accepted workers: 8/8; snapshot_s 3.61 (3.56–3.65, n=8); metadata_s 0.04 (0.03–0.04, n=8); enqueue_s 0.01 (0.01–0.01, n=8); transfer_s 3.45 (3.40–3.50, n=8); transfer_wait_s 0.01 (0.01–0.01, n=8); allocation_s 1.62 (1.57–1.68, n=8); copy_s 0.82 (0.82–0.83, n=8); checksum_s 0.00 (0.00–0.00, n=8); flush_s 1.01 (0.96–1.05, n=8); publish_s 0.12 (0.11–0.12, n=8)

## Restore / start-up time per worker

### `S-restored-validation.json` (submission raysubmit_GaBwU9R5aSsRx1bB) greedy_equality={'passed': True, 'prompts_per_engine': 32, 'engines': 4, 'reference': '/home/hongtao/hetero-wt/ShardedRestore/results/restore/ShardedRestore/S-reference.json', 'comparison': 'output_token_ids, token for token'}

| engine | rank | phases s | load_model minus snapshot_* s | restored | in-path checksum | outside-timer fingerprint | gate verdict |
|---|---|---|---|---|---|---|---|
| small_a | 0 | init_device=2.8, load_model=19.1, snapshot_manifest_verify=0.1, snapshot_read_tmpfs=0.8, snapshot_verify=0.0, snapshot_generation_vote=0.0, snapshot_device_placement=1.5, initialize_from_config=25.7, compile_or_warm_up_model=62.0 | 16.8 | True | False | True | accepted |
| small_a | 1 | init_device=2.8, load_model=19.2, snapshot_manifest_verify=0.1, snapshot_read_tmpfs=0.7, snapshot_verify=0.0, snapshot_generation_vote=0.0, snapshot_device_placement=1.5, initialize_from_config=25.7, compile_or_warm_up_model=62.0 | 16.8 | True | False | True | accepted |
| small_b | 0 | init_device=3.9, load_model=18.9, snapshot_manifest_verify=0.1, snapshot_read_tmpfs=0.8, snapshot_verify=0.0, snapshot_generation_vote=0.0, snapshot_device_placement=1.4, initialize_from_config=25.6, compile_or_warm_up_model=62.1 | 16.6 | True | False | True | accepted |
| small_b | 1 | init_device=3.8, load_model=19.0, snapshot_manifest_verify=0.1, snapshot_read_tmpfs=0.8, snapshot_verify=0.0, snapshot_generation_vote=0.0, snapshot_device_placement=1.4, initialize_from_config=25.6, compile_or_warm_up_model=62.1 | 16.6 | True | False | True | accepted |
| small_c | 0 | init_device=3.9, load_model=19.0, snapshot_manifest_verify=0.1, snapshot_read_tmpfs=0.8, snapshot_verify=0.0, snapshot_generation_vote=0.0, snapshot_device_placement=1.5, initialize_from_config=25.4, compile_or_warm_up_model=61.6 | 16.6 | True | False | True | accepted |
| small_c | 1 | init_device=3.9, load_model=19.0, snapshot_manifest_verify=0.1, snapshot_read_tmpfs=0.8, snapshot_verify=0.0, snapshot_generation_vote=0.0, snapshot_device_placement=1.4, initialize_from_config=25.4, compile_or_warm_up_model=61.6 | 16.6 | True | False | True | accepted |
| small_d | 0 | init_device=2.8, load_model=19.3, snapshot_manifest_verify=0.1, snapshot_read_tmpfs=0.8, snapshot_verify=0.0, snapshot_generation_vote=0.0, snapshot_device_placement=1.4, initialize_from_config=25.7, compile_or_warm_up_model=62.3 | 16.9 | True | False | True | accepted |
| small_d | 1 | init_device=2.8, load_model=19.3, snapshot_manifest_verify=0.1, snapshot_read_tmpfs=0.8, snapshot_verify=0.0, snapshot_generation_vote=0.0, snapshot_device_placement=1.5, initialize_from_config=25.7, compile_or_warm_up_model=62.3 | 16.9 | True | False | True | accepted |

Accepted workers 8/8: init_device 3.3 (2.8–3.9, n=8); load_model 19.1 (18.9–19.3, n=8); snapshot_manifest_verify 0.1 (0.1–0.1, n=8); snapshot_read_tmpfs 0.8 (0.7–0.8, n=8); snapshot_verify 0.0 (0.0–0.0, n=8); snapshot_generation_vote 0.0 (0.0–0.0, n=8); snapshot_device_placement 1.4 (1.4–1.5, n=8); initialize_from_config 25.6 (25.4–25.7, n=8); compile_or_warm_up_model 62.0 (61.6–62.3, n=8)

## Warm-up / KV split and cache hits (OverlapCompile startup profiles)

Accepted startup rows qualify host-call/configuration observations only, not loaded-runtime/source equivalence or device attribution. Unknown legacy counts remain unknown, not zero.

Compact projection: `/home/hongtao/hetero-wt/OverlapCompile/results/overlap/OverlapCompile/startup_projection.json`; source hashes matched=True.

Unprojected archived startup artifacts (not accepted or used; raw paths retained): /home/hongtao/hetero-wt/OverlapCompile/results/overlap/OverlapCompile/controlled-hs-baseline-20261002a/source-provision-deploy.json; /home/hongtao/hetero-wt/OverlapCompile/results/overlap/OverlapCompile/controlled-hs-baseline-20261002a/target-provision-deploy.json; /home/hongtao/hetero-wt/OverlapCompile/results/overlap/OverlapCompile/s-split-20261002c/cold-restore-deploy.json; /home/hongtao/hetero-wt/OverlapCompile/results/overlap/OverlapCompile/s-split-20261002c/normal-deploy.json; /home/hongtao/hetero-wt/OverlapCompile/results/overlap/OverlapCompile/s-split-20261002c/warm-restore-deploy.json

### `results/overlap/OverlapCompile/controlled-hs-baseline-20261002b/source-provision-deploy.json` (submission raysubmit_xAHhA56wgaH6y83u)

| engine | rank | phase | wall s | exclusive split s | hits/misses (hook) | executables | device exec s | cache | qualification | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| large_a | 0 | initialize_from_config | 5.6 | executable_load=0.2, cache_lookup_and_decompress=3.5, cache_load=0.0, backend_compile_and_load=0.0, trace=0.4, lower=1.0, warmup_execute_and_sync=0.4, allocation_and_sync=0.1, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_a | 1 | initialize_from_config | 5.6 | executable_load=0.2, cache_lookup_and_decompress=3.6, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=0.9, warmup_execute_and_sync=0.4, allocation_and_sync=0.0, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_a | 2 | initialize_from_config | 5.6 | executable_load=0.2, cache_lookup_and_decompress=3.5, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.1, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_a | 3 | initialize_from_config | 5.6 | executable_load=0.2, cache_lookup_and_decompress=3.5, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=1.0, warmup_execute_and_sync=0.4, allocation_and_sync=0.1, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 0 | initialize_from_config | 5.2 | executable_load=0.2, cache_lookup_and_decompress=3.2, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.1, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 1 | initialize_from_config | 5.2 | executable_load=0.2, cache_lookup_and_decompress=3.1, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.1, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 2 | initialize_from_config | 5.2 | executable_load=0.2, cache_lookup_and_decompress=3.2, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=0.9, warmup_execute_and_sync=0.3, allocation_and_sync=0.1, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 3 | initialize_from_config | 5.2 | executable_load=0.2, cache_lookup_and_decompress=3.1, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=0.9, warmup_execute_and_sync=0.3, allocation_and_sync=0.1, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |

### `results/overlap/OverlapCompile/controlled-hs-baseline-20261002b/source-warm-deploy.json` (submission raysubmit_Nx9RmYiTLpkfbypL)

| engine | rank | phase | wall s | exclusive split s | hits/misses (hook) | executables | device exec s | cache | qualification | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| large_a | 0 | initialize_from_config | 3.9 | executable_load=0.2, cache_lookup_and_decompress=1.8, cache_load=0.0, backend_compile_and_load=0.0, trace=0.4, lower=1.0, warmup_execute_and_sync=0.4, allocation_and_sync=0.0, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_a | 1 | initialize_from_config | 3.9 | executable_load=0.2, cache_lookup_and_decompress=1.8, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.1, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_a | 2 | initialize_from_config | 3.9 | executable_load=0.2, cache_lookup_and_decompress=0.2, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=1.0, warmup_execute_and_sync=1.9, allocation_and_sync=0.1, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_a | 3 | initialize_from_config | 3.9 | executable_load=0.2, cache_lookup_and_decompress=0.2, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=0.9, warmup_execute_and_sync=2.0, allocation_and_sync=0.1, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 0 | initialize_from_config | 3.2 | executable_load=0.2, cache_lookup_and_decompress=0.2, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=1.0, warmup_execute_and_sync=1.2, allocation_and_sync=0.0, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 1 | initialize_from_config | 3.2 | executable_load=0.2, cache_lookup_and_decompress=1.1, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.0, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 2 | initialize_from_config | 3.2 | executable_load=0.2, cache_lookup_and_decompress=0.2, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=0.9, warmup_execute_and_sync=1.3, allocation_and_sync=0.0, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 3 | initialize_from_config | 3.2 | executable_load=0.2, cache_lookup_and_decompress=0.2, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=1.0, warmup_execute_and_sync=1.3, allocation_and_sync=0.0, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |

### `results/overlap/OverlapCompile/controlled-hs-baseline-20261002b/target-cold-deploy.json` (submission raysubmit_u8V83uUSecZifGj7)

| engine | rank | phase | wall s | exclusive split s | hits/misses (hook) | executables | device exec s | cache | qualification | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| small_a | 0 | initialize_from_config | 32.0 | executable_load=0.0, cache_lookup_and_decompress=1.7, cache_load=0.0, backend_compile_and_load=23.4, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.1, aot_compile=3.6, other=1.6 | 4/73 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_a | 1 | initialize_from_config | 32.0 | executable_load=0.0, cache_lookup_and_decompress=1.4, cache_load=0.0, backend_compile_and_load=23.5, trace=0.4, lower=1.0, warmup_execute_and_sync=0.4, allocation_and_sync=0.1, aot_compile=3.4, other=1.9 | 0/77 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_b | 0 | initialize_from_config | 30.9 | executable_load=0.0, cache_lookup_and_decompress=1.8, cache_load=0.0, backend_compile_and_load=22.6, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.2, aot_compile=3.2, other=1.4 | 8/69 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_b | 1 | initialize_from_config | 30.9 | executable_load=0.0, cache_lookup_and_decompress=1.8, cache_load=0.0, backend_compile_and_load=22.6, trace=0.4, lower=1.0, warmup_execute_and_sync=0.4, allocation_and_sync=0.1, aot_compile=3.4, other=1.2 | 11/66 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_c | 0 | initialize_from_config | 30.7 | executable_load=0.0, cache_lookup_and_decompress=1.8, cache_load=0.0, backend_compile_and_load=22.5, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.1, aot_compile=3.3, other=1.4 | 11/66 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_c | 1 | initialize_from_config | 30.7 | executable_load=0.0, cache_lookup_and_decompress=1.7, cache_load=0.0, backend_compile_and_load=22.3, trace=0.4, lower=1.0, warmup_execute_and_sync=0.5, allocation_and_sync=0.1, aot_compile=3.4, other=1.4 | 12/65 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_d | 0 | initialize_from_config | 30.2 | executable_load=0.0, cache_lookup_and_decompress=1.8, cache_load=0.0, backend_compile_and_load=22.4, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.2, aot_compile=3.1, other=1.0 | 16/61 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_d | 1 | initialize_from_config | 30.2 | executable_load=0.0, cache_lookup_and_decompress=2.1, cache_load=0.0, backend_compile_and_load=22.2, trace=0.4, lower=1.0, warmup_execute_and_sync=0.4, allocation_and_sync=0.2, aot_compile=3.1, other=0.8 | 21/56 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |

### `results/overlap/OverlapCompile/controlled-hs-baseline-20261002b/target-provision-deploy.json` (submission raysubmit_N7BWZauphLnU2DY8)

| engine | rank | phase | wall s | exclusive split s | hits/misses (hook) | executables | device exec s | cache | qualification | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| small_a | 0 | initialize_from_config | 32.3 | executable_load=0.1, cache_lookup_and_decompress=2.8, cache_load=0.0, backend_compile_and_load=9.3, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.4, aot_compile=9.9, other=8.1 | 42/35 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_a | 1 | initialize_from_config | 32.3 | executable_load=0.0, cache_lookup_and_decompress=1.8, cache_load=0.0, backend_compile_and_load=23.1, trace=0.4, lower=1.0, warmup_execute_and_sync=1.0, allocation_and_sync=0.4, aot_compile=3.3, other=1.3 | 11/66 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_b | 0 | initialize_from_config | 30.7 | executable_load=0.1, cache_lookup_and_decompress=2.9, cache_load=0.0, backend_compile_and_load=8.2, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.5, aot_compile=10.0, other=7.3 | 54/23 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_b | 1 | initialize_from_config | 30.7 | executable_load=0.0, cache_lookup_and_decompress=1.8, cache_load=0.0, backend_compile_and_load=21.5, trace=0.4, lower=1.0, warmup_execute_and_sync=0.4, allocation_and_sync=0.4, aot_compile=4.0, other=1.2 | 13/64 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_c | 0 | initialize_from_config | 32.4 | executable_load=0.1, cache_lookup_and_decompress=2.4, cache_load=0.0, backend_compile_and_load=9.2, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.5, aot_compile=9.9, other=8.7 | 32/45 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_c | 1 | initialize_from_config | 32.4 | executable_load=0.0, cache_lookup_and_decompress=1.5, cache_load=0.0, backend_compile_and_load=23.4, trace=0.4, lower=1.0, warmup_execute_and_sync=0.6, allocation_and_sync=0.4, aot_compile=3.4, other=1.8 | 2/75 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_d | 0 | initialize_from_config | 31.3 | executable_load=0.2, cache_lookup_and_decompress=3.2, cache_load=0.0, backend_compile_and_load=8.3, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.4, aot_compile=10.2, other=7.3 | 58/19 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_d | 1 | initialize_from_config | 31.3 | executable_load=0.0, cache_lookup_and_decompress=1.7, cache_load=0.0, backend_compile_and_load=22.6, trace=0.4, lower=1.0, warmup_execute_and_sync=0.6, allocation_and_sync=0.4, aot_compile=3.3, other=1.2 | 12/65 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |

### `results/overlap/OverlapCompile/controlled-hs-baseline-20261002b/target-warm-deploy.json` (submission raysubmit_g7rEFEzbdHfikkMR)

| engine | rank | phase | wall s | exclusive split s | hits/misses (hook) | executables | device exec s | cache | qualification | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| small_a | 0 | initialize_from_config | 4.8 | executable_load=0.2, cache_lookup_and_decompress=1.3, cache_load=0.0, backend_compile_and_load=0.0, trace=0.4, lower=1.0, warmup_execute_and_sync=1.8, allocation_and_sync=0.0, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_a | 1 | initialize_from_config | 4.8 | executable_load=0.2, cache_lookup_and_decompress=2.8, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.0, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_b | 0 | initialize_from_config | 4.0 | executable_load=0.2, cache_lookup_and_decompress=0.8, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=1.0, warmup_execute_and_sync=1.4, allocation_and_sync=0.1, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_b | 1 | initialize_from_config | 4.0 | executable_load=0.2, cache_lookup_and_decompress=2.0, cache_load=0.0, backend_compile_and_load=0.0, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.0, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_c | 0 | initialize_from_config | 4.8 | executable_load=0.2, cache_lookup_and_decompress=1.5, cache_load=0.0, backend_compile_and_load=0.0, trace=0.4, lower=1.0, warmup_execute_and_sync=1.6, allocation_and_sync=0.0, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_c | 1 | initialize_from_config | 4.8 | executable_load=0.2, cache_lookup_and_decompress=2.8, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.0, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_d | 0 | initialize_from_config | 4.9 | executable_load=0.2, cache_lookup_and_decompress=1.2, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=1.0, warmup_execute_and_sync=2.0, allocation_and_sync=0.0, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_d | 1 | initialize_from_config | 4.9 | executable_load=0.2, cache_lookup_and_decompress=2.8, cache_load=0.0, backend_compile_and_load=0.0, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.0, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |

### `results/overlap/OverlapCompile/controlled-hs-baseline-20261002b/source-provision-deploy.json` (submission raysubmit_xAHhA56wgaH6y83u)

| engine | rank | phase | wall s | exclusive split s | hits/misses (hook) | executables | device exec s | cache | qualification | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| large_a | 0 | compile_or_warm_up_model | 53.3 | executable_load=2.7, cache_lookup_and_decompress=9.3, cache_load=0.0, backend_compile_and_load=0.0, trace=16.6, lower=21.6, warmup_execute_and_sync=2.0, allocation_and_sync=0.0, aot_compile=0.1, other=1.0 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_a | 1 | compile_or_warm_up_model | 53.3 | executable_load=2.5, cache_lookup_and_decompress=9.8, cache_load=0.0, backend_compile_and_load=0.0, trace=16.4, lower=21.3, warmup_execute_and_sync=2.2, allocation_and_sync=0.0, aot_compile=0.1, other=1.0 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_a | 2 | compile_or_warm_up_model | 53.3 | executable_load=2.6, cache_lookup_and_decompress=9.7, cache_load=0.0, backend_compile_and_load=0.0, trace=16.4, lower=21.1, warmup_execute_and_sync=2.4, allocation_and_sync=0.0, aot_compile=0.1, other=1.0 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_a | 3 | compile_or_warm_up_model | 53.3 | executable_load=2.5, cache_lookup_and_decompress=9.5, cache_load=0.0, backend_compile_and_load=0.0, trace=16.5, lower=21.6, warmup_execute_and_sync=2.0, allocation_and_sync=0.0, aot_compile=0.1, other=1.0 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 0 | compile_or_warm_up_model | 53.4 | executable_load=2.5, cache_lookup_and_decompress=7.6, cache_load=0.0, backend_compile_and_load=0.0, trace=16.7, lower=21.4, warmup_execute_and_sync=4.0, allocation_and_sync=0.0, aot_compile=0.1, other=1.0 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 1 | compile_or_warm_up_model | 53.4 | executable_load=2.6, cache_lookup_and_decompress=7.6, cache_load=0.0, backend_compile_and_load=0.0, trace=15.3, lower=22.2, warmup_execute_and_sync=4.4, allocation_and_sync=0.0, aot_compile=0.1, other=1.2 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 2 | compile_or_warm_up_model | 53.4 | executable_load=2.6, cache_lookup_and_decompress=7.5, cache_load=0.0, backend_compile_and_load=0.0, trace=16.3, lower=21.2, warmup_execute_and_sync=4.6, allocation_and_sync=0.0, aot_compile=0.1, other=1.1 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 3 | compile_or_warm_up_model | 53.4 | executable_load=2.6, cache_lookup_and_decompress=7.5, cache_load=0.0, backend_compile_and_load=0.0, trace=15.1, lower=22.0, warmup_execute_and_sync=4.9, allocation_and_sync=0.0, aot_compile=0.1, other=1.1 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |

### `results/overlap/OverlapCompile/controlled-hs-baseline-20261002b/source-warm-deploy.json` (submission raysubmit_Nx9RmYiTLpkfbypL)

| engine | rank | phase | wall s | exclusive split s | hits/misses (hook) | executables | device exec s | cache | qualification | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| large_a | 0 | compile_or_warm_up_model | 48.7 | executable_load=2.6, cache_lookup_and_decompress=4.5, cache_load=0.0, backend_compile_and_load=0.0, trace=16.2, lower=21.7, warmup_execute_and_sync=2.5, allocation_and_sync=0.0, aot_compile=0.1, other=1.0 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_a | 1 | compile_or_warm_up_model | 48.7 | executable_load=2.6, cache_lookup_and_decompress=4.6, cache_load=0.0, backend_compile_and_load=0.0, trace=15.7, lower=21.3, warmup_execute_and_sync=3.3, allocation_and_sync=0.0, aot_compile=0.1, other=1.1 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_a | 2 | compile_or_warm_up_model | 48.7 | executable_load=2.6, cache_lookup_and_decompress=3.1, cache_load=0.0, backend_compile_and_load=0.0, trace=15.8, lower=21.4, warmup_execute_and_sync=4.3, allocation_and_sync=0.0, aot_compile=0.1, other=1.3 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_a | 3 | compile_or_warm_up_model | 48.7 | executable_load=2.6, cache_lookup_and_decompress=3.6, cache_load=0.0, backend_compile_and_load=0.0, trace=15.9, lower=21.5, warmup_execute_and_sync=3.5, allocation_and_sync=0.0, aot_compile=0.1, other=1.3 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 0 | compile_or_warm_up_model | 48.2 | executable_load=2.5, cache_lookup_and_decompress=4.3, cache_load=0.0, backend_compile_and_load=0.0, trace=16.0, lower=21.9, warmup_execute_and_sync=2.4, allocation_and_sync=0.0, aot_compile=0.1, other=0.9 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 1 | compile_or_warm_up_model | 48.2 | executable_load=2.6, cache_lookup_and_decompress=1.1, cache_load=0.0, backend_compile_and_load=0.0, trace=15.9, lower=21.5, warmup_execute_and_sync=4.2, allocation_and_sync=0.0, aot_compile=0.1, other=2.7 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 2 | compile_or_warm_up_model | 48.2 | executable_load=2.5, cache_lookup_and_decompress=1.2, cache_load=0.0, backend_compile_and_load=0.0, trace=15.9, lower=21.5, warmup_execute_and_sync=4.3, allocation_and_sync=0.0, aot_compile=0.1, other=2.6 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 3 | compile_or_warm_up_model | 48.2 | executable_load=2.6, cache_lookup_and_decompress=1.1, cache_load=0.0, backend_compile_and_load=0.0, trace=15.8, lower=21.3, warmup_execute_and_sync=4.6, allocation_and_sync=0.0, aot_compile=0.1, other=2.7 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |

### `results/overlap/OverlapCompile/controlled-hs-baseline-20261002b/target-cold-deploy.json` (submission raysubmit_u8V83uUSecZifGj7)

| engine | rank | phase | wall s | exclusive split s | hits/misses (hook) | executables | device exec s | cache | qualification | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| small_a | 0 | compile_or_warm_up_model | 242.2 | executable_load=0.0, cache_lookup_and_decompress=3.5, cache_load=0.0, backend_compile_and_load=173.2, trace=15.8, lower=20.4, warmup_execute_and_sync=19.1, allocation_and_sync=0.0, aot_compile=7.3, other=2.9 | 5/179 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_a | 1 | compile_or_warm_up_model | 242.2 | executable_load=0.0, cache_lookup_and_decompress=3.5, cache_load=0.0, backend_compile_and_load=171.5, trace=15.8, lower=20.4, warmup_execute_and_sync=20.7, allocation_and_sync=0.0, aot_compile=7.3, other=2.9 | 1/183 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_b | 0 | compile_or_warm_up_model | 242.2 | executable_load=0.0, cache_lookup_and_decompress=3.8, cache_load=0.0, backend_compile_and_load=171.0, trace=15.9, lower=20.4, warmup_execute_and_sync=21.2, allocation_and_sync=0.0, aot_compile=7.1, other=2.9 | 10/174 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_b | 1 | compile_or_warm_up_model | 242.2 | executable_load=0.0, cache_lookup_and_decompress=3.6, cache_load=0.0, backend_compile_and_load=172.2, trace=15.7, lower=20.4, warmup_execute_and_sync=20.4, allocation_and_sync=0.0, aot_compile=7.1, other=2.8 | 8/176 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_c | 0 | compile_or_warm_up_model | 242.2 | executable_load=0.0, cache_lookup_and_decompress=3.7, cache_load=0.0, backend_compile_and_load=171.6, trace=15.9, lower=20.4, warmup_execute_and_sync=20.5, allocation_and_sync=0.0, aot_compile=7.3, other=2.8 | 8/176 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_c | 1 | compile_or_warm_up_model | 242.2 | executable_load=0.0, cache_lookup_and_decompress=3.5, cache_load=0.0, backend_compile_and_load=170.7, trace=15.8, lower=20.3, warmup_execute_and_sync=21.8, allocation_and_sync=0.0, aot_compile=7.5, other=2.7 | 7/177 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_d | 0 | compile_or_warm_up_model | 242.2 | executable_load=0.0, cache_lookup_and_decompress=3.7, cache_load=0.0, backend_compile_and_load=172.0, trace=15.7, lower=20.6, warmup_execute_and_sync=20.6, allocation_and_sync=0.0, aot_compile=7.2, other=2.4 | 16/168 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_d | 1 | compile_or_warm_up_model | 242.2 | executable_load=0.0, cache_lookup_and_decompress=3.8, cache_load=0.0, backend_compile_and_load=172.0, trace=15.8, lower=20.5, warmup_execute_and_sync=20.4, allocation_and_sync=0.0, aot_compile=6.9, other=2.8 | 12/172 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |

### `results/overlap/OverlapCompile/controlled-hs-baseline-20261002b/target-provision-deploy.json` (submission raysubmit_N7BWZauphLnU2DY8)

| engine | rank | phase | wall s | exclusive split s | hits/misses (hook) | executables | device exec s | cache | qualification | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| small_a | 0 | compile_or_warm_up_model | 253.3 | executable_load=0.2, cache_lookup_and_decompress=7.2, cache_load=0.0, backend_compile_and_load=152.2, trace=15.9, lower=20.6, warmup_execute_and_sync=20.8, allocation_and_sync=0.0, aot_compile=26.5, other=9.8 | 173/11 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_a | 1 | compile_or_warm_up_model | 252.8 | executable_load=0.1, cache_lookup_and_decompress=6.1, cache_load=0.0, backend_compile_and_load=162.6, trace=15.2, lower=20.6, warmup_execute_and_sync=22.5, allocation_and_sync=0.0, aot_compile=2.6, other=23.1 | 112/72 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_b | 0 | compile_or_warm_up_model | 253.5 | executable_load=0.2, cache_lookup_and_decompress=8.0, cache_load=0.0, backend_compile_and_load=150.4, trace=15.7, lower=20.3, warmup_execute_and_sync=23.1, allocation_and_sync=0.0, aot_compile=26.7, other=9.1 | 171/13 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_b | 1 | compile_or_warm_up_model | 253.1 | executable_load=0.0, cache_lookup_and_decompress=3.4, cache_load=0.0, backend_compile_and_load=170.8, trace=15.9, lower=20.5, warmup_execute_and_sync=21.8, allocation_and_sync=0.0, aot_compile=7.4, other=13.3 | 12/172 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_c | 0 | compile_or_warm_up_model | 253.5 | executable_load=0.2, cache_lookup_and_decompress=7.2, cache_load=0.0, backend_compile_and_load=151.8, trace=15.9, lower=20.5, warmup_execute_and_sync=21.8, allocation_and_sync=0.0, aot_compile=26.9, other=9.1 | 169/15 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_c | 1 | compile_or_warm_up_model | 253.1 | executable_load=0.0, cache_lookup_and_decompress=3.7, cache_load=0.0, backend_compile_and_load=170.5, trace=15.8, lower=20.4, warmup_execute_and_sync=22.2, allocation_and_sync=0.0, aot_compile=6.5, other=13.9 | 19/165 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_d | 0 | compile_or_warm_up_model | 253.7 | executable_load=0.2, cache_lookup_and_decompress=7.2, cache_load=0.0, backend_compile_and_load=152.4, trace=15.8, lower=20.6, warmup_execute_and_sync=21.1, allocation_and_sync=0.0, aot_compile=26.5, other=9.7 | 173/11 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_d | 1 | compile_or_warm_up_model | 253.3 | executable_load=0.1, cache_lookup_and_decompress=5.9, cache_load=0.0, backend_compile_and_load=163.2, trace=15.4, lower=20.6, warmup_execute_and_sync=22.6, allocation_and_sync=0.0, aot_compile=2.6, other=22.8 | 101/83 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |

### `results/overlap/OverlapCompile/controlled-hs-baseline-20261002b/target-warm-deploy.json` (submission raysubmit_g7rEFEzbdHfikkMR)

| engine | rank | phase | wall s | exclusive split s | hits/misses (hook) | executables | device exec s | cache | qualification | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| small_a | 0 | compile_or_warm_up_model | 48.1 | executable_load=2.9, cache_lookup_and_decompress=3.0, cache_load=0.0, backend_compile_and_load=0.0, trace=16.1, lower=20.7, warmup_execute_and_sync=2.7, allocation_and_sync=0.0, aot_compile=0.1, other=2.5 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_a | 1 | compile_or_warm_up_model | 48.1 | executable_load=2.9, cache_lookup_and_decompress=5.8, cache_load=0.0, backend_compile_and_load=0.0, trace=15.6, lower=20.1, warmup_execute_and_sync=2.6, allocation_and_sync=0.0, aot_compile=0.1, other=0.9 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_b | 0 | compile_or_warm_up_model | 50.1 | executable_load=2.8, cache_lookup_and_decompress=3.2, cache_load=0.0, backend_compile_and_load=0.0, trace=15.8, lower=20.2, warmup_execute_and_sync=2.4, allocation_and_sync=0.0, aot_compile=0.1, other=5.7 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_b | 1 | compile_or_warm_up_model | 50.1 | executable_load=2.8, cache_lookup_and_decompress=7.3, cache_load=0.0, backend_compile_and_load=0.0, trace=15.8, lower=20.7, warmup_execute_and_sync=2.3, allocation_and_sync=0.0, aot_compile=0.1, other=0.9 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_c | 0 | compile_or_warm_up_model | 49.5 | executable_load=2.8, cache_lookup_and_decompress=3.1, cache_load=0.0, backend_compile_and_load=0.0, trace=15.9, lower=20.4, warmup_execute_and_sync=2.2, allocation_and_sync=0.0, aot_compile=0.1, other=5.0 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_c | 1 | compile_or_warm_up_model | 49.5 | executable_load=3.0, cache_lookup_and_decompress=6.9, cache_load=0.0, backend_compile_and_load=0.0, trace=15.5, lower=20.1, warmup_execute_and_sync=2.9, allocation_and_sync=0.0, aot_compile=0.1, other=0.9 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_d | 0 | compile_or_warm_up_model | 47.7 | executable_load=2.9, cache_lookup_and_decompress=3.3, cache_load=0.0, backend_compile_and_load=0.0, trace=15.8, lower=20.4, warmup_execute_and_sync=2.3, allocation_and_sync=0.0, aot_compile=0.1, other=2.9 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_d | 1 | compile_or_warm_up_model | 47.7 | executable_load=2.8, cache_lookup_and_decompress=5.1, cache_load=0.0, backend_compile_and_load=0.0, trace=15.8, lower=20.6, warmup_execute_and_sync=2.4, allocation_and_sync=0.0, aot_compile=0.1, other=0.9 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |

### `results/overlap/OverlapCompile/controlled-hs-overlap-20261002b/source-provision-deploy.json` (submission raysubmit_r8iFECm6WCzWMAKN)

| engine | rank | phase | wall s | exclusive split s | hits/misses (hook) | executables | device exec s | cache | qualification | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| large_a | 0 | initialize_from_config | 3.9 | executable_load=0.2, cache_lookup_and_decompress=1.7, cache_load=0.0, backend_compile_and_load=0.0, trace=0.4, lower=1.0, warmup_execute_and_sync=0.4, allocation_and_sync=0.1, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_a | 1 | initialize_from_config | 3.9 | executable_load=0.2, cache_lookup_and_decompress=1.6, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=0.9, warmup_execute_and_sync=0.6, allocation_and_sync=0.1, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_a | 2 | initialize_from_config | 3.9 | executable_load=0.2, cache_lookup_and_decompress=0.9, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=0.9, warmup_execute_and_sync=1.2, allocation_and_sync=0.1, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_a | 3 | initialize_from_config | 3.9 | executable_load=0.2, cache_lookup_and_decompress=1.8, cache_load=0.0, backend_compile_and_load=0.0, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.1, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 0 | initialize_from_config | 3.8 | executable_load=0.2, cache_lookup_and_decompress=1.7, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=0.9, warmup_execute_and_sync=0.4, allocation_and_sync=0.1, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 1 | initialize_from_config | 3.8 | executable_load=0.2, cache_lookup_and_decompress=1.7, cache_load=0.0, backend_compile_and_load=0.0, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.1, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 2 | initialize_from_config | 3.8 | executable_load=0.2, cache_lookup_and_decompress=1.7, cache_load=0.0, backend_compile_and_load=0.0, trace=0.4, lower=1.0, warmup_execute_and_sync=0.4, allocation_and_sync=0.1, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 3 | initialize_from_config | 3.8 | executable_load=0.2, cache_lookup_and_decompress=1.1, cache_load=0.0, backend_compile_and_load=0.0, trace=0.3, lower=0.9, warmup_execute_and_sync=1.0, allocation_and_sync=0.1, aot_compile=0.1, other=0.1 | 77/0 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |

### `results/overlap/OverlapCompile/controlled-hs-overlap-20261002b/target-provision-deploy.json` (submission raysubmit_x3cziwZAskShgnii)

| engine | rank | phase | wall s | exclusive split s | hits/misses (hook) | executables | device exec s | cache | qualification | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| small_a | 0 | initialize_from_config | 31.7 | executable_load=0.1, cache_lookup_and_decompress=2.7, cache_load=0.0, backend_compile_and_load=9.2, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.4, aot_compile=9.8, other=7.8 | 43/34 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_a | 1 | initialize_from_config | 31.7 | executable_load=0.0, cache_lookup_and_decompress=1.7, cache_load=0.0, backend_compile_and_load=23.1, trace=0.4, lower=1.0, warmup_execute_and_sync=0.6, allocation_and_sync=0.4, aot_compile=3.3, other=1.1 | 11/66 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_b | 0 | initialize_from_config | 31.4 | executable_load=0.1, cache_lookup_and_decompress=2.8, cache_load=0.0, backend_compile_and_load=9.2, trace=0.3, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.5, aot_compile=9.7, other=7.5 | 46/31 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_b | 1 | initialize_from_config | 31.4 | executable_load=0.0, cache_lookup_and_decompress=1.7, cache_load=0.0, backend_compile_and_load=23.0, trace=0.4, lower=1.0, warmup_execute_and_sync=0.6, allocation_and_sync=0.4, aot_compile=3.3, other=1.1 | 13/64 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_c | 0 | initialize_from_config | 32.0 | executable_load=0.1, cache_lookup_and_decompress=2.6, cache_load=0.0, backend_compile_and_load=9.4, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.4, aot_compile=9.9, other=7.9 | 39/38 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_c | 1 | initialize_from_config | 32.0 | executable_load=0.0, cache_lookup_and_decompress=1.7, cache_load=0.0, backend_compile_and_load=23.0, trace=0.4, lower=1.0, warmup_execute_and_sync=1.0, allocation_and_sync=0.4, aot_compile=3.3, other=1.3 | 11/66 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_d | 0 | initialize_from_config | 32.4 | executable_load=0.1, cache_lookup_and_decompress=2.7, cache_load=0.0, backend_compile_and_load=8.9, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.4, aot_compile=10.0, other=8.5 | 37/40 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_d | 1 | initialize_from_config | 32.4 | executable_load=0.0, cache_lookup_and_decompress=1.5, cache_load=0.0, backend_compile_and_load=23.5, trace=0.4, lower=1.0, warmup_execute_and_sync=0.6, allocation_and_sync=0.4, aot_compile=3.3, other=1.7 | 2/75 | 43 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |

### `results/overlap/OverlapCompile/controlled-hs-overlap-20261002b/source-provision-deploy.json` (submission raysubmit_r8iFECm6WCzWMAKN)

| engine | rank | phase | wall s | exclusive split s | hits/misses (hook) | executables | device exec s | cache | qualification | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| large_a | 0 | compile_or_warm_up_model | 50.2 | executable_load=2.6, cache_lookup_and_decompress=3.9, cache_load=0.0, backend_compile_and_load=0.0, trace=16.5, lower=21.6, warmup_execute_and_sync=4.6, allocation_and_sync=0.0, aot_compile=0.1, other=0.9 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_a | 1 | compile_or_warm_up_model | 50.2 | executable_load=2.5, cache_lookup_and_decompress=4.7, cache_load=0.0, backend_compile_and_load=0.0, trace=16.3, lower=21.2, warmup_execute_and_sync=4.4, allocation_and_sync=0.0, aot_compile=0.1, other=0.9 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_a | 2 | compile_or_warm_up_model | 50.2 | executable_load=2.5, cache_lookup_and_decompress=1.0, cache_load=0.0, backend_compile_and_load=0.0, trace=16.4, lower=21.0, warmup_execute_and_sync=6.0, allocation_and_sync=0.0, aot_compile=0.1, other=3.1 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_a | 3 | compile_or_warm_up_model | 50.2 | executable_load=2.5, cache_lookup_and_decompress=1.5, cache_load=0.0, backend_compile_and_load=0.0, trace=15.3, lower=22.0, warmup_execute_and_sync=6.0, allocation_and_sync=0.0, aot_compile=0.1, other=2.7 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 0 | compile_or_warm_up_model | 49.8 | executable_load=2.6, cache_lookup_and_decompress=4.6, cache_load=0.0, backend_compile_and_load=0.0, trace=16.4, lower=21.0, warmup_execute_and_sync=4.1, allocation_and_sync=0.0, aot_compile=0.1, other=0.9 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 1 | compile_or_warm_up_model | 49.8 | executable_load=2.6, cache_lookup_and_decompress=1.2, cache_load=0.0, backend_compile_and_load=0.0, trace=15.2, lower=22.1, warmup_execute_and_sync=5.9, allocation_and_sync=0.0, aot_compile=0.1, other=2.7 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 2 | compile_or_warm_up_model | 49.8 | executable_load=2.5, cache_lookup_and_decompress=1.0, cache_load=0.0, backend_compile_and_load=0.0, trace=16.3, lower=21.4, warmup_execute_and_sync=5.5, allocation_and_sync=0.0, aot_compile=0.1, other=2.8 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| large_b | 3 | compile_or_warm_up_model | 49.7 | executable_load=2.5, cache_lookup_and_decompress=1.0, cache_load=0.0, backend_compile_and_load=0.0, trace=16.5, lower=21.2, warmup_execute_and_sync=5.5, allocation_and_sync=0.0, aot_compile=0.1, other=2.9 | 184/0 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |

### `results/overlap/OverlapCompile/controlled-hs-overlap-20261002b/target-provision-deploy.json` (submission raysubmit_x3cziwZAskShgnii)

| engine | rank | phase | wall s | exclusive split s | hits/misses (hook) | executables | device exec s | cache | qualification | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| small_a | 0 | compile_or_warm_up_model | 255.8 | executable_load=0.2, cache_lookup_and_decompress=7.8, cache_load=0.0, backend_compile_and_load=152.2, trace=15.9, lower=20.4, warmup_execute_and_sync=23.1, allocation_and_sync=0.0, aot_compile=27.3, other=9.1 | 172/12 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_a | 1 | compile_or_warm_up_model | 255.4 | executable_load=0.1, cache_lookup_and_decompress=5.5, cache_load=0.0, backend_compile_and_load=164.9, trace=15.9, lower=20.4, warmup_execute_and_sync=23.1, allocation_and_sync=0.0, aot_compile=3.7, other=21.8 | 77/107 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_b | 0 | compile_or_warm_up_model | 255.8 | executable_load=0.2, cache_lookup_and_decompress=8.0, cache_load=0.0, backend_compile_and_load=150.8, trace=15.8, lower=20.4, warmup_execute_and_sync=23.7, allocation_and_sync=0.0, aot_compile=27.8, other=9.0 | 171/13 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_b | 1 | compile_or_warm_up_model | 255.4 | executable_load=0.0, cache_lookup_and_decompress=4.2, cache_load=0.0, backend_compile_and_load=171.9, trace=15.8, lower=20.5, warmup_execute_and_sync=21.4, allocation_and_sync=0.0, aot_compile=7.2, other=14.4 | 10/174 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_c | 0 | compile_or_warm_up_model | 255.3 | executable_load=0.2, cache_lookup_and_decompress=8.0, cache_load=0.0, backend_compile_and_load=152.4, trace=15.7, lower=20.4, warmup_execute_and_sync=21.8, allocation_and_sync=0.0, aot_compile=27.7, other=9.1 | 170/14 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_c | 1 | compile_or_warm_up_model | 254.9 | executable_load=0.0, cache_lookup_and_decompress=4.1, cache_load=0.0, backend_compile_and_load=169.9, trace=15.7, lower=20.4, warmup_execute_and_sync=23.2, allocation_and_sync=0.0, aot_compile=7.3, other=14.4 | 26/158 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_d | 0 | compile_or_warm_up_model | 255.9 | executable_load=0.2, cache_lookup_and_decompress=7.5, cache_load=0.0, backend_compile_and_load=153.7, trace=15.8, lower=20.5, warmup_execute_and_sync=21.6, allocation_and_sync=0.0, aot_compile=27.5, other=9.1 | 172/12 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |
| small_d | 1 | compile_or_warm_up_model | 255.6 | executable_load=0.1, cache_lookup_and_decompress=5.6, cache_load=0.0, backend_compile_and_load=165.7, trace=16.0, lower=20.5, warmup_execute_and_sync=22.3, allocation_and_sync=0.0, aot_compile=4.0, other=21.3 | 84/100 | 207 | — | observed keyed reads; cold/warm labels alone are not cache proof | controlled | accepted |

### `results/overlap/OverlapCompile/controlled-s-20261002a/normal-deploy.json` (submission raysubmit_inzQsy3t1FgzGenM)

Exclusions: small_a: missing phase annotation or TPU lanes; small_b: missing phase annotation or TPU lanes; small_c: missing phase annotation or TPU lanes; small_d: missing phase annotation or TPU lanes

| engine | rank | phase | wall s | exclusive split s | hits/misses (hook) | executables | device exec s | cache | qualification | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| small_a | 0 | initialize_from_config | 32.7 | executable_load=0.1, cache_lookup_and_decompress=2.6, cache_load=0.0, backend_compile_and_load=9.2, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.5, aot_compile=10.0, other=8.6 | 37/40 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| small_a | 1 | initialize_from_config | 32.7 | executable_load=0.0, cache_lookup_and_decompress=1.4, cache_load=0.0, backend_compile_and_load=23.4, trace=0.4, lower=1.0, warmup_execute_and_sync=0.7, allocation_and_sync=0.5, aot_compile=3.5, other=1.8 | 2/75 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| small_b | 0 | initialize_from_config | 32.3 | executable_load=0.1, cache_lookup_and_decompress=2.8, cache_load=0.0, backend_compile_and_load=9.3, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.5, aot_compile=10.0, other=7.9 | 44/33 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| small_b | 1 | initialize_from_config | 32.3 | executable_load=0.0, cache_lookup_and_decompress=1.8, cache_load=0.0, backend_compile_and_load=23.3, trace=0.4, lower=1.0, warmup_execute_and_sync=0.8, allocation_and_sync=0.5, aot_compile=3.4, other=1.3 | 12/65 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| small_c | 0 | initialize_from_config | 30.8 | executable_load=0.1, cache_lookup_and_decompress=3.0, cache_load=0.0, backend_compile_and_load=8.1, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.4, aot_compile=10.1, other=7.3 | 57/20 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| small_c | 1 | initialize_from_config | 30.8 | executable_load=0.0, cache_lookup_and_decompress=1.6, cache_load=0.0, backend_compile_and_load=22.4, trace=0.4, lower=1.0, warmup_execute_and_sync=0.5, allocation_and_sync=0.4, aot_compile=3.3, other=1.3 | 12/65 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| small_d | 0 | initialize_from_config | 31.4 | executable_load=0.1, cache_lookup_and_decompress=2.7, cache_load=0.0, backend_compile_and_load=8.5, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.5, aot_compile=10.3, other=7.5 | 50/27 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| small_d | 1 | initialize_from_config | 31.4 | executable_load=0.0, cache_lookup_and_decompress=2.2, cache_load=0.0, backend_compile_and_load=23.0, trace=0.4, lower=1.0, warmup_execute_and_sync=0.4, allocation_and_sync=0.4, aot_compile=3.3, other=0.8 | 20/57 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |

### `results/overlap/OverlapCompile/controlled-s-20261002a/normal-deploy.json` (submission raysubmit_inzQsy3t1FgzGenM)

Exclusions: small_a: missing phase annotation or TPU lanes; small_b: missing phase annotation or TPU lanes; small_c: missing phase annotation or TPU lanes; small_d: missing phase annotation or TPU lanes

| engine | rank | phase | wall s | exclusive split s | hits/misses (hook) | executables | device exec s | cache | qualification | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| small_a | 0 | compile_or_warm_up_model | 256.2 | executable_load=0.2, cache_lookup_and_decompress=7.5, cache_load=0.0, backend_compile_and_load=151.9, trace=16.5, lower=21.2, warmup_execute_and_sync=21.1, allocation_and_sync=0.0, aot_compile=28.2, other=9.5 | 174/10 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| small_a | 1 | compile_or_warm_up_model | 255.8 | executable_load=0.1, cache_lookup_and_decompress=5.8, cache_load=0.0, backend_compile_and_load=162.2, trace=15.5, lower=21.0, warmup_execute_and_sync=24.2, allocation_and_sync=0.0, aot_compile=2.8, other=24.2 | 109/75 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| small_b | 0 | compile_or_warm_up_model | 256.3 | executable_load=0.2, cache_lookup_and_decompress=7.8, cache_load=0.0, backend_compile_and_load=152.0, trace=16.1, lower=20.9, warmup_execute_and_sync=21.3, allocation_and_sync=0.0, aot_compile=28.3, other=9.7 | 171/13 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| small_b | 1 | compile_or_warm_up_model | 255.9 | executable_load=0.0, cache_lookup_and_decompress=4.2, cache_load=0.0, backend_compile_and_load=170.3, trace=15.7, lower=21.9, warmup_execute_and_sync=21.9, allocation_and_sync=0.0, aot_compile=7.5, other=14.4 | 30/154 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| small_c | 0 | compile_or_warm_up_model | 256.7 | executable_load=0.2, cache_lookup_and_decompress=7.7, cache_load=0.0, backend_compile_and_load=152.2, trace=15.8, lower=20.6, warmup_execute_and_sync=21.6, allocation_and_sync=0.0, aot_compile=28.2, other=10.2 | 172/12 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| small_c | 1 | compile_or_warm_up_model | 256.3 | executable_load=0.1, cache_lookup_and_decompress=4.9, cache_load=0.0, backend_compile_and_load=168.0, trace=15.9, lower=21.6, warmup_execute_and_sync=22.2, allocation_and_sync=0.0, aot_compile=5.2, other=18.5 | 53/131 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| small_d | 0 | compile_or_warm_up_model | 256.6 | executable_load=0.2, cache_lookup_and_decompress=7.8, cache_load=0.0, backend_compile_and_load=152.3, trace=16.2, lower=20.7, warmup_execute_and_sync=21.9, allocation_and_sync=0.0, aot_compile=28.0, other=9.4 | 171/13 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| small_d | 1 | compile_or_warm_up_model | 256.2 | executable_load=0.0, cache_lookup_and_decompress=3.8, cache_load=0.0, backend_compile_and_load=171.8, trace=16.3, lower=20.8, warmup_execute_and_sync=21.9, allocation_and_sync=0.0, aot_compile=7.4, other=14.2 | 14/170 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |

### `results/overlap/OverlapCompile/controlled-h-20261002a/normal-deploy.json` (submission raysubmit_JGDg4hHPcUpekJaN)

Exclusions: large_a: missing phase annotation or TPU lanes; large_b: missing phase annotation or TPU lanes

| engine | rank | phase | wall s | exclusive split s | hits/misses (hook) | executables | device exec s | cache | qualification | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| large_a | 0 | initialize_from_config | 35.6 | executable_load=0.1, cache_lookup_and_decompress=2.4, cache_load=0.0, backend_compile_and_load=11.9, trace=0.4, lower=1.0, warmup_execute_and_sync=0.4, allocation_and_sync=0.4, aot_compile=10.4, other=8.6 | 31/46 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| large_a | 1 | initialize_from_config | 35.6 | executable_load=0.0, cache_lookup_and_decompress=1.4, cache_load=0.0, backend_compile_and_load=26.8, trace=0.4, lower=1.0, warmup_execute_and_sync=0.4, allocation_and_sync=0.4, aot_compile=3.4, other=1.8 | 0/77 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| large_a | 2 | initialize_from_config | 35.6 | executable_load=0.0, cache_lookup_and_decompress=1.5, cache_load=0.0, backend_compile_and_load=26.7, trace=0.4, lower=1.0, warmup_execute_and_sync=0.5, allocation_and_sync=0.4, aot_compile=3.3, other=1.7 | 1/76 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| large_a | 3 | initialize_from_config | 35.6 | executable_load=0.0, cache_lookup_and_decompress=1.4, cache_load=0.0, backend_compile_and_load=26.6, trace=0.4, lower=1.0, warmup_execute_and_sync=0.4, allocation_and_sync=0.4, aot_compile=3.5, other=1.8 | 0/77 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| large_b | 0 | initialize_from_config | 34.5 | executable_load=0.1, cache_lookup_and_decompress=3.1, cache_load=0.0, backend_compile_and_load=11.6, trace=0.4, lower=1.0, warmup_execute_and_sync=0.3, allocation_and_sync=0.5, aot_compile=9.9, other=7.6 | 52/25 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| large_b | 1 | initialize_from_config | 34.5 | executable_load=0.0, cache_lookup_and_decompress=1.8, cache_load=0.0, backend_compile_and_load=25.4, trace=0.4, lower=1.0, warmup_execute_and_sync=1.0, allocation_and_sync=0.5, aot_compile=3.2, other=1.3 | 15/62 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| large_b | 2 | initialize_from_config | 34.5 | executable_load=0.0, cache_lookup_and_decompress=1.9, cache_load=0.0, backend_compile_and_load=25.3, trace=0.4, lower=1.0, warmup_execute_and_sync=1.0, allocation_and_sync=0.4, aot_compile=3.2, other=1.3 | 15/62 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| large_b | 3 | initialize_from_config | 34.5 | executable_load=0.0, cache_lookup_and_decompress=2.9, cache_load=0.0, backend_compile_and_load=25.0, trace=0.4, lower=1.0, warmup_execute_and_sync=0.7, allocation_and_sync=0.5, aot_compile=3.0, other=1.1 | 19/58 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |

### `results/overlap/OverlapCompile/controlled-h-20261002a/normal-deploy.json` (submission raysubmit_JGDg4hHPcUpekJaN)

Exclusions: large_a: missing phase annotation or TPU lanes; large_b: missing phase annotation or TPU lanes

| engine | rank | phase | wall s | exclusive split s | hits/misses (hook) | executables | device exec s | cache | qualification | gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| large_a | 0 | compile_or_warm_up_model | 250.4 | executable_load=0.3, cache_lookup_and_decompress=8.5, cache_load=0.0, backend_compile_and_load=150.4, trace=15.9, lower=22.5, warmup_execute_and_sync=15.1, allocation_and_sync=0.0, aot_compile=28.5, other=9.4 | 173/11 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| large_a | 1 | compile_or_warm_up_model | 250.0 | executable_load=0.0, cache_lookup_and_decompress=3.4, cache_load=0.0, backend_compile_and_load=169.6, trace=16.9, lower=21.8, warmup_execute_and_sync=17.0, allocation_and_sync=0.0, aot_compile=7.8, other=13.6 | 2/182 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| large_a | 2 | compile_or_warm_up_model | 250.0 | executable_load=0.0, cache_lookup_and_decompress=3.5, cache_load=0.0, backend_compile_and_load=170.5, trace=17.0, lower=21.9, warmup_execute_and_sync=15.6, allocation_and_sync=0.0, aot_compile=7.9, other=13.5 | 2/182 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| large_a | 3 | compile_or_warm_up_model | 249.9 | executable_load=0.0, cache_lookup_and_decompress=3.5, cache_load=0.0, backend_compile_and_load=170.2, trace=16.9, lower=21.7, warmup_execute_and_sync=16.5, allocation_and_sync=0.0, aot_compile=7.8, other=13.4 | 5/179 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| large_b | 0 | compile_or_warm_up_model | 250.0 | executable_load=0.2, cache_lookup_and_decompress=7.7, cache_load=0.0, backend_compile_and_load=150.7, trace=16.9, lower=21.4, warmup_execute_and_sync=15.8, allocation_and_sync=0.0, aot_compile=28.0, other=9.2 | 173/11 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| large_b | 1 | compile_or_warm_up_model | 249.7 | executable_load=0.1, cache_lookup_and_decompress=5.5, cache_load=0.0, backend_compile_and_load=164.8, trace=17.1, lower=21.8, warmup_execute_and_sync=17.4, allocation_and_sync=0.0, aot_compile=4.4, other=18.5 | 80/104 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| large_b | 2 | compile_or_warm_up_model | 249.6 | executable_load=0.1, cache_lookup_and_decompress=5.5, cache_load=0.0, backend_compile_and_load=163.6, trace=16.8, lower=21.8, warmup_execute_and_sync=18.7, allocation_and_sync=0.0, aot_compile=4.6, other=18.6 | 80/104 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |
| large_b | 3 | compile_or_warm_up_model | 249.6 | executable_load=0.1, cache_lookup_and_decompress=5.5, cache_load=0.0, backend_compile_and_load=166.5, trace=17.0, lower=22.0, warmup_execute_and_sync=15.9, allocation_and_sync=0.0, aot_compile=4.3, other=18.3 | 85/99 | None | — | observed keyed reads; cold/warm labels alone are not cache proof | diagnostic_only | NOT ACCEPTED: evidence_qualified, production_config, effective_cache_matches_requested, rpc_cache_matches_requested, coverage_matched_thresholds, requested_controls, keyed_cache_evidence |

## Switch cost C = first dispatch − last old completion + ramp deficit

| source file | from | to | treatment | dispatch gap s | ramp deficit s | C s | observation verdict | performance eligible | scope/verdict |
|---|---|---|---|---|---|---|---|---|---|
| controlled-hs-baseline-20261002b/cold-transition-observed.json | H | S | cold | 451.409829 | 6.886727 | 458.296556 | accepted | False | OBSERVED ONLY |
| controlled-hs-baseline-20261002b/warm-transition-observed.json | H | S | warm | 218.716232 | 6.544148 | 225.260380 | accepted | False | OBSERVED ONLY |
| overlap_switch.json | H | S | overlap | — | — | — | NOT ACCEPTED: first_dispatch_anchor, ramp_deficit_measured, gap_matches_anchors, cost_is_gap_plus_deficit, target_all_requests_complete, no_run_error | False | NOT ACCEPTED: first_dispatch_anchor, ramp_deficit_measured, gap_matches_anchors, cost_is_gap_plus_deficit, target_all_requests_complete, no_run_error |
| overlap_switch.json | S | H | cold | — | — | — | NOT ACCEPTED: last_completion_anchor, first_dispatch_anchor, ramp_deficit_measured, gap_matches_anchors, cost_is_gap_plus_deficit, source_all_requests_complete, target_all_requests_complete, no_run_error | False | NOT ACCEPTED: last_completion_anchor, first_dispatch_anchor, ramp_deficit_measured, gap_matches_anchors, cost_is_gap_plus_deficit, source_all_requests_complete, target_all_requests_complete, no_run_error |
| overlap_switch.json | S | H | warm | — | — | — | NOT ACCEPTED: last_completion_anchor, first_dispatch_anchor, ramp_deficit_measured, gap_matches_anchors, cost_is_gap_plus_deficit, source_all_requests_complete, target_all_requests_complete, no_run_error | False | NOT ACCEPTED: last_completion_anchor, first_dispatch_anchor, ramp_deficit_measured, gap_matches_anchors, cost_is_gap_plus_deficit, source_all_requests_complete, target_all_requests_complete, no_run_error |
| overlap_switch.json | S | H | overlap | — | — | — | NOT ACCEPTED: last_completion_anchor, first_dispatch_anchor, ramp_deficit_measured, gap_matches_anchors, cost_is_gap_plus_deficit, source_all_requests_complete, target_all_requests_complete, no_run_error | False | NOT ACCEPTED: last_completion_anchor, first_dispatch_anchor, ramp_deficit_measured, gap_matches_anchors, cost_is_gap_plus_deficit, source_all_requests_complete, target_all_requests_complete, no_run_error |

controlled-hs-baseline-20261002b/cold-transition-observed.json: unqualified/unfrozen pilot observation only; comparison exclusions: source_catalogue_matches_manifest, source_qualified_source, source_qualified_control, source_frozen_runtime, target_catalogue_matches_manifest, target_qualified_source, target_qualified_control, target_frozen_runtime, matching_runtime

Prior target provisioning=920.8023388385773 s; C plus provisioning=1379.0988948719628 s. joint prior target graph/bundle, reference generation, weight snapshot and release; not pure compile

No supported S/H device attribution
Not a same-workload historical crossover
Cold startup shares one fresh cache root among replicas; intra-startup shared hits are possible
Warm-control population qualification is separate from successful deploy

controlled-hs-baseline-20261002b/warm-transition-observed.json: unqualified/unfrozen pilot observation only; comparison exclusions: source_catalogue_matches_manifest, source_qualified_source, source_qualified_control, source_frozen_runtime, target_catalogue_matches_manifest, target_qualified_source, target_qualified_control, target_frozen_runtime, matching_runtime

Prior target provisioning=920.8023388385773 s; C plus provisioning=1146.062718814605 s. joint prior target graph/bundle, reference generation, weight snapshot and release; not pure compile

No supported S/H device attribution
Not a same-workload historical crossover
Cold startup shares one fresh cache root among replicas; intra-startup shared hits are possible
Warm-control population qualification is separate from successful deploy

overlap_switch.json: incomplete or rejected transition evidence; no accepted C; comparison exclusions: same_manifest_sha256, same_catalogue_sha256, source_catalogue_matches_manifest, source_qualified_source, source_qualified_control, source_frozen_runtime, target_catalogue_matches_manifest, target_qualified_source, target_qualified_control, target_frozen_runtime, matching_runtime

Loaded backend revision UNFROZEN and loaded_runtime null; runtime equivalence not established.
Actual catalogue differs from historical catalogue; no same-workload historical crossover.
No supported S/H phase-specific TPU device attribution.
FAILED: TPU topology/plugin initialization refused while serving PID570736 owned TPU; no replay result.
No target deploy, target deserialization/listener proof, overlap C or post-W0 target output comparison.
12.0956s W0 attempt delta is incidental, NOT successful-compile interference.

overlap_switch.json: incomplete or rejected transition evidence; no accepted C; comparison exclusions: same_manifest_sha256, same_catalogue_sha256, source_catalogue_matches_manifest, source_qualified_source, source_qualified_control, source_frozen_runtime, target_catalogue_matches_manifest, target_qualified_source, target_qualified_control, target_frozen_runtime, matching_runtime

Loaded backend revision UNFROZEN and loaded_runtime null; runtime equivalence not established.
Actual catalogue differs from historical catalogue; no same-workload historical crossover.
No supported S/H phase-specific TPU device attribution.
NOT_MEASURED: dependent S-to-H arms cancelled after observed live-serving mechanism blocker.

overlap_switch.json: incomplete or rejected transition evidence; no accepted C; comparison exclusions: same_manifest_sha256, same_catalogue_sha256, source_catalogue_matches_manifest, source_qualified_source, source_qualified_control, source_frozen_runtime, target_catalogue_matches_manifest, target_qualified_source, target_qualified_control, target_frozen_runtime, matching_runtime

Loaded backend revision UNFROZEN and loaded_runtime null; runtime equivalence not established.
Actual catalogue differs from historical catalogue; no same-workload historical crossover.
No supported S/H phase-specific TPU device attribution.
NOT_MEASURED: dependent S-to-H arms cancelled after observed live-serving mechanism blocker.

overlap_switch.json: incomplete or rejected transition evidence; no accepted C; comparison exclusions: same_manifest_sha256, same_catalogue_sha256, source_catalogue_matches_manifest, source_qualified_source, source_qualified_control, source_frozen_runtime, target_catalogue_matches_manifest, target_qualified_source, target_qualified_control, target_frozen_runtime, matching_runtime

Loaded backend revision UNFROZEN and loaded_runtime null; runtime equivalence not established.
Actual catalogue differs from historical catalogue; no same-workload historical crossover.
No supported S/H phase-specific TPU device attribution.
NOT_MEASURED: dependent S-to-H arms cancelled after observed live-serving mechanism blocker.

## Direction-matched break-even (reconfig_breakeven.py formula, measured throughput table)

All numeric crossovers are rate-model estimates, not observed crossover experiments. Unknown or mismatched actual catalogue/manifest and missing source/control proof suppress the number. Compatible but unfrozen evidence is labeled ESTIMATE ONLY, never headline evidence.

| workload | from→to | treatment | C s | gain/phase s | N* tokens | × phase | verdict |
|---|---|---|---|---|---|---|---|
| W0 | H→S | cold | 458.3 | — | ineligible | — | NOT ACCEPTED: source_matching_manifest_sha256, source_matching_catalogue_sha256, source_rate_source_qualified, source_rate_control_qualified, target_matching_manifest_sha256, target_matching_catalogue_sha256, target_rate_source_qualified, target_rate_control_qualified |
| W3 | H→S | cold | 458.3 | — | ineligible | — | NOT ACCEPTED: source_matching_manifest_sha256, source_matching_catalogue_sha256, source_rate_source_qualified, source_rate_control_qualified, target_matching_manifest_sha256, target_matching_catalogue_sha256, target_rate_source_qualified, target_rate_control_qualified |
| W5 | H→S | cold | 458.3 | — | ineligible | — | NOT ACCEPTED: source_matching_manifest_sha256, source_matching_catalogue_sha256, source_rate_source_qualified, source_rate_control_qualified, target_matching_manifest_sha256, target_matching_catalogue_sha256, target_rate_source_qualified, target_rate_control_qualified |
| W0 | H→S | warm | 225.3 | — | ineligible | — | NOT ACCEPTED: source_matching_manifest_sha256, source_matching_catalogue_sha256, source_rate_source_qualified, source_rate_control_qualified, target_matching_manifest_sha256, target_matching_catalogue_sha256, target_rate_source_qualified, target_rate_control_qualified |
| W3 | H→S | warm | 225.3 | — | ineligible | — | NOT ACCEPTED: source_matching_manifest_sha256, source_matching_catalogue_sha256, source_rate_source_qualified, source_rate_control_qualified, target_matching_manifest_sha256, target_matching_catalogue_sha256, target_rate_source_qualified, target_rate_control_qualified |
| W5 | H→S | warm | 225.3 | — | ineligible | — | NOT ACCEPTED: source_matching_manifest_sha256, source_matching_catalogue_sha256, source_rate_source_qualified, source_rate_control_qualified, target_matching_manifest_sha256, target_matching_catalogue_sha256, target_rate_source_qualified, target_rate_control_qualified |

