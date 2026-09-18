# Static heterogeneous bulk inference behind llm-d: empirical report

## Research conclusion

Acceptance audit: **acceptance_audit_passed**; 102/102 declared runs eligible. Completion of the experiment does not imply a useful system speedup.

**Small-instance regimes.** Paired S-versus-H makespan reductions at equal 32-chip allocation:

| Workload | Median reduction | Six-run range or available-pair range | Coverage |
|---|---:|---|---|
| E_LONG | +4.50% | [+4.40%, +4.70%] | complete |
| E_SHORT | +29.74% | [+28.14%, +33.95%] | complete |
| W0 | +26.01% | [+24.92%, +26.41%] | complete |
| W1 | -2.00% | [-2.48%, -1.05%] | complete |
| W2 | +40.55% | [+37.72%, +42.29%] | complete |

**Frozen mixed-layout policy.** Positive reduction favors MR; negative means slower than the strongest homogeneous reference.

| Workload | Strongest homogeneous | MR makespan reduction | Range |
|---|---|---:|---|
| W0 | S | -15.42% | [-17.32%, -14.41%] |
| W1 | H | -3.63% | [-4.61%, -2.34%] |
| W2 | S | -61.12% | [-64.74%, -55.05%] |

Enabled-arm observations: {'enabled_runs': 24, 'proposals': 0, 'committed': 0}. Unknown eligibility counts are not interpreted as zero. **The frozen policy made no migration proposals.** These end-to-end contrasts do not identify a migration or KV-transfer benefit: they measure the policy in a regime where neither ran. This limitation also applies to the held-out growing workload; no thresholds were retuned. Actual migrations were exercised separately by qualification and the matched mechanism group. 

**Recompute versus host KV under fixed real background load.** Median first-destination-token clock envelopes include the measured handoff path, not only transfer kernels. They are not end-to-end gateway makespans.

| Checkpoint | Recompute median envelope s | Host-KV median envelope s |
|---|---|---|
| p256-h64 | [0.19543051719665527, 0.248307466506958] | [2.4351563453674316, 2.485182523727417] |
| p1024-h257 | [0.41639018058776855, 0.46721720695495605] | [4.897570371627808, 4.94953179359436] |
| p4096-h257 | [0.7855052947998047, 0.8404600620269775] | [15.314831256866455, 15.367817878723145] |

Host transport is not a guaranteed optimization; the full trial/pair evidence and unmeasured wire-byte limitations appear below.

Recorded active-source free-block fractions never fell below 0.127451; the configured pressure threshold(s) were [0.12]. This is observed scheduler evidence, not an inferred eligibility count or proof of the admission mechanism causing the result. Per-run minima and artifact hashes: `results/hetero/20260918/permanent_fp32_v2_campaign_outcome.json`.

Final engine/group inspections and generation-scoped discovery cleanup: `results/hetero/20260918/permanent_fp32_v2_final_cleanup.json`. Remaining selected pool endpoints: [].


## Status: selected-cohort evidence (acceptance audit below)

102 eligible headline runs; 0 retained diagnostic or unqualified runs. Engineering functionality, a working mechanism, workload-specific speedup and system advantage are distinct claims. The selected-cohort acceptance audit is reported below.

Prospective protocol: `0f5e4c095b0e9ba137f2a07b7ed517a23388a70b5aa794897cab37745eae2159`. A local source snapshot is not a qualified loaded-source freeze.
- **selected source freeze**: qualified
- **declared scheduled runs**: 102
- **selected retained runs**: 102
- **eligible scheduled runs**: 102
- **missing or excluded scheduled runs**: 0
- **retained other-cohort history**: 61 runs, never pooled into selected effects
- **minimum useful makespan reduction**: 0.05
- **qualification binding**: {'backend_revision': '148b4295bf83c0eaa473b11fff5c7c643092f1873200cef1f52b07a2265861ed', 'engine_settings': {'activation_dtype': 'float32', 'backend_revision': '148b4295bf83c0eaa473b11fff5c7c643092f1873200cef1f52b07a2265861ed', 'dtype': 'bfloat16', 'enable_prefix_caching': False, 'enforce_eager': False, 'gpu_memory_utilization': 0.85, 'kv_cache_dtype': 'float32', 'matmul_precision': 'highest', 'max_model_len': 8192, 'max_num_batched_tokens': 2048, 'max_num_seqs': 512, 'model': '/models/gemma-4-31b', 'runtime_revision': '27c198dd51d9df1ffe681288039cbb6f038e62a8b235b08a39b06a21d3041d4f', 'seed': 0}, 'evidence': {'continuation': {'file_sha256': '4d7ed1d051c4f379d8d8a7592417e2a329adc3f90e13e96a99d26af816a71847', 'path': 'results/hetero/20260918/permanent_fp32_v2_continuation.json'}, 'gate_H': {'file_sha256': '7d583e14a05dd934dbd70c3594f0021335d7d7917f1203eb2b491ee30d325b4e', 'path': 'results/hetero/20260918/permanent_fp32_v2_gate_H.json'}, 'gate_M': {'file_sha256': 'd25ee01eb8e045f514e5676b44740d7084f5896132e84ddb260120cfd3aeb9c3', 'path': 'results/hetero/20260918/permanent_fp32_v2_gate_M.json'}, 'gate_S': {'file_sha256': 'e3775780d42d7f92cf91c5c73d237af7d77fdf3bd4f1ef3aa036ca3afd835b5c', 'path': 'results/hetero/20260918/permanent_fp32_v2_gate_S.json'}, 'kv_host': {'file_sha256': 'fdfd70a5299e5acfecc64224fc33cf1557469766f144b440a504fd0c5d30a3d4', 'path': 'results/hetero/20260918/permanent_fp32_v2_kv_host.json'}, 'lifecycle': {'file_sha256': '3637ae6eaf36ec5c15be5fe03a7cd46e759529cae0b6f3768749f15b49af41aa', 'path': 'results/hetero/20260918/permanent_fp32_v2_lifecycle_M.json'}, 'recompute': {'file_sha256': 'b9e125632c36d2e46e5644470ff341847fd8633da215d3f16818fd955548524a', 'path': 'results/hetero/20260918/permanent_fp32_v2_recompute.json'}}, 'runtime_revision': '27c198dd51d9df1ffe681288039cbb6f038e62a8b235b08a39b06a21d3041d4f'}

Missing data is reported as missing, never zero. Diagnostic timings below are not headline evidence. Replay/duplicates do not count as useful output; all allocated chips are charged. Iteration clocks are host observations, not device/kernel timings or independent repetitions.

## Workload E_LONG
| Arm | Rep | Makespan s | Useful tokens | Useful tok/s | Useful tok/s/allocated chip | Eligible | Exclusions |
|---|---:|---:|---:|---:|---:|---|---|
| H | 1 | 25.0534 | 12288 | 490.473 | 15.3273 | True | none |
| H | 2 | 25.0922 | 12288 | 489.715 | 15.3036 | True | none |
| H | 3 | 25.0747 | 12288 | 490.056 | 15.3142 | True | none |
| H | 4 | 25.0692 | 12288 | 490.163 | 15.3176 | True | none |
| H | 5 | 25.0489 | 12288 | 490.56 | 15.33 | True | none |
| H | 6 | 25.0608 | 12288 | 490.328 | 15.3228 | True | none |
| S | 1 | 23.9388 | 12288 | 513.308 | 16.0409 | True | none |
| S | 2 | 23.9129 | 12288 | 513.866 | 16.0583 | True | none |
| S | 3 | 23.9724 | 12288 | 512.589 | 16.0184 | True | none |
| S | 4 | 23.9195 | 12288 | 513.723 | 16.0538 | True | none |
| S | 5 | 23.916 | 12288 | 513.799 | 16.0562 | True | none |
| S | 6 | 23.9373 | 12288 | 513.341 | 16.0419 | True | none |

### Per-run accounting and iteration eligibility
| Arm/rep | Capture rows / eligible / control / invalid | Retirement/artifact status | Deep-backlog tok/s | Terminal drain s / fraction | Initialization s / init+serving s | Migrations committed / declined / eligible / fallback | Actual payload / network bytes |
|---|---|---|---|---|---|---|---|
| H/1 | 1078 / 1074 / 4 / 0 | available | missing | 25.05335831642151 / 1.0 | 366.09 / 391.1434 | 0 / 0 / missing / missing | missing / missing |
| H/2 | 1080 / 1076 / 4 / 0 | available | missing | 25.092153072357178 / 1.0 | 363.37 / 388.4622 | 0 / 0 / missing / missing | missing / missing |
| H/3 | 1077 / 1073 / 4 / 0 | available | missing | 25.074700593948364 / 1.0 | 364.6 / 389.67470000000003 | 0 / 0 / missing / missing | missing / missing |
| H/4 | 1078 / 1074 / 4 / 0 | available | missing | 25.069238662719727 / 1.0 | 361.53 / 386.5992 | 0 / 0 / missing / missing | missing / missing |
| H/5 | 1078 / 1074 / 4 / 0 | available | missing | 25.04893684387207 / 1.0 | 363.45 / 388.4989 | 0 / 0 / missing / missing | missing / missing |
| H/6 | 1077 / 1073 / 4 / 0 | available | missing | 25.060771226882935 / 1.0 | 365.18 / 390.24080000000004 | 0 / 0 / missing / missing | missing / missing |
| S/1 | 2160 / 2144 / 16 / 0 | available | missing | 23.93882942199707 / 1.0 | 364.03 / 387.9688 | 0 / 0 / missing / missing | missing / missing |
| S/2 | 2159 / 2144 / 15 / 0 | available | missing | 23.912865161895752 / 1.0 | 363.55 / 387.4629 | 0 / 0 / missing / missing | missing / missing |
| S/3 | 2159 / 2144 / 15 / 0 | available | missing | 23.972408533096313 / 1.0 | 363.64 / 387.6124 | 0 / 0 / missing / missing | missing / missing |
| S/4 | 2158 / 2144 / 14 / 0 | available | missing | 23.919528484344482 / 1.0 | 361.52 / 385.43949999999995 | 0 / 0 / missing / missing | missing / missing |
| S/5 | 2159 / 2144 / 15 / 0 | available | missing | 23.915979385375977 / 1.0 | 363.24 / 387.156 | 0 / 0 / missing / missing | missing / missing |
| S/6 | 2160 / 2144 / 16 / 0 | available | missing | 23.937322854995728 / 1.0 | 364.06 / 387.9973 | 0 / 0 / missing / missing | missing / missing |

- H/1 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_LONG_H_r1.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_LONG_H_r1/E_LONG_H_r1_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/2 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_LONG_H_r2.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_LONG_H_r2/E_LONG_H_r2_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/3 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_LONG_H_r3.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_LONG_H_r3/E_LONG_H_r3_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/4 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_LONG_H_r4.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_LONG_H_r4/E_LONG_H_r4_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/5 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_LONG_H_r5.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_LONG_H_r5/E_LONG_H_r5_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/6 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_LONG_H_r6.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_LONG_H_r6/E_LONG_H_r6_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/1 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_LONG_S_r1.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_LONG_S_r1/E_LONG_S_r1_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/2 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_LONG_S_r2.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_LONG_S_r2/E_LONG_S_r2_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/3 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_LONG_S_r3.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_LONG_S_r3/E_LONG_S_r3_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/4 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_LONG_S_r4.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_LONG_S_r4/E_LONG_S_r4_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/5 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_LONG_S_r5.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_LONG_S_r5/E_LONG_S_r5_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/6 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_LONG_S_r6.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_LONG_S_r6/E_LONG_S_r6_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.

Payload bytes and network bytes are distinct. No formula-priced transfer or inferred network volume is accepted. Per-instance/phase capture aggregates, raw CSV paths, gateway counters and legacy byte fields are retained in RESULTS.json. Unknown coverage is not zero coverage. Initialization is not borrowed from an unrelated deployment or summed across overlapping worker initializations.

### Paired contrasts
- **H1_instance_mix**: not_measured_or_not_qualified
- **H2_migration**: not_measured_or_not_qualified
- **S_vs_H**: measured_but_below_threshold
  - Makespan reduction median +4.50%; range [0.044, 0.047]; 6 whole-run pairs.
  - Sustained progress: missing_common_interval_no_sustained_claim; paired native throughput and terminal drain: [].
  - Rep 1: baseline 25.05 s; treatment 23.94 s; makespan reduction +4.45%; throughput increase +4.66%.
  - Rep 2: baseline 25.09 s; treatment 23.91 s; makespan reduction +4.70%; throughput increase +4.93%.
  - Rep 3: baseline 25.07 s; treatment 23.97 s; makespan reduction +4.40%; throughput increase +4.60%.
  - Rep 4: baseline 25.07 s; treatment 23.92 s; makespan reduction +4.59%; throughput increase +4.81%.
  - Rep 5: baseline 25.05 s; treatment 23.92 s; makespan reduction +4.52%; throughput increase +4.74%.
  - Rep 6: baseline 25.06 s; treatment 23.94 s; makespan reduction +4.48%; throughput increase +4.69%.
- **M0_vs_H**: not_measured_or_not_qualified
- **MR_vs_H**: not_measured_or_not_qualified
- **M0_vs_S**: not_measured_or_not_qualified
- **MR_vs_S**: not_measured_or_not_qualified
- **H3_transport**: not_measured_or_not_qualified
- **H4_system**: not_measured_or_not_qualified

## Workload E_SHORT
| Arm | Rep | Makespan s | Useful tokens | Useful tok/s | Useful tok/s/allocated chip | Eligible | Exclusions |
|---|---:|---:|---:|---:|---:|---|---|
| H | 1 | 51.4744 | 65536 | 1273.176 | 39.7867 | True | none |
| H | 2 | 51.069 | 65536 | 1283.284 | 40.1026 | True | none |
| H | 3 | 51.09 | 65536 | 1282.756 | 40.0861 | True | none |
| H | 4 | 51.445 | 65536 | 1273.905 | 39.8095 | True | none |
| H | 5 | 51.4371 | 65536 | 1274.1 | 39.8156 | True | none |
| H | 6 | 51.4957 | 65536 | 1272.65 | 39.7703 | True | none |
| S | 1 | 36.6614 | 65536 | 1787.602 | 55.8626 | True | none |
| S | 2 | 36.1882 | 65536 | 1810.978 | 56.5931 | True | none |
| S | 3 | 33.7427 | 65536 | 1942.23 | 60.6947 | True | none |
| S | 4 | 35.7598 | 65536 | 1832.67 | 57.2709 | True | none |
| S | 5 | 35.8378 | 65536 | 1828.683 | 57.1463 | True | none |
| S | 6 | 37.0055 | 65536 | 1770.98 | 55.3431 | True | none |

### Per-run accounting and iteration eligibility
| Arm/rep | Capture rows / eligible / control / invalid | Retirement/artifact status | Deep-backlog tok/s | Terminal drain s / fraction | Initialization s / init+serving s | Migrations committed / declined / eligible / fallback | Actual payload / network bytes |
|---|---|---|---|---|---|---|---|
| H/1 | 1158 / 1154 / 4 / 0 | available | missing | 51.47444415092468 / 1.0 | 366.09 / 417.5644 | 0 / 0 / missing / missing | missing / missing |
| H/2 | 1158 / 1154 / 4 / 0 | available | missing | 51.06898260116577 / 1.0 | 363.37 / 414.439 | 0 / 0 / missing / missing | missing / missing |
| H/3 | 1159 / 1155 / 4 / 0 | available | missing | 51.09001302719116 / 1.0 | 364.6 / 415.69000000000005 | 0 / 0 / missing / missing | missing / missing |
| H/4 | 1194 / 1190 / 4 / 0 | available | missing | 51.44495868682861 / 1.0 | 361.53 / 412.97499999999997 | 0 / 0 / missing / missing | missing / missing |
| H/5 | 1184 / 1180 / 4 / 0 | available | missing | 51.43709206581116 / 1.0 | 363.45 / 414.8871 | 0 / 0 / missing / missing | missing / missing |
| H/6 | 1158 / 1154 / 4 / 0 | available | missing | 51.495713233947754 / 1.0 | 365.18 / 416.6757 | 0 / 0 / missing / missing | missing / missing |
| S/1 | 2488 / 2480 / 8 / 0 | available | missing | 36.66140699386597 / 1.0 | 364.03 / 400.6914 | 0 / 0 / missing / missing | missing / missing |
| S/2 | 2473 / 2465 / 8 / 0 | available | missing | 36.18818235397339 / 1.0 | 363.55 / 399.7382 | 0 / 0 / missing / missing | missing / missing |
| S/3 | 2323 / 2315 / 8 / 0 | available | missing | 33.742666721343994 / 1.0 | 363.64 / 397.3827 | 0 / 0 / missing / missing | missing / missing |
| S/4 | 2295 / 2287 / 8 / 0 | available | missing | 35.759849071502686 / 1.0 | 361.52 / 397.27979999999997 | 0 / 0 / missing / missing | missing / missing |
| S/5 | 2500 / 2492 / 8 / 0 | available | missing | 35.83781456947327 / 1.0 | 363.24 / 399.0778 | 0 / 0 / missing / missing | missing / missing |
| S/6 | 2479 / 2471 / 8 / 0 | available | missing | 37.00550842285156 / 1.0 | 364.06 / 401.0655 | 0 / 0 / missing / missing | missing / missing |

- H/1 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_SHORT_H_r1.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_SHORT_H_r1/E_SHORT_H_r1_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/2 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_SHORT_H_r2.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_SHORT_H_r2/E_SHORT_H_r2_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/3 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_SHORT_H_r3.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_SHORT_H_r3/E_SHORT_H_r3_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/4 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_SHORT_H_r4.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_SHORT_H_r4/E_SHORT_H_r4_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/5 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_SHORT_H_r5.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_SHORT_H_r5/E_SHORT_H_r5_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/6 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_SHORT_H_r6.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_SHORT_H_r6/E_SHORT_H_r6_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/1 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_SHORT_S_r1.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_SHORT_S_r1/E_SHORT_S_r1_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/2 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_SHORT_S_r2.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_SHORT_S_r2/E_SHORT_S_r2_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/3 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_SHORT_S_r3.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_SHORT_S_r3/E_SHORT_S_r3_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/4 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_SHORT_S_r4.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_SHORT_S_r4/E_SHORT_S_r4_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/5 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_SHORT_S_r5.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_SHORT_S_r5/E_SHORT_S_r5_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/6 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/E_SHORT_S_r6.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-E_SHORT_S_r6/E_SHORT_S_r6_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.

Payload bytes and network bytes are distinct. No formula-priced transfer or inferred network volume is accepted. Per-instance/phase capture aggregates, raw CSV paths, gateway counters and legacy byte fields are retained in RESULTS.json. Unknown coverage is not zero coverage. Initialization is not borrowed from an unrelated deployment or summed across overlapping worker initializations.

### Paired contrasts
- **H1_instance_mix**: not_measured_or_not_qualified
- **H2_migration**: not_measured_or_not_qualified
- **S_vs_H**: supported
  - Makespan reduction median +29.74%; range [0.2814, 0.3395]; 6 whole-run pairs.
  - Sustained progress: missing_common_interval_no_sustained_claim; paired native throughput and terminal drain: [].
  - Rep 1: baseline 51.47 s; treatment 36.66 s; makespan reduction +28.78%; throughput increase +40.40%.
  - Rep 2: baseline 51.07 s; treatment 36.19 s; makespan reduction +29.14%; throughput increase +41.12%.
  - Rep 3: baseline 51.09 s; treatment 33.74 s; makespan reduction +33.95%; throughput increase +51.41%.
  - Rep 4: baseline 51.45 s; treatment 35.76 s; makespan reduction +30.49%; throughput increase +43.86%.
  - Rep 5: baseline 51.44 s; treatment 35.84 s; makespan reduction +30.33%; throughput increase +43.53%.
  - Rep 6: baseline 51.5 s; treatment 37.01 s; makespan reduction +28.14%; throughput increase +39.16%.
- **M0_vs_H**: not_measured_or_not_qualified
- **MR_vs_H**: not_measured_or_not_qualified
- **M0_vs_S**: not_measured_or_not_qualified
- **MR_vs_S**: not_measured_or_not_qualified
- **H3_transport**: not_measured_or_not_qualified
- **H4_system**: not_measured_or_not_qualified

## Workload W0
| Arm | Rep | Makespan s | Useful tokens | Useful tok/s | Useful tok/s/allocated chip | Eligible | Exclusions |
|---|---:|---:|---:|---:|---:|---|---|
| H | 1 | 467.9825 | 721595 | 1541.927 | 48.1852 | True | none |
| H | 2 | 468.0961 | 721595 | 1541.553 | 48.1735 | True | none |
| H | 3 | 471.234 | 721595 | 1531.288 | 47.8528 | True | none |
| H | 4 | 470.2108 | 721595 | 1534.62 | 47.9569 | True | none |
| H | 5 | 472.1958 | 721595 | 1528.169 | 47.7553 | True | none |
| H | 6 | 473.3154 | 721595 | 1524.554 | 47.6423 | True | none |
| M0 | 1 | 402.5579 | 721595 | 1792.525 | 56.0164 | True | none |
| M0 | 2 | 403.0331 | 721595 | 1790.411 | 55.9504 | True | none |
| M0 | 3 | 407.552 | 721595 | 1770.559 | 55.33 | True | none |
| M0 | 4 | 403.4372 | 721595 | 1788.618 | 55.8943 | True | none |
| M0 | 5 | 404.8645 | 721595 | 1782.312 | 55.6973 | True | none |
| M0 | 6 | 397.2381 | 721595 | 1816.53 | 56.7666 | True | none |
| MK-H | 1 | 401.2917 | 721595 | 1798.181 | 56.1932 | True | none |
| MK-H | 2 | 402.4362 | 721595 | 1793.067 | 56.0333 | True | none |
| MK-H | 3 | 401.7255 | 721595 | 1796.239 | 56.1325 | True | none |
| MK-H | 4 | 402.9774 | 721595 | 1790.659 | 55.9581 | True | none |
| MK-H | 5 | 405.547 | 721595 | 1779.313 | 55.6035 | True | none |
| MK-H | 6 | 408.8069 | 721595 | 1765.124 | 55.1601 | True | none |
| MR | 1 | 401.9695 | 721595 | 1795.149 | 56.0984 | True | none |
| MR | 2 | 400.8605 | 721595 | 1800.115 | 56.2536 | True | none |
| MR | 3 | 401.8987 | 721595 | 1795.465 | 56.1083 | True | none |
| MR | 4 | 406.5354 | 721595 | 1774.987 | 55.4683 | True | none |
| MR | 5 | 403.7286 | 721595 | 1787.327 | 55.854 | True | none |
| MR | 6 | 400.8299 | 721595 | 1800.253 | 56.2579 | True | none |
| S | 1 | 351.3465 | 721595 | 2053.799 | 64.1812 | True | none |
| S | 2 | 346.7504 | 721595 | 2081.021 | 65.0319 | True | none |
| S | 3 | 348.7145 | 721595 | 2069.3 | 64.6656 | True | none |
| S | 4 | 346.5295 | 721595 | 2082.348 | 65.0734 | True | none |
| S | 5 | 347.4813 | 721595 | 2076.644 | 64.8951 | True | none |
| S | 6 | 350.1999 | 721595 | 2060.523 | 64.3914 | True | none |

### Per-run accounting and iteration eligibility
| Arm/rep | Capture rows / eligible / control / invalid | Retirement/artifact status | Deep-backlog tok/s | Terminal drain s / fraction | Initialization s / init+serving s | Migrations committed / declined / eligible / fallback | Actual payload / network bytes |
|---|---|---|---|---|---|---|---|
| H/1 | 19543 / 19539 / 4 / 0 | available | 1657.990803151352 | 164.02525115013123 / 0.35049437602250516 | 366.09 / 834.0725 | 0 / 0 / missing / missing | missing / missing |
| H/2 | 19259 / 19255 / 4 / 0 | available | 1654.0328282544788 | 161.95419573783875 / 0.34598490122049824 | 363.37 / 831.4661 | 0 / 0 / missing / missing | missing / missing |
| H/3 | 19409 / 19405 / 4 / 0 | available | 1640.4419717966475 | 164.25262260437012 / 0.34855850482038847 | 364.6 / 835.8340000000001 | 0 / 0 / missing / missing | missing / missing |
| H/4 | 19162 / 19158 / 4 / 0 | available | 1647.7561954836601 | 163.12935090065002 / 0.34692811122270395 | 361.53 / 831.7408 | 0 / 0 / missing / missing | missing / missing |
| H/5 | 19021 / 19017 / 4 / 0 | available | 1638.2512895849795 | 166.40247559547424 / 0.35240143158472387 | 363.45 / 835.6458 | 0 / 0 / missing / missing | missing / missing |
| H/6 | 19222 / 19218 / 4 / 0 | available | 1652.4912609994499 | 163.36755657196045 / 0.3451557840099417 | 365.18 / 838.4954 | 0 / 0 / missing / missing | missing / missing |
| M0/1 | 35805 / 35799 / 6 / 0 | available | 2027.8269033915876 | 134.35150265693665 / 0.3337445093678275 | 361.58 / 764.1379 | 0 / 0 / missing / missing | missing / missing |
| M0/2 | 33691 / 33685 / 6 / 0 | available | 2018.9813390160625 | 134.93072319030762 / 0.33478823171566857 | 362.58 / 765.6131 | 0 / 0 / missing / missing | missing / missing |
| M0/3 | 35820 / 35814 / 6 / 0 | available | 2026.8484372245528 | 147.17591285705566 / 0.36112176507820937 | 364.13 / 771.682 | 0 / 0 / missing / missing | missing / missing |
| M0/4 | 36100 / 36094 / 6 / 0 | available | 2026.3017386800348 | 141.19626331329346 / 0.3499832482071245 | 363.16 / 766.5972 | 0 / 0 / missing / missing | missing / missing |
| M0/5 | 37411 / 37405 / 6 / 0 | available | 2049.253973656188 | 150.63330125808716 / 0.3720585542358253 | 363.83 / 768.6945000000001 | 0 / 0 / missing / missing | missing / missing |
| M0/6 | 35554 / 35548 / 6 / 0 | available | 2045.9315498280264 | 138.8349711894989 / 0.3495006221422492 | 362.77 / 760.0081 | 0 / 0 / missing / missing | missing / missing |
| MK-H/1 | 35357 / 35351 / 6 / 0 | available | 2016.6354602287086 | 137.58174443244934 / 0.34284725661946447 | 361.58 / 762.8716999999999 | 0 / 0 / missing / missing | missing / missing |
| MK-H/2 | 33735 / 33729 / 6 / 0 | available | 2031.4319549137158 | 141.82898879051208 / 0.35242604852089515 | 362.58 / 765.0162 | 0 / 0 / missing / missing | missing / missing |
| MK-H/3 | 35983 / 35977 / 6 / 0 | available | 2035.4807378721111 | 142.527578830719 / 0.354788442809862 | 364.13 / 765.8555 | 0 / 0 / missing / missing | missing / missing |
| MK-H/4 | 36587 / 36581 / 6 / 0 | available | 2037.3419283499284 | 143.81787657737732 / 0.3568881523052995 | 363.16 / 766.1374000000001 | 0 / 0 / missing / missing | missing / missing |
| MK-H/5 | 37001 / 36995 / 6 / 0 | available | 2027.8392541507844 | 141.589684009552 / 0.3491326115126681 | 363.83 / 769.377 | 0 / 0 / missing / missing | missing / missing |
| MK-H/6 | 34795 / 34789 / 6 / 0 | available | 2003.4907450562866 | 136.68333435058594 / 0.33434690166037645 | 362.77 / 771.5769 | 0 / 0 / missing / missing | missing / missing |
| MR/1 | 33523 / 33517 / 6 / 0 | available | 2076.9403166903403 | 149.4742419719696 / 0.3718546996987691 | 361.58 / 763.5495 | 0 / 0 / missing / missing | missing / missing |
| MR/2 | 36307 / 36301 / 6 / 0 | available | 2055.6112846991764 | 144.92462253570557 / 0.3615337672996024 | 362.58 / 763.4404999999999 | 0 / 0 / missing / missing | missing / missing |
| MR/3 | 35348 / 35342 / 6 / 0 | available | 2050.1337318180676 | 141.28046703338623 / 0.35153253846962 | 364.13 / 766.0287000000001 | 0 / 0 / missing / missing | missing / missing |
| MR/4 | 35368 / 35362 / 6 / 0 | available | 2031.4633393630606 | 143.60325574874878 / 0.35323681345859953 | 363.16 / 769.6954000000001 | 0 / 0 / missing / missing | missing / missing |
| MR/5 | 36035 / 36029 / 6 / 0 | available | 2037.6750105715369 | 142.91165614128113 / 0.3539794909675182 | 363.83 / 767.5586 | 0 / 0 / missing / missing | missing / missing |
| MR/6 | 34169 / 34163 / 6 / 0 | available | 2023.1192006910826 | 139.95244646072388 / 0.3491567435492008 | 362.77 / 763.5998999999999 | 0 / 0 / missing / missing | missing / missing |
| S/1 | 47556 / 47548 / 8 / 0 | available | 2654.7722686987986 | 166.6730613708496 / 0.47438374520386983 | 364.03 / 715.3765 | 0 / 0 / missing / missing | missing / missing |
| S/2 | 46484 / 46476 / 8 / 0 | available | 2662.831103092182 | 163.4153537750244 / 0.47127662146089827 | 363.55 / 710.3004000000001 | 0 / 0 / missing / missing | missing / missing |
| S/3 | 47061 / 47053 / 8 / 0 | available | 2671.9285451367264 | 162.47516632080078 / 0.46592604809628324 | 363.64 / 712.3544999999999 | 0 / 0 / missing / missing | missing / missing |
| S/4 | 46240 / 46232 / 8 / 0 | available | 2660.977243672555 | 161.52995085716248 / 0.4661362028249531 | 361.52 / 708.0495 | 0 / 0 / missing / missing | missing / missing |
| S/5 | 47141 / 47133 / 8 / 0 | available | 2655.398386010899 | 164.25100016593933 / 0.4726901844018913 | 363.24 / 710.7212999999999 | 0 / 0 / missing / missing | missing / missing |
| S/6 | 45494 / 45486 / 8 / 0 | available | 2653.0458215156427 | 164.4279282093048 / 0.46952596165318344 | 364.06 / 714.2599 | 0 / 0 / missing / missing | missing / missing |

- H/1 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_H_r1.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_H_r1/W0_H_r1_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/2 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_H_r2.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_H_r2/W0_H_r2_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/3 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_H_r3.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_H_r3/W0_H_r3_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/4 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_H_r4.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_H_r4/W0_H_r4_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/5 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_H_r5.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_H_r5/W0_H_r5_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/6 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_H_r6.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_H_r6/W0_H_r6_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- M0/1 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_M0_r1.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_M0_r1/W0_M0_r1_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- M0/2 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_M0_r2.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_M0_r2/W0_M0_r2_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- M0/3 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_M0_r3.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_M0_r3/W0_M0_r3_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- M0/4 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_M0_r4.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_M0_r4/W0_M0_r4_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- M0/5 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_M0_r5.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_M0_r5/W0_M0_r5_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- M0/6 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_M0_r6.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_M0_r6/W0_M0_r6_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MK-H/1 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_MK-H_r1.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_MK-H_r1/W0_MK-H_r1_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MK-H/2 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_MK-H_r2.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_MK-H_r2/W0_MK-H_r2_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MK-H/3 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_MK-H_r3.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_MK-H_r3/W0_MK-H_r3_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MK-H/4 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_MK-H_r4.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_MK-H_r4/W0_MK-H_r4_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MK-H/5 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_MK-H_r5.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_MK-H_r5/W0_MK-H_r5_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MK-H/6 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_MK-H_r6.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_MK-H_r6/W0_MK-H_r6_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MR/1 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_MR_r1.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_MR_r1/W0_MR_r1_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MR/2 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_MR_r2.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_MR_r2/W0_MR_r2_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MR/3 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_MR_r3.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_MR_r3/W0_MR_r3_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MR/4 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_MR_r4.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_MR_r4/W0_MR_r4_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MR/5 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_MR_r5.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_MR_r5/W0_MR_r5_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MR/6 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_MR_r6.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_MR_r6/W0_MR_r6_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/1 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_S_r1.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_S_r1/W0_S_r1_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/2 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_S_r2.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_S_r2/W0_S_r2_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/3 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_S_r3.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_S_r3/W0_S_r3_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/4 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_S_r4.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_S_r4/W0_S_r4_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/5 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_S_r5.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_S_r5/W0_S_r5_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/6 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W0_S_r6.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W0_S_r6/W0_S_r6_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.

Payload bytes and network bytes are distinct. No formula-priced transfer or inferred network volume is accepted. Per-instance/phase capture aggregates, raw CSV paths, gateway counters and legacy byte fields are retained in RESULTS.json. Unknown coverage is not zero coverage. Initialization is not borrowed from an unrelated deployment or summed across overlapping worker initializations.

### Paired contrasts
- **H1_instance_mix**: not_supported
  - Makespan reduction median -16.33%; range [-0.1687, -0.1343]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': -0.23615787037526192, 'baseline_terminal_drain_s': 166.6730613708496, 'treatment_terminal_drain_s': 134.35150265693665}, {'repetition': 2, 'native_throughput_increase': -0.2417914389419803, 'baseline_terminal_drain_s': 163.4153537750244, 'treatment_terminal_drain_s': 134.93072319030762}, {'repetition': 3, 'native_throughput_increase': -0.24142865238155686, 'baseline_terminal_drain_s': 162.47516632080078, 'treatment_terminal_drain_s': 147.17591285705566}, {'repetition': 4, 'native_throughput_increase': -0.23851218814504815, 'baseline_terminal_drain_s': 161.52995085716248, 'treatment_terminal_drain_s': 141.19626331329346}, {'repetition': 5, 'native_throughput_increase': -0.22826872816824217, 'baseline_terminal_drain_s': 164.25100016593933, 'treatment_terminal_drain_s': 150.63330125808716}, {'repetition': 6, 'native_throughput_increase': -0.22883670789401656, 'baseline_terminal_drain_s': 164.4279282093048, 'treatment_terminal_drain_s': 138.8349711894989}].
  - Rep 1: baseline 351.35 s; treatment 402.56 s; makespan reduction -14.58%; throughput increase -12.72%.
  - Rep 2: baseline 346.75 s; treatment 403.03 s; makespan reduction -16.23%; throughput increase -13.96%.
  - Rep 3: baseline 348.71 s; treatment 407.55 s; makespan reduction -16.87%; throughput increase -14.44%.
  - Rep 4: baseline 346.53 s; treatment 403.44 s; makespan reduction -16.42%; throughput increase -14.11%.
  - Rep 5: baseline 347.48 s; treatment 404.86 s; makespan reduction -16.51%; throughput increase -14.17%.
  - Rep 6: baseline 350.2 s; treatment 397.24 s; makespan reduction -13.43%; throughput increase -11.84%.
- **H2_migration**: not_exercised_no_migrations
  - Makespan reduction median +0.22%; range [-0.009, 0.0139]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': 0.024219726652511264, 'baseline_terminal_drain_s': 134.35150265693665, 'treatment_terminal_drain_s': 149.4742419719696}, {'repetition': 2, 'native_throughput_increase': 0.01814278565891314, 'baseline_terminal_drain_s': 134.93072319030762, 'treatment_terminal_drain_s': 144.92462253570557}, {'repetition': 3, 'native_throughput_increase': 0.011488424179067147, 'baseline_terminal_drain_s': 147.17591285705566, 'treatment_terminal_drain_s': 141.28046703338623}, {'repetition': 4, 'native_throughput_increase': 0.00254730111734891, 'baseline_terminal_drain_s': 141.19626331329346, 'treatment_terminal_drain_s': 143.60325574874878}, {'repetition': 5, 'native_throughput_increase': -0.005650330917252089, 'baseline_terminal_drain_s': 150.63330125808716, 'treatment_terminal_drain_s': 142.91165614128113}, {'repetition': 6, 'native_throughput_increase': -0.011150103794460464, 'baseline_terminal_drain_s': 138.8349711894989, 'treatment_terminal_drain_s': 139.95244646072388}].
  - Rep 1: baseline 402.56 s; treatment 401.97 s; makespan reduction +0.15%; throughput increase +0.15%.
  - Rep 2: baseline 403.03 s; treatment 400.86 s; makespan reduction +0.54%; throughput increase +0.54%.
  - Rep 3: baseline 407.55 s; treatment 401.9 s; makespan reduction +1.39%; throughput increase +1.41%.
  - Rep 4: baseline 403.44 s; treatment 406.54 s; makespan reduction -0.77%; throughput increase -0.76%.
  - Rep 5: baseline 404.86 s; treatment 403.73 s; makespan reduction +0.28%; throughput increase +0.28%.
  - Rep 6: baseline 397.24 s; treatment 400.83 s; makespan reduction -0.90%; throughput increase -0.90%.
- **S_vs_H**: supported
  - Makespan reduction median +26.01%; range [0.2492, 0.2641]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': 0.6011984286359482, 'baseline_terminal_drain_s': 164.02525115013123, 'treatment_terminal_drain_s': 166.6730613708496}, {'repetition': 2, 'native_throughput_increase': 0.6099022084720653, 'baseline_terminal_drain_s': 161.95419573783875, 'treatment_terminal_drain_s': 163.4153537750244}, {'repetition': 3, 'native_throughput_increase': 0.6287857730257733, 'baseline_terminal_drain_s': 164.25262260437012, 'treatment_terminal_drain_s': 162.47516632080078}, {'repetition': 4, 'native_throughput_increase': 0.614909566698056, 'baseline_terminal_drain_s': 163.12935090065002, 'treatment_terminal_drain_s': 161.52995085716248}, {'repetition': 5, 'native_throughput_increase': 0.6208736735886193, 'baseline_terminal_drain_s': 166.40247559547424, 'treatment_terminal_drain_s': 164.25100016593933}, {'repetition': 6, 'native_throughput_increase': 0.6054825124527698, 'baseline_terminal_drain_s': 163.36755657196045, 'treatment_terminal_drain_s': 164.4279282093048}].
  - Rep 1: baseline 467.98 s; treatment 351.35 s; makespan reduction +24.92%; throughput increase +33.20%.
  - Rep 2: baseline 468.1 s; treatment 346.75 s; makespan reduction +25.92%; throughput increase +35.00%.
  - Rep 3: baseline 471.23 s; treatment 348.71 s; makespan reduction +26.00%; throughput increase +35.13%.
  - Rep 4: baseline 470.21 s; treatment 346.53 s; makespan reduction +26.30%; throughput increase +35.69%.
  - Rep 5: baseline 472.2 s; treatment 347.48 s; makespan reduction +26.41%; throughput increase +35.89%.
  - Rep 6: baseline 473.32 s; treatment 350.2 s; makespan reduction +26.01%; throughput increase +35.16%.
- **M0_vs_H**: supported
  - Makespan reduction median +14.09%; range [0.1351, 0.1607]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': 0.22306281768106673, 'baseline_terminal_drain_s': 164.02525115013123, 'treatment_terminal_drain_s': 134.35150265693665}, {'repetition': 2, 'native_throughput_increase': 0.22064163692973282, 'baseline_terminal_drain_s': 161.95419573783875, 'treatment_terminal_drain_s': 134.93072319030762}, {'repetition': 3, 'native_throughput_increase': 0.23555021882590865, 'baseline_terminal_drain_s': 164.25262260437012, 'treatment_terminal_drain_s': 147.17591285705566}, {'repetition': 4, 'native_throughput_increase': 0.2297339522885311, 'baseline_terminal_drain_s': 163.12935090065002, 'treatment_terminal_drain_s': 141.19626331329346}, {'repetition': 5, 'native_throughput_increase': 0.25087890159715864, 'baseline_terminal_drain_s': 166.40247559547424, 'treatment_terminal_drain_s': 150.63330125808716}, {'repetition': 6, 'native_throughput_increase': 0.23808917972166355, 'baseline_terminal_drain_s': 163.36755657196045, 'treatment_terminal_drain_s': 138.8349711894989}].
  - Rep 1: baseline 467.98 s; treatment 402.56 s; makespan reduction +13.98%; throughput increase +16.25%.
  - Rep 2: baseline 468.1 s; treatment 403.03 s; makespan reduction +13.90%; throughput increase +16.14%.
  - Rep 3: baseline 471.23 s; treatment 407.55 s; makespan reduction +13.51%; throughput increase +15.63%.
  - Rep 4: baseline 470.21 s; treatment 403.44 s; makespan reduction +14.20%; throughput increase +16.55%.
  - Rep 5: baseline 472.2 s; treatment 404.86 s; makespan reduction +14.26%; throughput increase +16.63%.
  - Rep 6: baseline 473.32 s; treatment 397.24 s; makespan reduction +16.07%; throughput increase +19.15%.
- **MR_vs_H**: supported
  - Makespan reduction median +14.43%; range [0.1354, 0.1531]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': 0.25268506480415254, 'baseline_terminal_drain_s': 164.02525115013123, 'treatment_terminal_drain_s': 149.4742419719696}, {'repetition': 2, 'native_throughput_increase': 0.24278747651489385, 'baseline_terminal_drain_s': 161.95419573783875, 'treatment_terminal_drain_s': 144.92462253570557}, {'repetition': 3, 'native_throughput_increase': 0.24974474383431966, 'baseline_terminal_drain_s': 164.25262260437012, 'treatment_terminal_drain_s': 141.28046703338623}, {'repetition': 4, 'native_throughput_increase': 0.23286645495923763, 'baseline_terminal_drain_s': 163.12935090065002, 'treatment_terminal_drain_s': 143.60325574874878}, {'repetition': 5, 'native_throughput_increase': 0.24381102186572612, 'baseline_terminal_drain_s': 166.40247559547424, 'treatment_terminal_drain_s': 142.91165614128113}, {'repetition': 6, 'native_throughput_increase': 0.2242843568609687, 'baseline_terminal_drain_s': 163.36755657196045, 'treatment_terminal_drain_s': 139.95244646072388}].
  - Rep 1: baseline 467.98 s; treatment 401.97 s; makespan reduction +14.11%; throughput increase +16.42%.
  - Rep 2: baseline 468.1 s; treatment 400.86 s; makespan reduction +14.36%; throughput increase +16.77%.
  - Rep 3: baseline 471.23 s; treatment 401.9 s; makespan reduction +14.71%; throughput increase +17.25%.
  - Rep 4: baseline 470.21 s; treatment 406.54 s; makespan reduction +13.54%; throughput increase +15.66%.
  - Rep 5: baseline 472.2 s; treatment 403.73 s; makespan reduction +14.50%; throughput increase +16.96%.
  - Rep 6: baseline 473.32 s; treatment 400.83 s; makespan reduction +15.31%; throughput increase +18.08%.
- **M0_vs_S**: not_supported
  - Makespan reduction median -16.33%; range [-0.1687, -0.1343]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': -0.23615787037526192, 'baseline_terminal_drain_s': 166.6730613708496, 'treatment_terminal_drain_s': 134.35150265693665}, {'repetition': 2, 'native_throughput_increase': -0.2417914389419803, 'baseline_terminal_drain_s': 163.4153537750244, 'treatment_terminal_drain_s': 134.93072319030762}, {'repetition': 3, 'native_throughput_increase': -0.24142865238155686, 'baseline_terminal_drain_s': 162.47516632080078, 'treatment_terminal_drain_s': 147.17591285705566}, {'repetition': 4, 'native_throughput_increase': -0.23851218814504815, 'baseline_terminal_drain_s': 161.52995085716248, 'treatment_terminal_drain_s': 141.19626331329346}, {'repetition': 5, 'native_throughput_increase': -0.22826872816824217, 'baseline_terminal_drain_s': 164.25100016593933, 'treatment_terminal_drain_s': 150.63330125808716}, {'repetition': 6, 'native_throughput_increase': -0.22883670789401656, 'baseline_terminal_drain_s': 164.4279282093048, 'treatment_terminal_drain_s': 138.8349711894989}].
  - Rep 1: baseline 351.35 s; treatment 402.56 s; makespan reduction -14.58%; throughput increase -12.72%.
  - Rep 2: baseline 346.75 s; treatment 403.03 s; makespan reduction -16.23%; throughput increase -13.96%.
  - Rep 3: baseline 348.71 s; treatment 407.55 s; makespan reduction -16.87%; throughput increase -14.44%.
  - Rep 4: baseline 346.53 s; treatment 403.44 s; makespan reduction -16.42%; throughput increase -14.11%.
  - Rep 5: baseline 347.48 s; treatment 404.86 s; makespan reduction -16.51%; throughput increase -14.17%.
  - Rep 6: baseline 350.2 s; treatment 397.24 s; makespan reduction -13.43%; throughput increase -11.84%.
- **MR_vs_S**: not_supported
  - Makespan reduction median -15.42%; range [-0.1732, -0.1441]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': -0.21765782279007873, 'baseline_terminal_drain_s': 166.6730613708496, 'treatment_terminal_drain_s': 149.4742419719696}, {'repetition': 2, 'native_throughput_increase': -0.22803542353395168, 'baseline_terminal_drain_s': 163.4153537750244, 'treatment_terminal_drain_s': 144.92462253570557}, {'repetition': 3, 'native_throughput_increase': -0.2327138629700296, 'baseline_terminal_drain_s': 162.47516632080078, 'treatment_terminal_drain_s': 141.28046703338623}, {'repetition': 4, 'native_throughput_increase': -0.23657244939106237, 'baseline_terminal_drain_s': 161.52995085716248, 'treatment_terminal_drain_s': 143.60325574874878}, {'repetition': 5, 'native_throughput_increase': -0.2326292652332833, 'baseline_terminal_drain_s': 164.25100016593933, 'treatment_terminal_drain_s': 142.91165614128113}, {'repetition': 6, 'native_throughput_increase': -0.23743525864347614, 'baseline_terminal_drain_s': 164.4279282093048, 'treatment_terminal_drain_s': 139.95244646072388}].
  - Rep 1: baseline 351.35 s; treatment 401.97 s; makespan reduction -14.41%; throughput increase -12.59%.
  - Rep 2: baseline 346.75 s; treatment 400.86 s; makespan reduction -15.60%; throughput increase -13.50%.
  - Rep 3: baseline 348.71 s; treatment 401.9 s; makespan reduction -15.25%; throughput increase -13.23%.
  - Rep 4: baseline 346.53 s; treatment 406.54 s; makespan reduction -17.32%; throughput increase -14.76%.
  - Rep 5: baseline 347.48 s; treatment 403.73 s; makespan reduction -16.19%; throughput increase -13.93%.
  - Rep 6: baseline 350.2 s; treatment 400.83 s; makespan reduction -14.46%; throughput increase -12.63%.
- **H3_transport**: not_exercised_no_transferred_KV
  - Makespan reduction median -0.17%; range [-0.0199, 0.0088]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': -0.02903543061734637, 'baseline_terminal_drain_s': 149.4742419719696, 'treatment_terminal_drain_s': 137.58174443244934}, {'repetition': 2, 'native_throughput_increase': -0.011762598291534032, 'baseline_terminal_drain_s': 144.92462253570557, 'treatment_terminal_drain_s': 141.82898879051208}, {'repetition': 3, 'native_throughput_increase': -0.007147335668177157, 'baseline_terminal_drain_s': 141.28046703338623, 'treatment_terminal_drain_s': 142.527578830719}, {'repetition': 4, 'native_throughput_increase': 0.0028937706494427395, 'baseline_terminal_drain_s': 143.60325574874878, 'treatment_terminal_drain_s': 143.81787657737732}, {'repetition': 5, 'native_throughput_increase': -0.004826950504729255, 'baseline_terminal_drain_s': 142.91165614128113, 'treatment_terminal_drain_s': 141.589684009552}, {'repetition': 6, 'native_throughput_increase': -0.009702075699786206, 'baseline_terminal_drain_s': 139.95244646072388, 'treatment_terminal_drain_s': 136.68333435058594}].
  - Rep 1: baseline 401.97 s; treatment 401.29 s; makespan reduction +0.17%; throughput increase +0.17%.
  - Rep 2: baseline 400.86 s; treatment 402.44 s; makespan reduction -0.39%; throughput increase -0.39%.
  - Rep 3: baseline 401.9 s; treatment 401.73 s; makespan reduction +0.04%; throughput increase +0.04%.
  - Rep 4: baseline 406.54 s; treatment 402.98 s; makespan reduction +0.88%; throughput increase +0.88%.
  - Rep 5: baseline 403.73 s; treatment 405.55 s; makespan reduction -0.45%; throughput increase -0.45%.
  - Rep 6: baseline 400.83 s; treatment 408.81 s; makespan reduction -1.99%; throughput increase -1.95%.
- **H4_system**: not_supported
  - Makespan reduction median -15.42%; range [-0.1732, -0.1441]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': -0.21765782279007873, 'baseline_terminal_drain_s': 166.6730613708496, 'treatment_terminal_drain_s': 149.4742419719696}, {'repetition': 2, 'native_throughput_increase': -0.22803542353395168, 'baseline_terminal_drain_s': 163.4153537750244, 'treatment_terminal_drain_s': 144.92462253570557}, {'repetition': 3, 'native_throughput_increase': -0.2327138629700296, 'baseline_terminal_drain_s': 162.47516632080078, 'treatment_terminal_drain_s': 141.28046703338623}, {'repetition': 4, 'native_throughput_increase': -0.23657244939106237, 'baseline_terminal_drain_s': 161.52995085716248, 'treatment_terminal_drain_s': 143.60325574874878}, {'repetition': 5, 'native_throughput_increase': -0.2326292652332833, 'baseline_terminal_drain_s': 164.25100016593933, 'treatment_terminal_drain_s': 142.91165614128113}, {'repetition': 6, 'native_throughput_increase': -0.23743525864347614, 'baseline_terminal_drain_s': 164.4279282093048, 'treatment_terminal_drain_s': 139.95244646072388}].
  - Rep 1: baseline 351.35 s; treatment 401.97 s; makespan reduction -14.41%; throughput increase -12.59%.
  - Rep 2: baseline 346.75 s; treatment 400.86 s; makespan reduction -15.60%; throughput increase -13.50%.
  - Rep 3: baseline 348.71 s; treatment 401.9 s; makespan reduction -15.25%; throughput increase -13.23%.
  - Rep 4: baseline 346.53 s; treatment 406.54 s; makespan reduction -17.32%; throughput increase -14.76%.
  - Rep 5: baseline 347.48 s; treatment 403.73 s; makespan reduction -16.19%; throughput increase -13.93%.
  - Rep 6: baseline 350.2 s; treatment 400.83 s; makespan reduction -14.46%; throughput increase -12.63%.

## Workload W1
| Arm | Rep | Makespan s | Useful tokens | Useful tok/s | Useful tok/s/allocated chip | Eligible | Exclusions |
|---|---:|---:|---:|---:|---:|---|---|
| H | 1 | 488.8694 | 663746 | 1357.716 | 42.4286 | True | none |
| H | 2 | 492.0879 | 663746 | 1348.836 | 42.1511 | True | none |
| H | 3 | 494.333 | 663746 | 1342.71 | 41.9597 | True | none |
| H | 4 | 491.7353 | 663746 | 1349.803 | 42.1814 | True | none |
| H | 5 | 489.7636 | 663746 | 1355.237 | 42.3512 | True | none |
| H | 6 | 494.2443 | 663746 | 1342.951 | 41.9672 | True | none |
| M0 | 1 | 510.0702 | 663746 | 1301.284 | 40.6651 | True | none |
| M0 | 2 | 507.3565 | 663746 | 1308.244 | 40.8826 | True | none |
| M0 | 3 | 505.6145 | 663746 | 1312.751 | 41.0235 | True | none |
| M0 | 4 | 510.9991 | 663746 | 1298.918 | 40.5912 | True | none |
| M0 | 5 | 523.394 | 663746 | 1268.158 | 39.6299 | True | none |
| M0 | 6 | 515.5023 | 663746 | 1287.571 | 40.2366 | True | none |
| MR | 1 | 508.1735 | 663746 | 1306.141 | 40.8169 | True | none |
| MR | 2 | 507.1169 | 663746 | 1308.862 | 40.9019 | True | none |
| MR | 3 | 511.5916 | 663746 | 1297.414 | 40.5442 | True | none |
| MR | 4 | 510.2953 | 663746 | 1300.71 | 40.6472 | True | none |
| MR | 5 | 512.3177 | 663746 | 1295.575 | 40.4867 | True | none |
| MR | 6 | 505.8184 | 663746 | 1312.222 | 41.0069 | True | none |
| S | 1 | 500.937 | 663746 | 1325.009 | 41.4065 | True | none |
| S | 2 | 500.8575 | 663746 | 1325.219 | 41.4131 | True | none |
| S | 3 | 499.5469 | 663746 | 1328.696 | 41.5218 | True | none |
| S | 4 | 502.6553 | 663746 | 1320.479 | 41.265 | True | none |
| S | 5 | 501.9039 | 663746 | 1322.456 | 41.3268 | True | none |
| S | 6 | 502.3198 | 663746 | 1321.361 | 41.2925 | True | none |

### Per-run accounting and iteration eligibility
| Arm/rep | Capture rows / eligible / control / invalid | Retirement/artifact status | Deep-backlog tok/s | Terminal drain s / fraction | Initialization s / init+serving s | Migrations committed / declined / eligible / fallback | Actual payload / network bytes |
|---|---|---|---|---|---|---|---|
| H/1 | 35939 / 35935 / 4 / 0 | available | 1443.51360080294 | 369.32444882392883 / 0.7554664417750695 | 366.09 / 854.9594 | 0 / 0 / missing / missing | missing / missing |
| H/2 | 35902 / 35898 / 4 / 0 | available | 1479.8361895582505 | 378.8133544921875 / 0.7698082515735601 | 363.37 / 855.4579 | 0 / 0 / missing / missing | missing / missing |
| H/3 | 36301 / 36297 / 4 / 0 | available | 1442.5424500452912 | 377.50712418556213 / 0.7636696540966204 | 364.6 / 858.933 | 0 / 0 / missing / missing | missing / missing |
| H/4 | 36070 / 36066 / 4 / 0 | available | 1455.6203314990107 | 375.4804232120514 / 0.7635824334740375 | 361.53 / 853.2653 | 0 / 0 / missing / missing | missing / missing |
| H/5 | 36359 / 36355 / 4 / 0 | available | 1435.7255166853665 | 370.55923652648926 / 0.7566082843183835 | 363.45 / 853.2136 | 0 / 0 / missing / missing | missing / missing |
| H/6 | 36460 / 36456 / 4 / 0 | available | 1443.278929012961 | 373.540864944458 / 0.7557818823049791 | 365.18 / 859.4243 | 0 / 0 / missing / missing | missing / missing |
| M0/1 | 67767 / 67761 / 6 / 0 | available | 1339.294641496267 | 396.0840561389923 / 0.7765285700089741 | 361.58 / 871.6502 | 0 / 0 / missing / missing | missing / missing |
| M0/2 | 68050 / 68044 / 6 / 0 | available | 1430.300864034027 | 391.1001343727112 / 0.7708586146908882 | 362.58 / 869.9365 | 0 / 0 / missing / missing | missing / missing |
| M0/3 | 67191 / 67185 / 6 / 0 | available | 1413.1029118820843 | 390.11543107032776 / 0.771566959856234 | 364.13 / 869.7445 | 0 / 0 / missing / missing | missing / missing |
| M0/4 | 67830 / 67824 / 6 / 0 | available | 1407.819928547305 | 397.44418382644653 / 0.7777786031978983 | 363.16 / 874.1591000000001 | 0 / 0 / missing / missing | missing / missing |
| M0/5 | 69125 / 69119 / 6 / 0 | available | 1421.0747183186902 | 405.1335971355438 / 0.7740509327353472 | 363.83 / 887.2239999999999 | 0 / 0 / missing / missing | missing / missing |
| M0/6 | 69433 / 69427 / 6 / 0 | available | 1433.639778335456 | 405.50921154022217 / 0.7866293178257265 | 362.77 / 878.2723 | 0 / 0 / missing / missing | missing / missing |
| MR/1 | 66003 / 65997 / 6 / 0 | available | 1450.5125858353938 | 395.03443670272827 / 0.7773613812834784 | 361.58 / 869.7535 | 0 / 0 / missing / missing | missing / missing |
| MR/2 | 66990 / 66984 / 6 / 0 | available | 1414.4992182439416 | 392.63214468955994 / 0.7742439070840593 | 362.58 / 869.6968999999999 | 0 / 0 / missing / missing | missing / missing |
| MR/3 | 66916 / 66910 / 6 / 0 | available | 1384.8365020425802 | 398.77194261550903 / 0.7794732054511221 | 364.13 / 875.7216000000001 | 0 / 0 / missing / missing | missing / missing |
| MR/4 | 68640 / 68634 / 6 / 0 | available | 1391.0658366448629 | 398.8902380466461 / 0.7816850438087344 | 363.16 / 873.4553000000001 | 0 / 0 / missing / missing | missing / missing |
| MR/5 | 68106 / 68100 / 6 / 0 | available | 1432.313876247509 | 400.2944085597992 / 0.781340220799752 | 363.83 / 876.1477 | 0 / 0 / missing / missing | missing / missing |
| MR/6 | 67656 / 67650 / 6 / 0 | available | 1401.8584416134688 | 392.4313244819641 / 0.7758343631901304 | 362.77 / 868.5884 | 0 / 0 / missing / missing | missing / missing |
| S/1 | 93435 / 93427 / 8 / 0 | available | 1580.2758034035573 | 404.0343713760376 / 0.8065572367405276 | 364.03 / 864.967 | 0 / 0 / missing / missing | missing / missing |
| S/2 | 92876 / 92868 / 8 / 0 | available | 1589.9430194134936 | 393.55296564102173 / 0.7857582759123842 | 363.55 / 864.4075 | 0 / 0 / missing / missing | missing / missing |
| S/3 | 92398 / 92390 / 8 / 0 | available | 1564.1765819840296 | 398.7017116546631 / 0.7981267066083665 | 363.64 / 863.1868999999999 | 0 / 0 / missing / missing | missing / missing |
| S/4 | 92523 / 92515 / 8 / 0 | available | 1603.05808415913 | 407.15079641342163 / 0.8099999416351448 | 361.52 / 864.1753 | 0 / 0 / missing / missing | missing / missing |
| S/5 | 92984 / 92976 / 8 / 0 | available | 1667.2673693049635 | 407.844957113266 / 0.8125957541108313 | 363.24 / 865.1439 | 0 / 0 / missing / missing | missing / missing |
| S/6 | 92051 / 92043 / 8 / 0 | available | 1587.2729214311141 | 398.8396894931793 / 0.7939954932912819 | 364.06 / 866.3797999999999 | 0 / 0 / missing / missing | missing / missing |

- H/1 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_H_r1.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_H_r1/W1_H_r1_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/2 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_H_r2.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_H_r2/W1_H_r2_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/3 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_H_r3.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_H_r3/W1_H_r3_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/4 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_H_r4.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_H_r4/W1_H_r4_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/5 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_H_r5.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_H_r5/W1_H_r5_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/6 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_H_r6.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_H_r6/W1_H_r6_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- M0/1 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_M0_r1.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_M0_r1/W1_M0_r1_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- M0/2 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_M0_r2.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_M0_r2/W1_M0_r2_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- M0/3 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_M0_r3.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_M0_r3/W1_M0_r3_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- M0/4 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_M0_r4.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_M0_r4/W1_M0_r4_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- M0/5 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_M0_r5.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_M0_r5/W1_M0_r5_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- M0/6 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_M0_r6.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_M0_r6/W1_M0_r6_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MR/1 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_MR_r1.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_MR_r1/W1_MR_r1_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MR/2 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_MR_r2.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_MR_r2/W1_MR_r2_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MR/3 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_MR_r3.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_MR_r3/W1_MR_r3_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MR/4 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_MR_r4.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_MR_r4/W1_MR_r4_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MR/5 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_MR_r5.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_MR_r5/W1_MR_r5_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MR/6 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_MR_r6.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_MR_r6/W1_MR_r6_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/1 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_S_r1.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_S_r1/W1_S_r1_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/2 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_S_r2.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_S_r2/W1_S_r2_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/3 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_S_r3.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_S_r3/W1_S_r3_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/4 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_S_r4.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_S_r4/W1_S_r4_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/5 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_S_r5.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_S_r5/W1_S_r5_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/6 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W1_S_r6.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W1_S_r6/W1_S_r6_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.

Payload bytes and network bytes are distinct. No formula-priced transfer or inferred network volume is accepted. Per-instance/phase capture aggregates, raw CSV paths, gateway counters and legacy byte fields are retained in RESULTS.json. Unknown coverage is not zero coverage. Initialization is not borrowed from an unrelated deployment or summed across overlapping worker initializations.

### Paired contrasts
- **H1_instance_mix**: not_supported
  - Makespan reduction median -4.11%; range [-0.0687, -0.0228]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': -0.07219811385822905, 'baseline_terminal_drain_s': 369.32444882392883, 'treatment_terminal_drain_s': 396.0840561389923}, {'repetition': 2, 'native_throughput_increase': -0.03347351948394395, 'baseline_terminal_drain_s': 378.8133544921875, 'treatment_terminal_drain_s': 391.1001343727112}, {'repetition': 3, 'native_throughput_increase': -0.020408091396050487, 'baseline_terminal_drain_s': 377.50712418556213, 'treatment_terminal_drain_s': 390.11543107032776}, {'repetition': 4, 'native_throughput_increase': -0.03283851009588501, 'baseline_terminal_drain_s': 375.4804232120514, 'treatment_terminal_drain_s': 397.44418382644653}, {'repetition': 5, 'native_throughput_increase': -0.010204456350751734, 'baseline_terminal_drain_s': 370.55923652648926, 'treatment_terminal_drain_s': 405.1335971355438}, {'repetition': 6, 'native_throughput_increase': -0.006678647130320958, 'baseline_terminal_drain_s': 373.540864944458, 'treatment_terminal_drain_s': 405.50921154022217}].
  - Rep 1: baseline 488.87 s; treatment 510.07 s; makespan reduction -4.34%; throughput increase -4.16%.
  - Rep 2: baseline 492.09 s; treatment 507.36 s; makespan reduction -3.10%; throughput increase -3.01%.
  - Rep 3: baseline 494.33 s; treatment 505.61 s; makespan reduction -2.28%; throughput increase -2.23%.
  - Rep 4: baseline 491.74 s; treatment 511.0 s; makespan reduction -3.92%; throughput increase -3.77%.
  - Rep 5: baseline 489.76 s; treatment 523.39 s; makespan reduction -6.87%; throughput increase -6.43%.
  - Rep 6: baseline 494.24 s; treatment 515.5 s; makespan reduction -4.30%; throughput increase -4.12%.
- **H2_migration**: not_exercised_no_migrations
  - Makespan reduction median +0.26%; range [-0.0118, 0.0212]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': 0.08304217824307392, 'baseline_terminal_drain_s': 396.0840561389923, 'treatment_terminal_drain_s': 395.03443670272827}, {'repetition': 2, 'native_throughput_increase': -0.01104777756025277, 'baseline_terminal_drain_s': 391.1001343727112, 'treatment_terminal_drain_s': 392.63214468955994}, {'repetition': 3, 'native_throughput_increase': -0.020003079465639728, 'baseline_terminal_drain_s': 390.11543107032776, 'treatment_terminal_drain_s': 398.77194261550903}, {'repetition': 4, 'native_throughput_increase': -0.011900735003609553, 'baseline_terminal_drain_s': 397.44418382644653, 'treatment_terminal_drain_s': 398.8902380466461}, {'repetition': 5, 'native_throughput_increase': 0.00790891413656003, 'baseline_terminal_drain_s': 405.1335971355438, 'treatment_terminal_drain_s': 400.2944085597992}, {'repetition': 6, 'native_throughput_increase': -0.02216828606617438, 'baseline_terminal_drain_s': 405.50921154022217, 'treatment_terminal_drain_s': 392.4313244819641}].
  - Rep 1: baseline 510.07 s; treatment 508.17 s; makespan reduction +0.37%; throughput increase +0.37%.
  - Rep 2: baseline 507.36 s; treatment 507.12 s; makespan reduction +0.05%; throughput increase +0.05%.
  - Rep 3: baseline 505.61 s; treatment 511.59 s; makespan reduction -1.18%; throughput increase -1.17%.
  - Rep 4: baseline 511.0 s; treatment 510.3 s; makespan reduction +0.14%; throughput increase +0.14%.
  - Rep 5: baseline 523.39 s; treatment 512.32 s; makespan reduction +2.12%; throughput increase +2.16%.
  - Rep 6: baseline 515.5 s; treatment 505.82 s; makespan reduction +1.88%; throughput increase +1.91%.
- **S_vs_H**: not_supported
  - Makespan reduction median -2.00%; range [-0.0248, -0.0105]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': 0.09474257985830192, 'baseline_terminal_drain_s': 369.32444882392883, 'treatment_terminal_drain_s': 404.0343713760376}, {'repetition': 2, 'native_throughput_increase': 0.07440474197898306, 'baseline_terminal_drain_s': 378.8133544921875, 'treatment_terminal_drain_s': 393.55296564102173}, {'repetition': 3, 'native_throughput_increase': 0.08431927388682348, 'baseline_terminal_drain_s': 377.50712418556213, 'treatment_terminal_drain_s': 398.7017116546631}, {'repetition': 4, 'native_throughput_increase': 0.10128860491271552, 'baseline_terminal_drain_s': 375.4804232120514, 'treatment_terminal_drain_s': 407.15079641342163}, {'repetition': 5, 'native_throughput_increase': 0.16127167061441772, 'baseline_terminal_drain_s': 370.55923652648926, 'treatment_terminal_drain_s': 407.844957113266}, {'repetition': 6, 'native_throughput_increase': 0.09976865145299985, 'baseline_terminal_drain_s': 373.540864944458, 'treatment_terminal_drain_s': 398.8396894931793}].
  - Rep 1: baseline 488.87 s; treatment 500.94 s; makespan reduction -2.47%; throughput increase -2.41%.
  - Rep 2: baseline 492.09 s; treatment 500.86 s; makespan reduction -1.78%; throughput increase -1.75%.
  - Rep 3: baseline 494.33 s; treatment 499.55 s; makespan reduction -1.05%; throughput increase -1.04%.
  - Rep 4: baseline 491.74 s; treatment 502.66 s; makespan reduction -2.22%; throughput increase -2.17%.
  - Rep 5: baseline 489.76 s; treatment 501.9 s; makespan reduction -2.48%; throughput increase -2.42%.
  - Rep 6: baseline 494.24 s; treatment 502.32 s; makespan reduction -1.63%; throughput increase -1.61%.
- **M0_vs_H**: not_supported
  - Makespan reduction median -4.11%; range [-0.0687, -0.0228]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': -0.07219811385822905, 'baseline_terminal_drain_s': 369.32444882392883, 'treatment_terminal_drain_s': 396.0840561389923}, {'repetition': 2, 'native_throughput_increase': -0.03347351948394395, 'baseline_terminal_drain_s': 378.8133544921875, 'treatment_terminal_drain_s': 391.1001343727112}, {'repetition': 3, 'native_throughput_increase': -0.020408091396050487, 'baseline_terminal_drain_s': 377.50712418556213, 'treatment_terminal_drain_s': 390.11543107032776}, {'repetition': 4, 'native_throughput_increase': -0.03283851009588501, 'baseline_terminal_drain_s': 375.4804232120514, 'treatment_terminal_drain_s': 397.44418382644653}, {'repetition': 5, 'native_throughput_increase': -0.010204456350751734, 'baseline_terminal_drain_s': 370.55923652648926, 'treatment_terminal_drain_s': 405.1335971355438}, {'repetition': 6, 'native_throughput_increase': -0.006678647130320958, 'baseline_terminal_drain_s': 373.540864944458, 'treatment_terminal_drain_s': 405.50921154022217}].
  - Rep 1: baseline 488.87 s; treatment 510.07 s; makespan reduction -4.34%; throughput increase -4.16%.
  - Rep 2: baseline 492.09 s; treatment 507.36 s; makespan reduction -3.10%; throughput increase -3.01%.
  - Rep 3: baseline 494.33 s; treatment 505.61 s; makespan reduction -2.28%; throughput increase -2.23%.
  - Rep 4: baseline 491.74 s; treatment 511.0 s; makespan reduction -3.92%; throughput increase -3.77%.
  - Rep 5: baseline 489.76 s; treatment 523.39 s; makespan reduction -6.87%; throughput increase -6.43%.
  - Rep 6: baseline 494.24 s; treatment 515.5 s; makespan reduction -4.30%; throughput increase -4.12%.
- **MR_vs_H**: not_supported
  - Makespan reduction median -3.63%; range [-0.0461, -0.0234]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': 0.0048485757450160705, 'baseline_terminal_drain_s': 369.32444882392883, 'treatment_terminal_drain_s': 395.03443670272827}, {'repetition': 2, 'native_throughput_increase': -0.044151489046779346, 'baseline_terminal_drain_s': 378.8133544921875, 'treatment_terminal_drain_s': 392.63214468955994}, {'repetition': 3, 'native_throughput_increase': -0.04000294618775291, 'baseline_terminal_drain_s': 377.50712418556213, 'treatment_terminal_drain_s': 398.77194261550903}, {'repetition': 4, 'native_throughput_increase': -0.044348442692930146, 'baseline_terminal_drain_s': 375.4804232120514, 'treatment_terminal_drain_s': 398.8902380466461}, {'repetition': 5, 'native_throughput_increase': -0.0023762483832800996, 'baseline_terminal_drain_s': 370.55923652648926, 'treatment_terminal_drain_s': 400.2944085597992}, {'repetition': 6, 'native_throughput_increase': -0.028698879036375247, 'baseline_terminal_drain_s': 373.540864944458, 'treatment_terminal_drain_s': 392.4313244819641}].
  - Rep 1: baseline 488.87 s; treatment 508.17 s; makespan reduction -3.95%; throughput increase -3.80%.
  - Rep 2: baseline 492.09 s; treatment 507.12 s; makespan reduction -3.05%; throughput increase -2.96%.
  - Rep 3: baseline 494.33 s; treatment 511.59 s; makespan reduction -3.49%; throughput increase -3.37%.
  - Rep 4: baseline 491.74 s; treatment 510.3 s; makespan reduction -3.77%; throughput increase -3.64%.
  - Rep 5: baseline 489.76 s; treatment 512.32 s; makespan reduction -4.61%; throughput increase -4.40%.
  - Rep 6: baseline 494.24 s; treatment 505.82 s; makespan reduction -2.34%; throughput increase -2.29%.
- **M0_vs_S**: not_supported
  - Makespan reduction median -1.74%; range [-0.0428, -0.0121]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': -0.15249310366473445, 'baseline_terminal_drain_s': 404.0343713760376, 'treatment_terminal_drain_s': 396.0840561389923}, {'repetition': 2, 'native_throughput_increase': -0.10040746959495206, 'baseline_terminal_drain_s': 393.55296564102173, 'treatment_terminal_drain_s': 391.1001343727112}, {'repetition': 3, 'native_throughput_increase': -0.09658351355082984, 'baseline_terminal_drain_s': 398.7017116546631, 'treatment_terminal_drain_s': 390.11543107032776}, {'repetition': 4, 'native_throughput_increase': -0.1217910676731564, 'baseline_terminal_drain_s': 407.15079641342163, 'treatment_terminal_drain_s': 397.44418382644653}, {'repetition': 5, 'native_throughput_increase': -0.14766236988666315, 'baseline_terminal_drain_s': 407.844957113266, 'treatment_terminal_drain_s': 405.1335971355438}, {'repetition': 6, 'native_throughput_increase': -0.09679062814045847, 'baseline_terminal_drain_s': 398.8396894931793, 'treatment_terminal_drain_s': 405.50921154022217}].
  - Rep 1: baseline 500.94 s; treatment 510.07 s; makespan reduction -1.82%; throughput increase -1.79%.
  - Rep 2: baseline 500.86 s; treatment 507.36 s; makespan reduction -1.30%; throughput increase -1.28%.
  - Rep 3: baseline 499.55 s; treatment 505.61 s; makespan reduction -1.21%; throughput increase -1.20%.
  - Rep 4: baseline 502.66 s; treatment 511.0 s; makespan reduction -1.66%; throughput increase -1.63%.
  - Rep 5: baseline 501.9 s; treatment 523.39 s; makespan reduction -4.28%; throughput increase -4.11%.
  - Rep 6: baseline 502.32 s; treatment 515.5 s; makespan reduction -2.62%; throughput increase -2.56%.
- **MR_vs_S**: not_supported
  - Makespan reduction median -1.48%; range [-0.0241, -0.007]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': -0.0821142849170271, 'baseline_terminal_drain_s': 404.0343713760376, 'treatment_terminal_drain_s': 395.03443670272827}, {'repetition': 2, 'native_throughput_increase': -0.11034596776573202, 'baseline_terminal_drain_s': 393.55296564102173, 'treatment_terminal_drain_s': 392.63214468955994}, {'repetition': 3, 'native_throughput_increase': -0.11465462531984161, 'baseline_terminal_drain_s': 398.7017116546631, 'treatment_terminal_drain_s': 398.77194261550903}, {'repetition': 4, 'native_throughput_increase': -0.13224239945458105, 'baseline_terminal_drain_s': 407.15079641342163, 'treatment_terminal_drain_s': 398.8902380466461}, {'repetition': 5, 'native_throughput_increase': -0.14092130475473774, 'baseline_terminal_drain_s': 407.844957113266, 'treatment_terminal_drain_s': 400.2944085597992}, {'repetition': 6, 'native_throughput_increase': -0.11681323187349046, 'baseline_terminal_drain_s': 398.8396894931793, 'treatment_terminal_drain_s': 392.4313244819641}].
  - Rep 1: baseline 500.94 s; treatment 508.17 s; makespan reduction -1.44%; throughput increase -1.42%.
  - Rep 2: baseline 500.86 s; treatment 507.12 s; makespan reduction -1.25%; throughput increase -1.23%.
  - Rep 3: baseline 499.55 s; treatment 511.59 s; makespan reduction -2.41%; throughput increase -2.35%.
  - Rep 4: baseline 502.66 s; treatment 510.3 s; makespan reduction -1.52%; throughput increase -1.50%.
  - Rep 5: baseline 501.9 s; treatment 512.32 s; makespan reduction -2.07%; throughput increase -2.03%.
  - Rep 6: baseline 502.32 s; treatment 505.82 s; makespan reduction -0.70%; throughput increase -0.69%.
- **H3_transport**: not_measured_or_not_qualified
- **H4_system**: not_supported
  - Makespan reduction median -3.63%; range [-0.0461, -0.0234]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': 0.0048485757450160705, 'baseline_terminal_drain_s': 369.32444882392883, 'treatment_terminal_drain_s': 395.03443670272827}, {'repetition': 2, 'native_throughput_increase': -0.044151489046779346, 'baseline_terminal_drain_s': 378.8133544921875, 'treatment_terminal_drain_s': 392.63214468955994}, {'repetition': 3, 'native_throughput_increase': -0.04000294618775291, 'baseline_terminal_drain_s': 377.50712418556213, 'treatment_terminal_drain_s': 398.77194261550903}, {'repetition': 4, 'native_throughput_increase': -0.044348442692930146, 'baseline_terminal_drain_s': 375.4804232120514, 'treatment_terminal_drain_s': 398.8902380466461}, {'repetition': 5, 'native_throughput_increase': -0.0023762483832800996, 'baseline_terminal_drain_s': 370.55923652648926, 'treatment_terminal_drain_s': 400.2944085597992}, {'repetition': 6, 'native_throughput_increase': -0.028698879036375247, 'baseline_terminal_drain_s': 373.540864944458, 'treatment_terminal_drain_s': 392.4313244819641}].
  - Rep 1: baseline 488.87 s; treatment 508.17 s; makespan reduction -3.95%; throughput increase -3.80%.
  - Rep 2: baseline 492.09 s; treatment 507.12 s; makespan reduction -3.05%; throughput increase -2.96%.
  - Rep 3: baseline 494.33 s; treatment 511.59 s; makespan reduction -3.49%; throughput increase -3.37%.
  - Rep 4: baseline 491.74 s; treatment 510.3 s; makespan reduction -3.77%; throughput increase -3.64%.
  - Rep 5: baseline 489.76 s; treatment 512.32 s; makespan reduction -4.61%; throughput increase -4.40%.
  - Rep 6: baseline 494.24 s; treatment 505.82 s; makespan reduction -2.34%; throughput increase -2.29%.

## Workload W2
| Arm | Rep | Makespan s | Useful tokens | Useful tok/s | Useful tok/s/allocated chip | Eligible | Exclusions |
|---|---:|---:|---:|---:|---:|---|---|
| H | 1 | 82.9332 | 129022 | 1555.735 | 48.6167 | True | none |
| H | 2 | 83.6805 | 129022 | 1541.841 | 48.1825 | True | none |
| H | 3 | 81.7026 | 129022 | 1579.166 | 49.3489 | True | none |
| H | 4 | 80.5972 | 129022 | 1600.825 | 50.0258 | True | none |
| H | 5 | 81.6237 | 129022 | 1580.692 | 49.3966 | True | none |
| H | 6 | 79.9025 | 129022 | 1614.743 | 50.4607 | True | none |
| M0 | 1 | 79.3549 | 129022 | 1625.885 | 50.8089 | True | none |
| M0 | 2 | 78.9096 | 129022 | 1635.06 | 51.0956 | True | none |
| M0 | 3 | 77.7864 | 129022 | 1658.669 | 51.8334 | True | none |
| M0 | 4 | 79.5446 | 129022 | 1622.009 | 50.6878 | True | none |
| M0 | 5 | 77.7061 | 129022 | 1660.384 | 51.887 | True | none |
| M0 | 6 | 78.6911 | 129022 | 1639.6 | 51.2375 | True | none |
| MR | 1 | 78.3532 | 129022 | 1646.672 | 51.4585 | True | none |
| MR | 2 | 79.562 | 129022 | 1621.654 | 50.6767 | True | none |
| MR | 3 | 77.6891 | 129022 | 1660.747 | 51.8983 | True | none |
| MR | 4 | 78.4425 | 129022 | 1644.797 | 51.3999 | True | none |
| MR | 5 | 79.1552 | 129022 | 1629.988 | 50.9371 | True | none |
| MR | 6 | 77.1543 | 129022 | 1672.26 | 52.2581 | True | none |
| S | 1 | 48.4067 | 129022 | 2665.374 | 83.2929 | True | none |
| S | 2 | 48.2941 | 129022 | 2671.588 | 83.4871 | True | none |
| S | 3 | 48.588 | 129022 | 2655.431 | 82.9822 | True | none |
| S | 4 | 48.9118 | 129022 | 2637.851 | 82.4328 | True | none |
| S | 5 | 48.501 | 129022 | 2660.193 | 83.131 | True | none |
| S | 6 | 49.7616 | 129022 | 2592.804 | 81.0251 | True | none |

### Per-run accounting and iteration eligibility
| Arm/rep | Capture rows / eligible / control / invalid | Retirement/artifact status | Deep-backlog tok/s | Terminal drain s / fraction | Initialization s / init+serving s | Migrations committed / declined / eligible / fallback | Actual payload / network bytes |
|---|---|---|---|---|---|---|---|
| H/1 | 1579 / 1575 / 4 / 0 | available | 1587.887860505834 | 16.067153215408325 / 0.19373614997838273 | 366.09 / 449.0232 | 0 / 0 / missing / missing | missing / missing |
| H/2 | 1605 / 1601 / 4 / 0 | available | 1617.7115351778566 | 17.92340350151062 / 0.2141885929285982 | 363.37 / 447.0505 | 0 / 0 / missing / missing | missing / missing |
| H/3 | 1527 / 1523 / 4 / 0 | available | 1626.0783482004385 | 16.826637268066406 / 0.2059497170095293 | 364.6 / 446.30260000000004 | 0 / 0 / missing / missing | missing / missing |
| H/4 | 1571 / 1567 / 4 / 0 | available | 1634.7010167808776 | 15.313381671905518 / 0.18999894595202152 | 361.53 / 442.12719999999996 | 0 / 0 / missing / missing | missing / missing |
| H/5 | 1593 / 1589 / 4 / 0 | available | 1641.6595667884992 | 15.174132585525513 / 0.18590339606924836 | 363.45 / 445.0737 | 0 / 0 / missing / missing | missing / missing |
| H/6 | 1600 / 1596 / 4 / 0 | available | 1636.1164311417574 | 14.615211963653564 / 0.1829131263562788 | 365.18 / 445.0825 | 0 / 0 / missing / missing | missing / missing |
| M0/1 | 4522 / 4500 / 22 / 0 | available | 1667.834555182222 | 23.571074962615967 / 0.297033491680985 | 361.58 / 440.93489999999997 | 0 / 0 / missing / missing | missing / missing |
| M0/2 | 4301 / 4279 / 22 / 0 | available | 1686.7814304207982 | 22.398500680923462 / 0.2838500243016774 | 362.58 / 441.4896 | 0 / 0 / missing / missing | missing / missing |
| M0/3 | 4589 / 4569 / 20 / 0 | available | 1776.0646008022231 | 19.907897472381592 / 0.255930152150734 | 364.13 / 441.9164 | 0 / 0 / missing / missing | missing / missing |
| M0/4 | 4631 / 4605 / 26 / 0 | available | 1685.2820950212697 | 24.23516845703125 / 0.30467399416688695 | 363.16 / 442.7046 | 0 / 0 / missing / missing | missing / missing |
| M0/5 | 4065 / 4047 / 18 / 0 | available | 1706.58114635245 | 23.092439889907837 / 0.2971766070700984 | 363.83 / 441.5361 | 0 / 0 / missing / missing | missing / missing |
| M0/6 | 4682 / 4658 / 24 / 0 | available | 1700.3813857046525 | 20.52732825279236 / 0.26085943290164204 | 362.77 / 441.4611 | 0 / 0 / missing / missing | missing / missing |
| MR/1 | 4668 / 4644 / 24 / 0 | available | 1725.9757764265169 | 24.167928218841553 / 0.3084485024477786 | 361.58 / 439.9332 | 0 / 0 / missing / missing | missing / missing |
| MR/2 | 4678 / 4652 / 26 / 0 | available | 1699.5270247253427 | 22.67033338546753 / 0.28493930328513684 | 362.58 / 442.142 | 0 / 0 / missing / missing | missing / missing |
| MR/3 | 4547 / 4529 / 18 / 0 | available | 1751.6403767908184 | 24.468911170959473 / 0.3149592014185414 | 364.13 / 441.8191 | 0 / 0 / missing / missing | missing / missing |
| MR/4 | 4744 / 4720 / 24 / 0 | available | 1688.023683718001 | 22.090837478637695 / 0.28161813212546255 | 363.16 / 441.6025 | 0 / 0 / missing / missing | missing / missing |
| MR/5 | 4574 / 4550 / 24 / 0 | available | 1700.449163400879 | 23.33281707763672 / 0.29477301384431176 | 363.83 / 442.98519999999996 | 0 / 0 / missing / missing | missing / missing |
| MR/6 | 4527 / 4507 / 20 / 0 | available | 1747.378175965402 | 23.076807737350464 / 0.2990995975827773 | 362.77 / 439.9243 | 0 / 0 / missing / missing | missing / missing |
| S/1 | 3375 / 3367 / 8 / 0 | available | 2756.41271395725 | 11.163480281829834 / 0.23061840193986285 | 364.03 / 412.4367 | 0 / 0 / missing / missing | missing / missing |
| S/2 | 3471 / 3463 / 8 / 0 | available | 2766.979110759736 | 10.732319116592407 / 0.2222282171455967 | 363.55 / 411.8441 | 0 / 0 / missing / missing | missing / missing |
| S/3 | 3455 / 3447 / 8 / 0 | available | 2787.081545812511 | 11.18716812133789 / 0.23024565536328997 | 363.64 / 412.228 | 0 / 0 / missing / missing | missing / missing |
| S/4 | 3364 / 3356 / 8 / 0 | available | 2757.4085679403966 | 11.279603481292725 / 0.2306111145405644 | 361.52 / 410.43179999999995 | 0 / 0 / missing / missing | missing / missing |
| S/5 | 3390 / 3382 / 8 / 0 | available | 2769.1475287730805 | 11.481897592544556 / 0.23673533976657468 | 363.24 / 411.741 | 0 / 0 / missing / missing | missing / missing |
| S/6 | 3454 / 3446 / 8 / 0 | available | 2762.3911220778664 | 12.743185997009277 / 0.2560848435049434 | 364.06 / 413.8216 | 0 / 0 / missing / missing | missing / missing |

- H/1 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_H_r1.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_H_r1/W2_H_r1_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/2 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_H_r2.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_H_r2/W2_H_r2_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/3 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_H_r3.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_H_r3/W2_H_r3_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/4 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_H_r4.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_H_r4/W2_H_r4_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/5 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_H_r5.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_H_r5/W2_H_r5_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- H/6 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_H_r6.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_H_r6/W2_H_r6_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- M0/1 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_M0_r1.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_M0_r1/W2_M0_r1_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- M0/2 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_M0_r2.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_M0_r2/W2_M0_r2_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- M0/3 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_M0_r3.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_M0_r3/W2_M0_r3_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- M0/4 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_M0_r4.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_M0_r4/W2_M0_r4_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- M0/5 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_M0_r5.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_M0_r5/W2_M0_r5_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- M0/6 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_M0_r6.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_M0_r6/W2_M0_r6_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MR/1 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_MR_r1.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_MR_r1/W2_MR_r1_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MR/2 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_MR_r2.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_MR_r2/W2_MR_r2_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MR/3 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_MR_r3.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_MR_r3/W2_MR_r3_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MR/4 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_MR_r4.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_MR_r4/W2_MR_r4_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MR/5 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_MR_r5.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_MR_r5/W2_MR_r5_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- MR/6 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_MR_r6.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_MR_r6/W2_MR_r6_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/1 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_S_r1.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_S_r1/W2_S_r1_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/2 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_S_r2.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_S_r2/W2_S_r2_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/3 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_S_r3.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_S_r3/W2_S_r3_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/4 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_S_r4.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_S_r4/W2_S_r4_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/5 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_S_r5.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_S_r5/W2_S_r5_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.
- S/6 source: `/workspace/results/hetero/campaign-0f5e4c095b0e/W2_S_r6.json`; individual iterations: `/workspace/results/hetero/artifacts/fp32-0f5e4c095b0e-W2_S_r6/W2_S_r6_iterations.csv.gz`; sustained scope: start at ceil(10% of requests) client completions; end when remaining requests equal admitted cap; identical completion-count bounds in every arm.

Payload bytes and network bytes are distinct. No formula-priced transfer or inferred network volume is accepted. Per-instance/phase capture aggregates, raw CSV paths, gateway counters and legacy byte fields are retained in RESULTS.json. Unknown coverage is not zero coverage. Initialization is not borrowed from an unrelated deployment or summed across overlapping worker initializations.

### Paired contrasts
- **H1_instance_mix**: not_supported
  - Makespan reduction median -61.42%; range [-0.6393, -0.5814]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': -0.39492567758919106, 'baseline_terminal_drain_s': 11.163480281829834, 'treatment_terminal_drain_s': 23.571074962615967}, {'repetition': 2, 'native_throughput_increase': -0.39038880927523356, 'baseline_terminal_drain_s': 10.732319116592407, 'treatment_terminal_drain_s': 22.398500680923462}, {'repetition': 3, 'native_throughput_increase': -0.36275111739349875, 'baseline_terminal_drain_s': 11.18716812133789, 'treatment_terminal_drain_s': 19.907897472381592}, {'repetition': 4, 'native_throughput_increase': -0.3888166902012403, 'baseline_terminal_drain_s': 11.279603481292725, 'treatment_terminal_drain_s': 24.23516845703125}, {'repetition': 5, 'native_throughput_increase': -0.3837160611270932, 'baseline_terminal_drain_s': 11.481897592544556, 'treatment_terminal_drain_s': 23.092439889907837}, {'repetition': 6, 'native_throughput_increase': -0.38445306599970963, 'baseline_terminal_drain_s': 12.743185997009277, 'treatment_terminal_drain_s': 20.52732825279236}].
  - Rep 1: baseline 48.41 s; treatment 79.35 s; makespan reduction -63.93%; throughput increase -39.00%.
  - Rep 2: baseline 48.29 s; treatment 78.91 s; makespan reduction -63.39%; throughput increase -38.80%.
  - Rep 3: baseline 48.59 s; treatment 77.79 s; makespan reduction -60.09%; throughput increase -37.54%.
  - Rep 4: baseline 48.91 s; treatment 79.54 s; makespan reduction -62.63%; throughput increase -38.51%.
  - Rep 5: baseline 48.5 s; treatment 77.71 s; makespan reduction -60.22%; throughput increase -37.58%.
  - Rep 6: baseline 49.76 s; treatment 78.69 s; makespan reduction -58.14%; throughput increase -36.76%.
- **H2_migration**: not_exercised_no_migrations
  - Makespan reduction median +0.69%; range [-0.0186, 0.0195]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': 0.03486030497667825, 'baseline_terminal_drain_s': 23.571074962615967, 'treatment_terminal_drain_s': 24.167928218841553}, {'repetition': 2, 'native_throughput_increase': 0.007556162330625726, 'baseline_terminal_drain_s': 22.398500680923462, 'treatment_terminal_drain_s': 22.67033338546753}, {'repetition': 3, 'native_throughput_increase': -0.013751878169506071, 'baseline_terminal_drain_s': 19.907897472381592, 'treatment_terminal_drain_s': 24.468911170959473}, {'repetition': 4, 'native_throughput_increase': 0.0016267832577290253, 'baseline_terminal_drain_s': 24.23516845703125, 'treatment_terminal_drain_s': 22.090837478637695}, {'repetition': 5, 'native_throughput_increase': -0.003593138811287866, 'baseline_terminal_drain_s': 23.092439889907837, 'treatment_terminal_drain_s': 23.33281707763672}, {'repetition': 6, 'native_throughput_increase': 0.027638970089803427, 'baseline_terminal_drain_s': 20.52732825279236, 'treatment_terminal_drain_s': 23.076807737350464}].
  - Rep 1: baseline 79.35 s; treatment 78.35 s; makespan reduction +1.26%; throughput increase +1.28%.
  - Rep 2: baseline 78.91 s; treatment 79.56 s; makespan reduction -0.83%; throughput increase -0.82%.
  - Rep 3: baseline 77.79 s; treatment 77.69 s; makespan reduction +0.13%; throughput increase +0.13%.
  - Rep 4: baseline 79.54 s; treatment 78.44 s; makespan reduction +1.39%; throughput increase +1.40%.
  - Rep 5: baseline 77.71 s; treatment 79.16 s; makespan reduction -1.86%; throughput increase -1.83%.
  - Rep 6: baseline 78.69 s; treatment 77.15 s; makespan reduction +1.95%; throughput increase +1.99%.
- **S_vs_H**: supported
  - Makespan reduction median +40.55%; range [0.3772, 0.4229]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': 0.7358988518743217, 'baseline_terminal_drain_s': 16.067153215408325, 'treatment_terminal_drain_s': 11.163480281829834}, {'repetition': 2, 'native_throughput_increase': 0.7104280031331576, 'baseline_terminal_drain_s': 17.92340350151062, 'treatment_terminal_drain_s': 10.732319116592407}, {'repetition': 3, 'native_throughput_increase': 0.7139897034462952, 'baseline_terminal_drain_s': 16.826637268066406, 'treatment_terminal_drain_s': 11.18716812133789}, {'repetition': 4, 'native_throughput_increase': 0.686796875780014, 'baseline_terminal_drain_s': 15.313381671905518, 'treatment_terminal_drain_s': 11.279603481292725}, {'repetition': 5, 'native_throughput_increase': 0.68679766791737, 'baseline_terminal_drain_s': 15.174132585525513, 'treatment_terminal_drain_s': 11.481897592544556}, {'repetition': 6, 'native_throughput_increase': 0.6883829717119476, 'baseline_terminal_drain_s': 14.615211963653564, 'treatment_terminal_drain_s': 12.743185997009277}].
  - Rep 1: baseline 82.93 s; treatment 48.41 s; makespan reduction +41.63%; throughput increase +71.33%.
  - Rep 2: baseline 83.68 s; treatment 48.29 s; makespan reduction +42.29%; throughput increase +73.27%.
  - Rep 3: baseline 81.7 s; treatment 48.59 s; makespan reduction +40.53%; throughput increase +68.15%.
  - Rep 4: baseline 80.6 s; treatment 48.91 s; makespan reduction +39.31%; throughput increase +64.78%.
  - Rep 5: baseline 81.62 s; treatment 48.5 s; makespan reduction +40.58%; throughput increase +68.29%.
  - Rep 6: baseline 79.9 s; treatment 49.76 s; makespan reduction +37.72%; throughput increase +60.57%.
- **M0_vs_H**: measured_but_below_threshold
  - Makespan reduction median +4.55%; range [0.0131, 0.057]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': 0.05034782157155626, 'baseline_terminal_drain_s': 16.067153215408325, 'treatment_terminal_drain_s': 23.571074962615967}, {'repetition': 2, 'native_throughput_increase': 0.04269605163898871, 'baseline_terminal_drain_s': 17.92340350151062, 'treatment_terminal_drain_s': 22.398500680923462}, {'repetition': 3, 'native_throughput_increase': 0.09223802332020026, 'baseline_terminal_drain_s': 16.826637268066406, 'treatment_terminal_drain_s': 19.907897472381592}, {'repetition': 4, 'native_throughput_increase': 0.03094209749743637, 'baseline_terminal_drain_s': 15.313381671905518, 'treatment_terminal_drain_s': 24.23516845703125}, {'repetition': 5, 'native_throughput_increase': 0.039546310865750334, 'baseline_terminal_drain_s': 15.174132585525513, 'treatment_terminal_drain_s': 23.092439889907837}, {'repetition': 6, 'native_throughput_increase': 0.03927896165558842, 'baseline_terminal_drain_s': 14.615211963653564, 'treatment_terminal_drain_s': 20.52732825279236}].
  - Rep 1: baseline 82.93 s; treatment 79.35 s; makespan reduction +4.31%; throughput increase +4.51%.
  - Rep 2: baseline 83.68 s; treatment 78.91 s; makespan reduction +5.70%; throughput increase +6.05%.
  - Rep 3: baseline 81.7 s; treatment 77.79 s; makespan reduction +4.79%; throughput increase +5.03%.
  - Rep 4: baseline 80.6 s; treatment 79.54 s; makespan reduction +1.31%; throughput increase +1.32%.
  - Rep 5: baseline 81.62 s; treatment 77.71 s; makespan reduction +4.80%; throughput increase +5.04%.
  - Rep 6: baseline 79.9 s; treatment 78.69 s; makespan reduction +1.52%; throughput increase +1.54%.
- **MR_vs_H**: measured_but_below_threshold
  - Makespan reduction median +4.17%; range [0.0267, 0.0552]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': 0.08696326696313039, 'baseline_terminal_drain_s': 16.067153215408325, 'treatment_terminal_drain_s': 24.167928218841553}, {'repetition': 2, 'native_throughput_increase': 0.050574832266675473, 'baseline_terminal_drain_s': 17.92340350151062, 'treatment_terminal_drain_s': 22.67033338546753}, {'repetition': 3, 'native_throughput_increase': 0.07721769909139864, 'baseline_terminal_drain_s': 16.826637268066406, 'treatment_terminal_drain_s': 24.468911170959473}, {'repetition': 4, 'native_throughput_increase': 0.03261921684133329, 'baseline_terminal_drain_s': 15.313381671905518, 'treatment_terminal_drain_s': 22.090837478637695}, {'repetition': 5, 'native_throughput_increase': 0.03581107667004746, 'baseline_terminal_drain_s': 15.174132585525513, 'treatment_terminal_drain_s': 23.33281707763672}, {'repetition': 6, 'native_throughput_increase': 0.06800356179174916, 'baseline_terminal_drain_s': 14.615211963653564, 'treatment_terminal_drain_s': 23.076807737350464}].
  - Rep 1: baseline 82.93 s; treatment 78.35 s; makespan reduction +5.52%; throughput increase +5.85%.
  - Rep 2: baseline 83.68 s; treatment 79.56 s; makespan reduction +4.92%; throughput increase +5.18%.
  - Rep 3: baseline 81.7 s; treatment 77.69 s; makespan reduction +4.91%; throughput increase +5.17%.
  - Rep 4: baseline 80.6 s; treatment 78.44 s; makespan reduction +2.67%; throughput increase +2.75%.
  - Rep 5: baseline 81.62 s; treatment 79.16 s; makespan reduction +3.02%; throughput increase +3.12%.
  - Rep 6: baseline 79.9 s; treatment 77.15 s; makespan reduction +3.44%; throughput increase +3.56%.
- **M0_vs_S**: not_supported
  - Makespan reduction median -61.42%; range [-0.6393, -0.5814]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': -0.39492567758919106, 'baseline_terminal_drain_s': 11.163480281829834, 'treatment_terminal_drain_s': 23.571074962615967}, {'repetition': 2, 'native_throughput_increase': -0.39038880927523356, 'baseline_terminal_drain_s': 10.732319116592407, 'treatment_terminal_drain_s': 22.398500680923462}, {'repetition': 3, 'native_throughput_increase': -0.36275111739349875, 'baseline_terminal_drain_s': 11.18716812133789, 'treatment_terminal_drain_s': 19.907897472381592}, {'repetition': 4, 'native_throughput_increase': -0.3888166902012403, 'baseline_terminal_drain_s': 11.279603481292725, 'treatment_terminal_drain_s': 24.23516845703125}, {'repetition': 5, 'native_throughput_increase': -0.3837160611270932, 'baseline_terminal_drain_s': 11.481897592544556, 'treatment_terminal_drain_s': 23.092439889907837}, {'repetition': 6, 'native_throughput_increase': -0.38445306599970963, 'baseline_terminal_drain_s': 12.743185997009277, 'treatment_terminal_drain_s': 20.52732825279236}].
  - Rep 1: baseline 48.41 s; treatment 79.35 s; makespan reduction -63.93%; throughput increase -39.00%.
  - Rep 2: baseline 48.29 s; treatment 78.91 s; makespan reduction -63.39%; throughput increase -38.80%.
  - Rep 3: baseline 48.59 s; treatment 77.79 s; makespan reduction -60.09%; throughput increase -37.54%.
  - Rep 4: baseline 48.91 s; treatment 79.54 s; makespan reduction -62.63%; throughput increase -38.51%.
  - Rep 5: baseline 48.5 s; treatment 77.71 s; makespan reduction -60.22%; throughput increase -37.58%.
  - Rep 6: baseline 49.76 s; treatment 78.69 s; makespan reduction -58.14%; throughput increase -36.76%.
- **MR_vs_S**: not_supported
  - Makespan reduction median -61.12%; range [-0.6474, -0.5505]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': -0.37383260217639336, 'baseline_terminal_drain_s': 11.163480281829834, 'treatment_terminal_drain_s': 24.167928218841553}, {'repetition': 2, 'native_throughput_increase': -0.3857824881595512, 'baseline_terminal_drain_s': 10.732319116592407, 'treatment_terminal_drain_s': 22.67033338546753}, {'repetition': 3, 'native_throughput_increase': -0.37151448639075724, 'baseline_terminal_drain_s': 11.18716812133789, 'treatment_terminal_drain_s': 24.468911170959473}, {'repetition': 4, 'native_throughput_increase': -0.38782242742545625, 'baseline_terminal_drain_s': 11.279603481292725, 'treatment_terminal_drain_s': 22.090837478637695}, {'repetition': 5, 'native_throughput_increase': -0.3859304548666308, 'baseline_terminal_drain_s': 11.481897592544556, 'treatment_terminal_drain_s': 23.33281707763672}, {'repetition': 6, 'native_throughput_increase': -0.3674399827020053, 'baseline_terminal_drain_s': 12.743185997009277, 'treatment_terminal_drain_s': 23.076807737350464}].
  - Rep 1: baseline 48.41 s; treatment 78.35 s; makespan reduction -61.86%; throughput increase -38.22%.
  - Rep 2: baseline 48.29 s; treatment 79.56 s; makespan reduction -64.74%; throughput increase -39.30%.
  - Rep 3: baseline 48.59 s; treatment 77.69 s; makespan reduction -59.89%; throughput increase -37.46%.
  - Rep 4: baseline 48.91 s; treatment 78.44 s; makespan reduction -60.38%; throughput increase -37.65%.
  - Rep 5: baseline 48.5 s; treatment 79.16 s; makespan reduction -63.20%; throughput increase -38.73%.
  - Rep 6: baseline 49.76 s; treatment 77.15 s; makespan reduction -55.05%; throughput increase -35.50%.
- **H3_transport**: not_measured_or_not_qualified
- **H4_system**: not_supported
  - Makespan reduction median -61.12%; range [-0.6474, -0.5505]; 6 whole-run pairs.
  - Sustained progress: measured_common_interval; paired native throughput and terminal drain: [{'repetition': 1, 'native_throughput_increase': -0.37383260217639336, 'baseline_terminal_drain_s': 11.163480281829834, 'treatment_terminal_drain_s': 24.167928218841553}, {'repetition': 2, 'native_throughput_increase': -0.3857824881595512, 'baseline_terminal_drain_s': 10.732319116592407, 'treatment_terminal_drain_s': 22.67033338546753}, {'repetition': 3, 'native_throughput_increase': -0.37151448639075724, 'baseline_terminal_drain_s': 11.18716812133789, 'treatment_terminal_drain_s': 24.468911170959473}, {'repetition': 4, 'native_throughput_increase': -0.38782242742545625, 'baseline_terminal_drain_s': 11.279603481292725, 'treatment_terminal_drain_s': 22.090837478637695}, {'repetition': 5, 'native_throughput_increase': -0.3859304548666308, 'baseline_terminal_drain_s': 11.481897592544556, 'treatment_terminal_drain_s': 23.33281707763672}, {'repetition': 6, 'native_throughput_increase': -0.3674399827020053, 'baseline_terminal_drain_s': 12.743185997009277, 'treatment_terminal_drain_s': 23.076807737350464}].
  - Rep 1: baseline 48.41 s; treatment 78.35 s; makespan reduction -61.86%; throughput increase -38.22%.
  - Rep 2: baseline 48.29 s; treatment 79.56 s; makespan reduction -64.74%; throughput increase -39.30%.
  - Rep 3: baseline 48.59 s; treatment 77.69 s; makespan reduction -59.89%; throughput increase -37.46%.
  - Rep 4: baseline 48.91 s; treatment 78.44 s; makespan reduction -60.38%; throughput increase -37.65%.
  - Rep 5: baseline 48.5 s; treatment 79.16 s; makespan reduction -63.20%; throughput increase -38.73%.
  - Rep 6: baseline 49.76 s; treatment 77.15 s; makespan reduction -55.05%; throughput increase -35.50%.


## Research questions and matched effects

All conclusions below concern only the selected cohort. Partial pairs describe observations, not completion of the prospective schedule. Effects and every paired repetition are reported above; H4 uses the frozen MR treatment, never a retrospectively chosen winner.

### 1. In which regimes are smaller instances more efficient?

| Region | Pairs / expected | Status | H useful tok/s/chip | S useful tok/s/chip | S-vs-H makespan reduction median [range] |
|---|---|---|---:|---:|---|
| E_SHORT | 6 / 6 | complete | 39.81255 | 56.869699999999995 | 0.2974 [0.2814, 0.3395] |
| E_LONG | 6 / 6 | complete | 15.3202 | 16.047849999999997 | 0.045 [0.044, 0.047] |

Finite-batch prefill-plus-decode efficiency at allocated chip count; not a universal TP crossover or pure device timing. Deep-backlog claims require a measured common interval.

| Run | Scheduled batch median/max | Context tokens max | Free KV blocks min | Preemptions |
|---|---|---:|---:|---:|
| E_SHORT_H_r1 | 48.0 / 89 | 64347 | 15 | 0 |
| E_SHORT_S_r1 | 29.0 / 34 | 25517 | 0 | 0 |
| E_SHORT_H_r2 | 48.0 / 89 | 64347 | 15 | 0 |
| E_SHORT_S_r2 | 30 / 34 | 25517 | 0 | 0 |
| E_SHORT_S_r3 | 32 / 34 | 25517 | 0 | 0 |
| E_SHORT_H_r3 | 48 / 89 | 64347 | 15 | 0 |
| E_SHORT_S_r4 | 31 / 34 | 25517 | 0 | 0 |
| E_SHORT_H_r4 | 40.5 / 89 | 64347 | 15 | 0 |
| E_SHORT_H_r5 | 42.0 / 89 | 64347 | 15 | 0 |
| E_SHORT_S_r5 | 29.0 / 34 | 25517 | 0 | 0 |
| E_SHORT_S_r6 | 29 / 34 | 25517 | 0 | 0 |
| E_SHORT_H_r6 | 48.0 / 89 | 64347 | 15 | 0 |
| E_LONG_H_r1 | 9.0 / 15 | 65055 | 27 | 0 |
| E_LONG_S_r1 | 6.0 / 6 | 26076 | 0 | 0 |
| E_LONG_H_r2 | 9.0 / 15 | 65055 | 27 | 0 |
| E_LONG_S_r2 | 6.0 / 6 | 26076 | 0 | 0 |
| E_LONG_S_r3 | 6.0 / 6 | 26076 | 0 | 0 |
| E_LONG_H_r3 | 9 / 15 | 65055 | 27 | 0 |
| E_LONG_S_r4 | 6.0 / 6 | 26076 | 0 | 0 |
| E_LONG_H_r4 | 9.0 / 15 | 65055 | 27 | 0 |
| E_LONG_H_r5 | 9.0 / 15 | 65055 | 27 | 0 |
| E_LONG_S_r5 | 6.0 / 6 | 26076 | 0 | 0 |
| E_LONG_S_r6 | 6.0 / 6 | 26076 | 0 | 0 |
| E_LONG_H_r6 | 9 / 15 | 65055 | 27 | 0 |

### 2. Does migrating growing requests beat leaving them in place?

| Workload | Contrast | Verdict | Declared-pair coverage |
|---|---|---|---|
| W0 | H1_instance_mix | not_supported | complete; observed [1, 2, 3, 4, 5, 6]; missing [] |
| W0 | H2_migration | not_exercised_no_migrations | complete; observed [1, 2, 3, 4, 5, 6]; missing [] |
| W0 | H4_system | not_supported | complete; observed [1, 2, 3, 4, 5, 6]; missing [] |
| W1 | H1_instance_mix | not_supported | complete; observed [1, 2, 3, 4, 5, 6]; missing [] |
| W1 | H2_migration | not_exercised_no_migrations | complete; observed [1, 2, 3, 4, 5, 6]; missing [] |
| W1 | H4_system | not_supported | complete; observed [1, 2, 3, 4, 5, 6]; missing [] |
| W2 | H1_instance_mix | not_supported | complete; observed [1, 2, 3, 4, 5, 6]; missing [] |
| W2 | H2_migration | not_exercised_no_migrations | complete; observed [1, 2, 3, 4, 5, 6]; missing [] |
| W2 | H4_system | not_supported | complete; observed [1, 2, 3, 4, 5, 6]; missing [] |

### 3. Is rebuilding destination KV better or worse than transferring it?

| Workload | Contrast | Verdict | Declared-pair coverage |
|---|---|---|---|
| W0 | H3_transport | not_exercised_no_transferred_KV | complete; observed [1, 2, 3, 4, 5, 6]; missing [] |
| W1 | H3_transport | not_measured_or_not_qualified | not_declared; observed []; missing [] |
| W2 | H3_transport | not_measured_or_not_qualified | not_declared; observed []; missing [] |

Positive makespan reduction favors the treatment; throughput increase is a different denominator. Migration claims require actual eligible/committed coverage, not merely an enabled arm. Whole-run and terminal-drain gains are not sustained deep-backlog gains without common measured boundaries. The per-run tables and raw iteration paths above preserve this distinction. Optional ICI remains separate: host fallback is never relabelled ICI.

## Selected-cohort matched-history mechanism costs

Artifact: `results/hetero/20260918/permanent_fp32_v2_mechanism_cost.json`. Passed: True; 18/18 retained trials passed; 9/9 matched pairs passed. Errors: []. Scope: prospective direct-engine mechanism diagnostic under fixed real background; not a gateway headline/system comparison.

| Trial | Status / passed | Proposal to source-release ACK s | First destination token clock envelope s | Native prepare s | Replayed positions | Logical payload bytes | Wire bytes |
|---|---|---:|---|---:|---:|---:|---|
| p256-h64-r1-1-recompute | completed / True | 0.1419346397742629 | [0.16895198822021484, 0.2203998565673828] | 0.03556850700988434 | 319 | 0 | None |
| p256-h64-r1-2-kv_host | completed / True | 2.487329063937068 | [2.4582507610321045, 2.5090315341949463] | 1.1491746350075118 | 0 | 1006632960 | None |
| p256-h64-r2-1-kv_host | completed / True | 2.463207295164466 | [2.4351563453674316, 2.485182523727417] | 1.148684100015089 | 0 | 1006632960 | None |
| p256-h64-r2-2-recompute | completed / True | 0.16210756730288267 | [0.19543051719665527, 0.248307466506958] | 0.043144256982486695 | 319 | 0 | None |
| p256-h64-r3-1-recompute | completed / True | 0.16233630664646626 | [0.19766592979431152, 0.24866557121276855] | 0.045366546022705734 | 319 | 0 | None |
| p256-h64-r3-2-kv_host | completed / True | 2.405662381090224 | [2.362823486328125, 2.413557767868042] | 1.0826188549981453 | 0 | 1006632960 | None |
| p1024-h257-r1-1-recompute | completed / True | 0.1829953258857131 | [0.4429779052734375, 0.4951643943786621] | 0.04516846698243171 | 1280 | 0 | None |
| p1024-h257-r1-2-kv_host | completed / True | 5.174760753288865 | [5.139208793640137, 5.190058946609497] | 2.4957946540089324 | 0 | 2516582400 | None |
| p1024-h257-r2-1-kv_host | completed / True | 4.9268044121563435 | [4.897570371627808, 4.94953179359436] | 2.467181406012969 | 0 | 2516582400 | None |
| p1024-h257-r2-2-recompute | completed / True | 0.18369096610695124 | [0.41639018058776855, 0.46721720695495605] | 0.04051814699778333 | 1280 | 0 | None |
| p1024-h257-r3-1-recompute | completed / True | 0.1800462668761611 | [0.4009664058685303, 0.4517197608947754] | 0.047188936005113646 | 1280 | 0 | None |
| p1024-h257-r3-2-kv_host | completed / True | 4.8953484036028385 | [4.857518911361694, 4.90952205657959] | 2.4902418149868026 | 0 | 2516582400 | None |
| p4096-h257-r1-1-recompute | completed / True | 0.1610908592119813 | [0.7775607109069824, 0.8395864963531494] | 0.035160777013516054 | 4352 | 0 | None |
| p4096-h257-r1-2-kv_host | completed / True | 15.466111924499273 | [15.427010297775269, 15.477535009384155] | 8.626486188994022 | 0 | 8556380160 | None |
| p4096-h257-r2-1-kv_host | completed / True | 15.34946090914309 | [15.314831256866455, 15.367817878723145] | 8.59177895198809 | 0 | 8556380160 | None |
| p4096-h257-r2-2-recompute | completed / True | 0.20349992532283068 | [0.809612512588501, 0.8709433078765869] | 0.043960806011455134 | 4352 | 0 | None |
| p4096-h257-r3-1-recompute | completed / True | 0.18382965587079525 | [0.7855052947998047, 0.8404600620269775] | 0.040887787006795406 | 4352 | 0 | None |
| p4096-h257-r3-2-kv_host | completed / True | 14.768333861604333 | [14.722307920455933, 14.773467302322388] | 8.286077288008528 | 0 | 8556380160 | None |

| Case / repetition | Matched history / background | Pair passed | KV-host minus recompute source-release ACK s |
|---|---|---|---:|
| p256-h64 / 1 | True / True | True | 2.345394424162805 |
| p256-h64 / 2 | True / True | True | 2.3010997278615832 |
| p256-h64 / 3 | True / True | True | 2.2433260744437575 |
| p1024-h257 / 1 | True / True | True | 4.991765427403152 |
| p1024-h257 / 2 | True / True | True | 4.743113446049392 |
| p1024-h257 / 3 | True / True | True | 4.715302136726677 |
| p4096-h257 / 1 | True / True | True | 15.305021065287292 |
| p4096-h257 / 2 | True / True | True | 15.14596098382026 |
| p4096-h257 / 3 | True / True | True | 14.584504205733538 |

These are direct-engine paired diagnostics, not gateway makespan or system speedups. Host clocks, RPC round trips, observer overhead and background overlap remain charged. First-token envelopes assume no unobserved clock step; raw cross-host differences are not pure transport latency. Native export includes drain/gather/host assembly; replay iterations may mix background work. Local control CPU is not remote CPU. Logical array payload excludes serialization and replicated collective/RPC wire traffic; unknown network bytes remain null. Full phase/CPU/RPC clocks, export/import receipts, replay, destination admission, background progress, trial errors and cleanup are retained verbatim in RESULTS.json mechanism_cost.report.

## Gateway overhead in the selected cohort

Admission wait is a single gateway-clock selection/queue interval. Delivery residual spans gateway-to-engine API entry across host clocks (RPC, queueing and clock skew), not isolated ExtProc or native scheduling cost. First-segment TTFT is engine-observed. These overlapping distributions are not summed into makespan; migration coverage is reported separately per run. Missing observations are not zero.

| Workload / arm / rep | Requests | Admission wait median/max s | Delivery residual median [min,max] s | First-segment TTFT median s |
|---|---:|---|---|---:|
| E_LONG/H/1 | 48 | 0.0001 / 17.8799 | 0.419623 [0.003456, 3.665444] | 3.043943 |
| E_LONG/H/2 | 48 | 0.0001 / 17.8966 | 0.420874 [0.003455, 3.666995] | 3.200589 |
| E_LONG/H/3 | 48 | 0.0001 / 18.1602 | 0.415851 [0.003621, 3.637841] | 3.210315 |
| E_LONG/H/4 | 48 | 0.0001 / 17.8966 | 0.425674 [0.003837, 3.64045] | 2.91299 |
| E_LONG/H/5 | 48 | 0.0001 / 17.8885 | 0.415647 [0.003382, 3.655821] | 3.201402 |
| E_LONG/H/6 | 48 | 0.0001 / 18.1625 | 0.422895 [0.003713, 3.644517] | 3.21031 |
| E_LONG/S/1 | 48 | 5.87805 / 11.9821 | 0.572225 [0.003498, 1.599923] | 2.707726 |
| E_LONG/S/2 | 48 | 5.87785 / 11.9617 | 0.395232 [0.003426, 1.609206] | 2.880752 |
| E_LONG/S/3 | 48 | 5.86775 / 11.9769 | 0.404491 [0.003528, 1.618179] | 2.910075 |
| E_LONG/S/4 | 48 | 5.87925 / 11.9663 | 0.408253 [0.003428, 1.619777] | 2.907603 |
| E_LONG/S/5 | 48 | 5.88615 / 11.9729 | 0.574698 [0.003195, 1.600613] | 2.708122 |
| E_LONG/S/6 | 48 | 5.88985 / 11.981 | 0.573514 [0.003554, 1.59689] | 2.708739 |
| E_SHORT/H/1 | 256 | 0.0001 / 37.298 | 3.882636 [0.003451, 14.045251] | 2.420775 |
| E_SHORT/H/2 | 256 | 0.0001 / 38.6012 | 3.919244 [0.002655, 14.128961] | 2.242674 |
| E_SHORT/H/3 | 256 | 0.0001 / 36.9393 | 3.878213 [0.003122, 14.048036] | 2.422576 |
| E_SHORT/H/4 | 256 | 0.0001 / 39.778 | 3.882115 [0.003315, 14.036974] | 2.802252 |
| E_SHORT/H/5 | 256 | 0.0001 / 39.3321 | 3.815387 [0.003376, 14.061823] | 2.301464 |
| E_SHORT/H/6 | 256 | 0.0001 / 37.6209 | 3.862595 [0.003733, 14.028712] | 2.246141 |
| E_SHORT/S/1 | 256 | 0.0002 / 31.4437 | 1.870511 [0.00312, 5.600534] | 1.406245 |
| E_SHORT/S/2 | 256 | 0.0001 / 30.5898 | 1.762661 [0.002813, 5.670097] | 1.262991 |
| E_SHORT/S/3 | 256 | 0.0003 / 22.8613 | 1.770482 [0.003129, 5.646056] | 1.261974 |
| E_SHORT/S/4 | 256 | 0.0002 / 22.5915 | 1.775998 [0.003051, 5.573713] | 1.417505 |
| E_SHORT/S/5 | 256 | 0.0002 / 30.7281 | 1.727185 [0.003014, 5.706772] | 1.262492 |
| E_SHORT/S/6 | 256 | 0.0002 / 31.3729 | 1.784888 [0.003159, 5.611566] | 1.194979 |
| W0/H/1 | 1763 | 0.4515 / 375.8987 | 0.003866 [0.002666, 12.148395] | 0.266979 |
| W0/H/2 | 1763 | 0.6598 / 373.6839 | 0.004151 [0.002797, 11.793779] | 0.264438 |
| W0/H/3 | 1763 | 0.4011 / 375.9683 | 0.003885 [0.002463, 12.419503] | 0.268286 |
| W0/H/4 | 1763 | 0.2002 / 369.4969 | 0.003992 [0.002687, 11.102036] | 0.265839 |
| W0/H/5 | 1763 | 0.4524 / 374.0522 | 0.004018 [0.002836, 12.086874] | 0.270059 |
| W0/H/6 | 1763 | 0.7069 / 379.7404 | 0.004056 [0.00275, 13.397276] | 0.264524 |
| W0/M0/1 | 1763 | 6.5523 / 305.4011 | 0.003829 [0.002634, 9.757357] | 0.238652 |
| W0/M0/2 | 1763 | 6.1188 / 324.298 | 0.003752 [0.002498, 8.915646] | 0.239022 |
| W0/M0/3 | 1763 | 6.2834 / 293.8634 | 0.003965 [0.002616, 5.047641] | 0.232466 |
| W0/M0/4 | 1763 | 6.8178 / 305.9093 | 0.003813 [0.002619, 5.471095] | 0.227278 |
| W0/M0/5 | 1763 | 4.8495 / 328.6915 | 0.003766 [0.002649, 4.460679] | 0.243798 |
| W0/M0/6 | 1763 | 5.0379 / 302.7122 | 0.004044 [0.002745, 4.644493] | 0.242194 |
| W0/MK-H/1 | 1763 | 5.7849 / 322.2953 | 0.003981 [0.002554, 4.425866] | 0.243231 |
| W0/MK-H/2 | 1763 | 6.4352 / 309.2477 | 0.003935 [0.002605, 4.32544] | 0.246087 |
| W0/MK-H/3 | 1763 | 5.0223 / 303.3705 | 0.004497 [0.002776, 5.000842] | 0.239546 |
| W0/MK-H/4 | 1763 | 5.8043 / 279.0283 | 0.003828 [0.002544, 4.713734] | 0.239378 |
| W0/MK-H/5 | 1763 | 6.2697 / 320.5718 | 0.003722 [0.002516, 9.930895] | 0.24215 |
| W0/MK-H/6 | 1763 | 6.6582 / 318.0338 | 0.004096 [0.002665, 6.768634] | 0.242471 |
| W0/MR/1 | 1763 | 6.185 / 313.9405 | 0.003793 [0.002553, 4.844952] | 0.222817 |
| W0/MR/2 | 1763 | 4.5898 / 319.5377 | 0.003726 [0.002418, 4.93768] | 0.241081 |
| W0/MR/3 | 1763 | 6.6017 / 301.1579 | 0.003882 [0.002701, 8.951454] | 0.23682 |
| W0/MR/4 | 1763 | 6.7701 / 312.822 | 0.003787 [0.002587, 8.506675] | 0.239951 |
| W0/MR/5 | 1763 | 5.9828 / 318.5222 | 0.003784 [0.002653, 4.384365] | 0.237738 |
| W0/MR/6 | 1763 | 5.0179 / 296.3143 | 0.003897 [0.002762, 5.205675] | 0.241265 |
| W0/S/1 | 1763 | 0.2221 / 243.8319 | 0.00396 [0.002783, 4.753831] | 0.149614 |
| W0/S/2 | 1763 | 0.3021 / 256.445 | 0.003856 [0.00277, 4.656403] | 0.149553 |
| W0/S/3 | 1763 | 0.4019 / 257.4848 | 0.003859 [0.00266, 4.410932] | 0.150087 |
| W0/S/4 | 1763 | 0.5057 / 257.7269 | 0.003757 [0.002646, 5.786169] | 0.152655 |
| W0/S/5 | 1763 | 0.2522 / 251.404 | 0.004022 [0.002698, 4.481744] | 0.149916 |
| W0/S/6 | 1763 | 0.3024 / 250.5273 | 0.003955 [0.002638, 4.841223] | 0.148647 |
| W1/H/1 | 320 | 165.78925 / 437.3038 | 0.004168 [0.003106, 3.856768] | 0.323378 |
| W1/H/2 | 320 | 155.32665 / 439.9319 | 0.004126 [0.003126, 4.272512] | 0.325048 |
| W1/H/3 | 320 | 159.90935 / 439.8431 | 0.003938 [0.002927, 3.759201] | 0.32497 |
| W1/H/4 | 320 | 154.972 / 441.3432 | 0.004211 [0.003136, 3.583411] | 0.324541 |
| W1/H/5 | 320 | 166.41305 / 441.8806 | 0.003948 [0.002989, 4.011881] | 0.323174 |
| W1/H/6 | 320 | 154.72575 / 440.3554 | 0.004254 [0.002763, 3.892741] | 0.325433 |
| W1/M0/1 | 320 | 183.11935 / 458.6168 | 0.004023 [0.002928, 2.878775] | 0.3255 |
| W1/M0/2 | 320 | 172.7303 / 458.4735 | 0.003985 [0.002942, 2.703624] | 0.326998 |
| W1/M0/3 | 320 | 173.73115 / 453.6644 | 0.004225 [0.002889, 2.938597] | 0.324331 |
| W1/M0/4 | 320 | 173.88665 / 456.6128 | 0.004044 [0.002737, 2.863311] | 0.325154 |
| W1/M0/5 | 320 | 182.9874 / 472.6595 | 0.003857 [0.002737, 2.731395] | 0.326825 |
| W1/M0/6 | 320 | 177.6576 / 461.4355 | 0.004008 [0.002834, 2.586896] | 0.323386 |
| W1/MR/1 | 320 | 162.55135 / 451.7647 | 0.004131 [0.002943, 3.251949] | 0.326683 |
| W1/MR/2 | 320 | 160.74395 / 459.3394 | 0.003809 [0.002644, 2.915778] | 0.324932 |
| W1/MR/3 | 320 | 177.5925 / 454.796 | 0.003958 [0.002841, 2.742619] | 0.328235 |
| W1/MR/4 | 320 | 176.29495 / 458.4907 | 0.004019 [0.002961, 2.492728] | 0.323892 |
| W1/MR/5 | 320 | 174.9792 / 461.099 | 0.003949 [0.002873, 3.006667] | 0.32846 |
| W1/MR/6 | 320 | 168.8947 / 454.5141 | 0.00399 [0.002892, 2.92035] | 0.325699 |
| W1/S/1 | 320 | 153.55245 / 449.8607 | 0.004245 [0.003268, 1.69584] | 0.419693 |
| W1/S/2 | 320 | 153.05655 / 449.1515 | 0.003979 [0.002692, 1.396404] | 0.419022 |
| W1/S/3 | 320 | 153.5065 / 443.0324 | 0.004068 [0.002946, 1.547431] | 0.419645 |
| W1/S/4 | 320 | 145.48755 / 449.5214 | 0.003918 [0.00306, 1.577014] | 0.418695 |
| W1/S/5 | 320 | 151.34875 / 447.2319 | 0.004086 [0.003057, 1.757277] | 0.419599 |
| W1/S/6 | 320 | 155.41105 / 453.0972 | 0.004149 [0.00316, 1.747517] | 0.418924 |
| W2/H/1 | 800 | 0.0001 / 11.2563 | 2.411325 [0.003054, 10.98288] | 1.371942 |
| W2/H/2 | 800 | 0.0001 / 15.1753 | 2.578152 [0.002613, 10.879156] | 1.331828 |
| W2/H/3 | 800 | 0.0001 / 11.8017 | 1.903418 [0.002744, 11.041076] | 1.235099 |
| W2/H/4 | 800 | 0.0001 / 8.6182 | 2.203355 [0.002835, 10.866666] | 1.175033 |
| W2/H/5 | 800 | 0.0001 / 6.1471 | 2.406779 [0.002869, 10.810139] | 1.214866 |
| W2/H/6 | 800 | 0.0001 / 8.4908 | 2.104831 [0.002959, 10.870272] | 1.328258 |
| W2/M0/1 | 800 | 3.6267 / 35.8044 | 0.955157 [0.002717, 9.734809] | 0.685335 |
| W2/M0/2 | 800 | 3.66705 / 33.2375 | 0.880024 [0.00269, 9.522099] | 0.658334 |
| W2/M0/3 | 800 | 2.90455 / 46.277 | 0.932472 [0.002464, 9.367904] | 0.718794 |
| W2/M0/4 | 800 | 3.2065 / 33.7074 | 0.905733 [0.002641, 9.932521] | 0.682349 |
| W2/M0/5 | 800 | 2.3693 / 53.5342 | 0.94068 [0.002921, 9.146826] | 0.616523 |
| W2/M0/6 | 800 | 3.37105 / 30.6186 | 0.977705 [0.002669, 9.921315] | 0.733418 |
| W2/MR/1 | 800 | 3.53765 / 47.4274 | 0.947663 [0.003022, 9.515142] | 0.630278 |
| W2/MR/2 | 800 | 3.72625 / 36.1278 | 0.860089 [0.002831, 9.168498] | 0.663567 |
| W2/MR/3 | 800 | 3.5139 / 42.8712 | 0.928369 [0.002448, 9.279447] | 0.739741 |
| W2/MR/4 | 800 | 3.55925 / 37.0408 | 0.842099 [0.002886, 9.018837] | 0.755711 |
| W2/MR/5 | 800 | 3.85 / 36.756 | 0.918351 [0.002752, 9.649929] | 0.68829 |
| W2/MR/6 | 800 | 3.07305 / 38.0234 | 0.895065 [0.002233, 9.307232] | 0.6164 |
| W2/S/1 | 800 | 0.96225 / 18.0196 | 0.695314 [0.00287, 3.475278] | 0.570816 |
| W2/S/2 | 800 | 1.4414 / 21.6129 | 0.743702 [0.002843, 4.016251] | 0.655012 |
| W2/S/3 | 800 | 1.3436 / 18.361 | 0.73371 [0.002912, 4.218712] | 0.604854 |
| W2/S/4 | 800 | 1.0972 / 25.0446 | 0.68529 [0.002879, 4.046589] | 0.539714 |
| W2/S/5 | 800 | 0.89135 / 29.9087 | 0.707705 [0.002932, 3.860681] | 0.572695 |
| W2/S/6 | 800 | 1.1476 / 19.7871 | 0.722729 [0.002934, 3.32554] | 0.545383 |

## Initialization provenance

Initialization uses only an exact layout/generation-map/settings/runtime match to an all-created-actor deployment or gate.deployment wrapper. The coordinator engine_init_s interval already spans concurrent initialization; worker init times are never summed. Idempotent attach clocks, partially reused deployments, and ambiguous startups are not accepted. Per-instance generation values need not be identical. Initialization-plus-serving is the sum of noncontiguous measured components for a reused generation, not an actual wall interval or amortized initialization.

- E_LONG/H/r1: {"artifacts": [{"file_sha256": "72139912daf367a6e3812bbcb2445ad28aa355ef1b276e39aeea0077e3941de0", "path": "results/hetero/campaign-0f5e4c095b0e/r1_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789708911, "large_b": 1789708912}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_LONG/H/r2: {"artifacts": [{"file_sha256": "e66722fdb19e6af2f0b55481e1b2721069e3ecb732f2feb08d602e8f26d9fa46", "path": "results/hetero/campaign-0f5e4c095b0e/r2_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789714883, "large_b": 1789714883}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_LONG/H/r3: {"artifacts": [{"file_sha256": "3388fe6aae365888b07fbf9d877cef0ea8f5bb7aac3dd90648aa4a89b54e95ed", "path": "results/hetero/campaign-0f5e4c095b0e/r3_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789722290, "large_b": 1789722290}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_LONG/H/r4: {"artifacts": [{"file_sha256": "9b054725f2ef97f765ab2afeaaaec483faa4f46362ab5a7a770bad570e8a5531", "path": "results/hetero/campaign-0f5e4c095b0e/r4_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789731198, "large_b": 1789731198}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_LONG/H/r5: {"artifacts": [{"file_sha256": "540b82c27ee626215b35239477728998f9a785d6efb247791d092fa77238931d", "path": "results/hetero/campaign-0f5e4c095b0e/r5_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789735744, "large_b": 1789735744}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_LONG/H/r6: {"artifacts": [{"file_sha256": "a20000766ec0a21f046c32f13f2f7e05e77cc4cbf230d74658e43eccfdcbfc88", "path": "results/hetero/campaign-0f5e4c095b0e/r6_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789743183, "large_b": 1789743183}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_LONG/S/r1: {"artifacts": [{"file_sha256": "b0025f5814dd2e4a09c9dc85e247135e0b8e6453128d8cecb9cb42e388c0e935", "path": "results/hetero/campaign-0f5e4c095b0e/r1_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789710514, "small_b": 1789710514, "small_c": 1789710514, "small_d": 1789710513}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_LONG/S/r2: {"artifacts": [{"file_sha256": "f747dfd0704b30d3c4d736b078dc5c662bcb28f3a4be94e029f61694168b4bdc", "path": "results/hetero/campaign-0f5e4c095b0e/r2_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789719400, "small_b": 1789719400, "small_c": 1789719400, "small_d": 1789719399}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_LONG/S/r3: {"artifacts": [{"file_sha256": "c6b43f59ad1e0dc6a98d2772cf0caaad1ec0c482ac08aff4535c6d140f4cf660", "path": "results/hetero/campaign-0f5e4c095b0e/r3_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789720848, "small_b": 1789720848, "small_c": 1789720848, "small_d": 1789720847}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_LONG/S/r4: {"artifacts": [{"file_sha256": "fa236753525f2ebf45ef8452774ad9723b72bf4bbee4b18d7b888937e6aef81a", "path": "results/hetero/campaign-0f5e4c095b0e/r4_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789726820, "small_b": 1789726820, "small_c": 1789726820, "small_d": 1789726819}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_LONG/S/r5: {"artifacts": [{"file_sha256": "211960b06de19745686c01b245288bbaf39892e1bd4da4a67b313bffc649df15", "path": "results/hetero/campaign-0f5e4c095b0e/r5_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789737348, "small_b": 1789737348, "small_c": 1789737348, "small_d": 1789737347}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_LONG/S/r6: {"artifacts": [{"file_sha256": "ea0f48d0c5e743638f7387b904c9afec7bbbdc7c8f6ac7c2854db0cccfa9cfbd", "path": "results/hetero/campaign-0f5e4c095b0e/r6_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789741723, "small_b": 1789741723, "small_c": 1789741723, "small_d": 1789741722}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_SHORT/H/r1: {"artifacts": [{"file_sha256": "72139912daf367a6e3812bbcb2445ad28aa355ef1b276e39aeea0077e3941de0", "path": "results/hetero/campaign-0f5e4c095b0e/r1_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789708911, "large_b": 1789708912}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_SHORT/H/r2: {"artifacts": [{"file_sha256": "e66722fdb19e6af2f0b55481e1b2721069e3ecb732f2feb08d602e8f26d9fa46", "path": "results/hetero/campaign-0f5e4c095b0e/r2_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789714883, "large_b": 1789714883}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_SHORT/H/r3: {"artifacts": [{"file_sha256": "3388fe6aae365888b07fbf9d877cef0ea8f5bb7aac3dd90648aa4a89b54e95ed", "path": "results/hetero/campaign-0f5e4c095b0e/r3_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789722290, "large_b": 1789722290}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_SHORT/H/r4: {"artifacts": [{"file_sha256": "9b054725f2ef97f765ab2afeaaaec483faa4f46362ab5a7a770bad570e8a5531", "path": "results/hetero/campaign-0f5e4c095b0e/r4_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789731198, "large_b": 1789731198}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_SHORT/H/r5: {"artifacts": [{"file_sha256": "540b82c27ee626215b35239477728998f9a785d6efb247791d092fa77238931d", "path": "results/hetero/campaign-0f5e4c095b0e/r5_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789735744, "large_b": 1789735744}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_SHORT/H/r6: {"artifacts": [{"file_sha256": "a20000766ec0a21f046c32f13f2f7e05e77cc4cbf230d74658e43eccfdcbfc88", "path": "results/hetero/campaign-0f5e4c095b0e/r6_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789743183, "large_b": 1789743183}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_SHORT/S/r1: {"artifacts": [{"file_sha256": "b0025f5814dd2e4a09c9dc85e247135e0b8e6453128d8cecb9cb42e388c0e935", "path": "results/hetero/campaign-0f5e4c095b0e/r1_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789710514, "small_b": 1789710514, "small_c": 1789710514, "small_d": 1789710513}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_SHORT/S/r2: {"artifacts": [{"file_sha256": "f747dfd0704b30d3c4d736b078dc5c662bcb28f3a4be94e029f61694168b4bdc", "path": "results/hetero/campaign-0f5e4c095b0e/r2_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789719400, "small_b": 1789719400, "small_c": 1789719400, "small_d": 1789719399}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_SHORT/S/r3: {"artifacts": [{"file_sha256": "c6b43f59ad1e0dc6a98d2772cf0caaad1ec0c482ac08aff4535c6d140f4cf660", "path": "results/hetero/campaign-0f5e4c095b0e/r3_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789720848, "small_b": 1789720848, "small_c": 1789720848, "small_d": 1789720847}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_SHORT/S/r4: {"artifacts": [{"file_sha256": "fa236753525f2ebf45ef8452774ad9723b72bf4bbee4b18d7b888937e6aef81a", "path": "results/hetero/campaign-0f5e4c095b0e/r4_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789726820, "small_b": 1789726820, "small_c": 1789726820, "small_d": 1789726819}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_SHORT/S/r5: {"artifacts": [{"file_sha256": "211960b06de19745686c01b245288bbaf39892e1bd4da4a67b313bffc649df15", "path": "results/hetero/campaign-0f5e4c095b0e/r5_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789737348, "small_b": 1789737348, "small_c": 1789737348, "small_d": 1789737347}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- E_SHORT/S/r6: {"artifacts": [{"file_sha256": "ea0f48d0c5e743638f7387b904c9afec7bbbdc7c8f6ac7c2854db0cccfa9cfbd", "path": "results/hetero/campaign-0f5e4c095b0e/r6_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789741723, "small_b": 1789741723, "small_c": 1789741723, "small_d": 1789741722}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/H/r1: {"artifacts": [{"file_sha256": "72139912daf367a6e3812bbcb2445ad28aa355ef1b276e39aeea0077e3941de0", "path": "results/hetero/campaign-0f5e4c095b0e/r1_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789708911, "large_b": 1789708912}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/H/r2: {"artifacts": [{"file_sha256": "e66722fdb19e6af2f0b55481e1b2721069e3ecb732f2feb08d602e8f26d9fa46", "path": "results/hetero/campaign-0f5e4c095b0e/r2_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789714883, "large_b": 1789714883}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/H/r3: {"artifacts": [{"file_sha256": "3388fe6aae365888b07fbf9d877cef0ea8f5bb7aac3dd90648aa4a89b54e95ed", "path": "results/hetero/campaign-0f5e4c095b0e/r3_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789722290, "large_b": 1789722290}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/H/r4: {"artifacts": [{"file_sha256": "9b054725f2ef97f765ab2afeaaaec483faa4f46362ab5a7a770bad570e8a5531", "path": "results/hetero/campaign-0f5e4c095b0e/r4_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789731198, "large_b": 1789731198}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/H/r5: {"artifacts": [{"file_sha256": "540b82c27ee626215b35239477728998f9a785d6efb247791d092fa77238931d", "path": "results/hetero/campaign-0f5e4c095b0e/r5_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789735744, "large_b": 1789735744}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/H/r6: {"artifacts": [{"file_sha256": "a20000766ec0a21f046c32f13f2f7e05e77cc4cbf230d74658e43eccfdcbfc88", "path": "results/hetero/campaign-0f5e4c095b0e/r6_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789743183, "large_b": 1789743183}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/M0/r1: {"artifacts": [{"file_sha256": "d2a0520629ebed2b65b154667c4c67b6b937498b1ec5fc140a41b99a7cef31d8", "path": "results/hetero/campaign-0f5e4c095b0e/r1_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789711969, "small_a": 1789711969, "small_b": 1789711969}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/M0/r2: {"artifacts": [{"file_sha256": "84fcd62bc697dabb6b5df2c74844420157d4336ec46b808c6215231c4da38b7a", "path": "results/hetero/campaign-0f5e4c095b0e/r2_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789716482, "small_a": 1789716482, "small_b": 1789716482}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/M0/r3: {"artifacts": [{"file_sha256": "16c7f14f4b21106f0b318c33371565003cf64f816377b030a787dc2d18a8967b", "path": "results/hetero/campaign-0f5e4c095b0e/r3_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789723894, "small_a": 1789723894, "small_b": 1789723894}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/M0/r4: {"artifacts": [{"file_sha256": "1ad13690c95a9858e6a76d79a43d8fb60a3ec8e6e9cade83913c79ab4cb3e6fa", "path": "results/hetero/campaign-0f5e4c095b0e/r4_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789728271, "small_a": 1789728271, "small_b": 1789728271}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/M0/r5: {"artifacts": [{"file_sha256": "9a915bcfd2aafaa4b61823799017ebf604eb1574831c29358fcef1e90d597dc7", "path": "results/hetero/campaign-0f5e4c095b0e/r5_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789732797, "small_a": 1789732797, "small_b": 1789732797}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/M0/r6: {"artifacts": [{"file_sha256": "c2e0891b07e51a16eb36c95cb08ed1d09538b5a8a6d5e5939c09b7f9c5acc9dc", "path": "results/hetero/campaign-0f5e4c095b0e/r6_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789738801, "small_a": 1789738801, "small_b": 1789738801}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/MK-H/r1: {"artifacts": [{"file_sha256": "d2a0520629ebed2b65b154667c4c67b6b937498b1ec5fc140a41b99a7cef31d8", "path": "results/hetero/campaign-0f5e4c095b0e/r1_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789711969, "small_a": 1789711969, "small_b": 1789711969}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/MK-H/r2: {"artifacts": [{"file_sha256": "84fcd62bc697dabb6b5df2c74844420157d4336ec46b808c6215231c4da38b7a", "path": "results/hetero/campaign-0f5e4c095b0e/r2_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789716482, "small_a": 1789716482, "small_b": 1789716482}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/MK-H/r3: {"artifacts": [{"file_sha256": "16c7f14f4b21106f0b318c33371565003cf64f816377b030a787dc2d18a8967b", "path": "results/hetero/campaign-0f5e4c095b0e/r3_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789723894, "small_a": 1789723894, "small_b": 1789723894}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/MK-H/r4: {"artifacts": [{"file_sha256": "1ad13690c95a9858e6a76d79a43d8fb60a3ec8e6e9cade83913c79ab4cb3e6fa", "path": "results/hetero/campaign-0f5e4c095b0e/r4_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789728271, "small_a": 1789728271, "small_b": 1789728271}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/MK-H/r5: {"artifacts": [{"file_sha256": "9a915bcfd2aafaa4b61823799017ebf604eb1574831c29358fcef1e90d597dc7", "path": "results/hetero/campaign-0f5e4c095b0e/r5_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789732797, "small_a": 1789732797, "small_b": 1789732797}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/MK-H/r6: {"artifacts": [{"file_sha256": "c2e0891b07e51a16eb36c95cb08ed1d09538b5a8a6d5e5939c09b7f9c5acc9dc", "path": "results/hetero/campaign-0f5e4c095b0e/r6_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789738801, "small_a": 1789738801, "small_b": 1789738801}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/MR/r1: {"artifacts": [{"file_sha256": "d2a0520629ebed2b65b154667c4c67b6b937498b1ec5fc140a41b99a7cef31d8", "path": "results/hetero/campaign-0f5e4c095b0e/r1_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789711969, "small_a": 1789711969, "small_b": 1789711969}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/MR/r2: {"artifacts": [{"file_sha256": "84fcd62bc697dabb6b5df2c74844420157d4336ec46b808c6215231c4da38b7a", "path": "results/hetero/campaign-0f5e4c095b0e/r2_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789716482, "small_a": 1789716482, "small_b": 1789716482}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/MR/r3: {"artifacts": [{"file_sha256": "16c7f14f4b21106f0b318c33371565003cf64f816377b030a787dc2d18a8967b", "path": "results/hetero/campaign-0f5e4c095b0e/r3_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789723894, "small_a": 1789723894, "small_b": 1789723894}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/MR/r4: {"artifacts": [{"file_sha256": "1ad13690c95a9858e6a76d79a43d8fb60a3ec8e6e9cade83913c79ab4cb3e6fa", "path": "results/hetero/campaign-0f5e4c095b0e/r4_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789728271, "small_a": 1789728271, "small_b": 1789728271}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/MR/r5: {"artifacts": [{"file_sha256": "9a915bcfd2aafaa4b61823799017ebf604eb1574831c29358fcef1e90d597dc7", "path": "results/hetero/campaign-0f5e4c095b0e/r5_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789732797, "small_a": 1789732797, "small_b": 1789732797}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/MR/r6: {"artifacts": [{"file_sha256": "c2e0891b07e51a16eb36c95cb08ed1d09538b5a8a6d5e5939c09b7f9c5acc9dc", "path": "results/hetero/campaign-0f5e4c095b0e/r6_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789738801, "small_a": 1789738801, "small_b": 1789738801}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/S/r1: {"artifacts": [{"file_sha256": "b0025f5814dd2e4a09c9dc85e247135e0b8e6453128d8cecb9cb42e388c0e935", "path": "results/hetero/campaign-0f5e4c095b0e/r1_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789710514, "small_b": 1789710514, "small_c": 1789710514, "small_d": 1789710513}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/S/r2: {"artifacts": [{"file_sha256": "f747dfd0704b30d3c4d736b078dc5c662bcb28f3a4be94e029f61694168b4bdc", "path": "results/hetero/campaign-0f5e4c095b0e/r2_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789719400, "small_b": 1789719400, "small_c": 1789719400, "small_d": 1789719399}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/S/r3: {"artifacts": [{"file_sha256": "c6b43f59ad1e0dc6a98d2772cf0caaad1ec0c482ac08aff4535c6d140f4cf660", "path": "results/hetero/campaign-0f5e4c095b0e/r3_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789720848, "small_b": 1789720848, "small_c": 1789720848, "small_d": 1789720847}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/S/r4: {"artifacts": [{"file_sha256": "fa236753525f2ebf45ef8452774ad9723b72bf4bbee4b18d7b888937e6aef81a", "path": "results/hetero/campaign-0f5e4c095b0e/r4_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789726820, "small_b": 1789726820, "small_c": 1789726820, "small_d": 1789726819}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/S/r5: {"artifacts": [{"file_sha256": "211960b06de19745686c01b245288bbaf39892e1bd4da4a67b313bffc649df15", "path": "results/hetero/campaign-0f5e4c095b0e/r5_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789737348, "small_b": 1789737348, "small_c": 1789737348, "small_d": 1789737347}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W0/S/r6: {"artifacts": [{"file_sha256": "ea0f48d0c5e743638f7387b904c9afec7bbbdc7c8f6ac7c2854db0cccfa9cfbd", "path": "results/hetero/campaign-0f5e4c095b0e/r6_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789741723, "small_b": 1789741723, "small_c": 1789741723, "small_d": 1789741722}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/H/r1: {"artifacts": [{"file_sha256": "72139912daf367a6e3812bbcb2445ad28aa355ef1b276e39aeea0077e3941de0", "path": "results/hetero/campaign-0f5e4c095b0e/r1_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789708911, "large_b": 1789708912}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/H/r2: {"artifacts": [{"file_sha256": "e66722fdb19e6af2f0b55481e1b2721069e3ecb732f2feb08d602e8f26d9fa46", "path": "results/hetero/campaign-0f5e4c095b0e/r2_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789714883, "large_b": 1789714883}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/H/r3: {"artifacts": [{"file_sha256": "3388fe6aae365888b07fbf9d877cef0ea8f5bb7aac3dd90648aa4a89b54e95ed", "path": "results/hetero/campaign-0f5e4c095b0e/r3_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789722290, "large_b": 1789722290}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/H/r4: {"artifacts": [{"file_sha256": "9b054725f2ef97f765ab2afeaaaec483faa4f46362ab5a7a770bad570e8a5531", "path": "results/hetero/campaign-0f5e4c095b0e/r4_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789731198, "large_b": 1789731198}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/H/r5: {"artifacts": [{"file_sha256": "540b82c27ee626215b35239477728998f9a785d6efb247791d092fa77238931d", "path": "results/hetero/campaign-0f5e4c095b0e/r5_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789735744, "large_b": 1789735744}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/H/r6: {"artifacts": [{"file_sha256": "a20000766ec0a21f046c32f13f2f7e05e77cc4cbf230d74658e43eccfdcbfc88", "path": "results/hetero/campaign-0f5e4c095b0e/r6_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789743183, "large_b": 1789743183}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/M0/r1: {"artifacts": [{"file_sha256": "d2a0520629ebed2b65b154667c4c67b6b937498b1ec5fc140a41b99a7cef31d8", "path": "results/hetero/campaign-0f5e4c095b0e/r1_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789711969, "small_a": 1789711969, "small_b": 1789711969}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/M0/r2: {"artifacts": [{"file_sha256": "84fcd62bc697dabb6b5df2c74844420157d4336ec46b808c6215231c4da38b7a", "path": "results/hetero/campaign-0f5e4c095b0e/r2_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789716482, "small_a": 1789716482, "small_b": 1789716482}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/M0/r3: {"artifacts": [{"file_sha256": "16c7f14f4b21106f0b318c33371565003cf64f816377b030a787dc2d18a8967b", "path": "results/hetero/campaign-0f5e4c095b0e/r3_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789723894, "small_a": 1789723894, "small_b": 1789723894}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/M0/r4: {"artifacts": [{"file_sha256": "1ad13690c95a9858e6a76d79a43d8fb60a3ec8e6e9cade83913c79ab4cb3e6fa", "path": "results/hetero/campaign-0f5e4c095b0e/r4_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789728271, "small_a": 1789728271, "small_b": 1789728271}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/M0/r5: {"artifacts": [{"file_sha256": "9a915bcfd2aafaa4b61823799017ebf604eb1574831c29358fcef1e90d597dc7", "path": "results/hetero/campaign-0f5e4c095b0e/r5_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789732797, "small_a": 1789732797, "small_b": 1789732797}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/M0/r6: {"artifacts": [{"file_sha256": "c2e0891b07e51a16eb36c95cb08ed1d09538b5a8a6d5e5939c09b7f9c5acc9dc", "path": "results/hetero/campaign-0f5e4c095b0e/r6_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789738801, "small_a": 1789738801, "small_b": 1789738801}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/MR/r1: {"artifacts": [{"file_sha256": "d2a0520629ebed2b65b154667c4c67b6b937498b1ec5fc140a41b99a7cef31d8", "path": "results/hetero/campaign-0f5e4c095b0e/r1_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789711969, "small_a": 1789711969, "small_b": 1789711969}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/MR/r2: {"artifacts": [{"file_sha256": "84fcd62bc697dabb6b5df2c74844420157d4336ec46b808c6215231c4da38b7a", "path": "results/hetero/campaign-0f5e4c095b0e/r2_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789716482, "small_a": 1789716482, "small_b": 1789716482}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/MR/r3: {"artifacts": [{"file_sha256": "16c7f14f4b21106f0b318c33371565003cf64f816377b030a787dc2d18a8967b", "path": "results/hetero/campaign-0f5e4c095b0e/r3_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789723894, "small_a": 1789723894, "small_b": 1789723894}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/MR/r4: {"artifacts": [{"file_sha256": "1ad13690c95a9858e6a76d79a43d8fb60a3ec8e6e9cade83913c79ab4cb3e6fa", "path": "results/hetero/campaign-0f5e4c095b0e/r4_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789728271, "small_a": 1789728271, "small_b": 1789728271}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/MR/r5: {"artifacts": [{"file_sha256": "9a915bcfd2aafaa4b61823799017ebf604eb1574831c29358fcef1e90d597dc7", "path": "results/hetero/campaign-0f5e4c095b0e/r5_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789732797, "small_a": 1789732797, "small_b": 1789732797}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/MR/r6: {"artifacts": [{"file_sha256": "c2e0891b07e51a16eb36c95cb08ed1d09538b5a8a6d5e5939c09b7f9c5acc9dc", "path": "results/hetero/campaign-0f5e4c095b0e/r6_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789738801, "small_a": 1789738801, "small_b": 1789738801}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/S/r1: {"artifacts": [{"file_sha256": "b0025f5814dd2e4a09c9dc85e247135e0b8e6453128d8cecb9cb42e388c0e935", "path": "results/hetero/campaign-0f5e4c095b0e/r1_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789710514, "small_b": 1789710514, "small_c": 1789710514, "small_d": 1789710513}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/S/r2: {"artifacts": [{"file_sha256": "f747dfd0704b30d3c4d736b078dc5c662bcb28f3a4be94e029f61694168b4bdc", "path": "results/hetero/campaign-0f5e4c095b0e/r2_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789719400, "small_b": 1789719400, "small_c": 1789719400, "small_d": 1789719399}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/S/r3: {"artifacts": [{"file_sha256": "c6b43f59ad1e0dc6a98d2772cf0caaad1ec0c482ac08aff4535c6d140f4cf660", "path": "results/hetero/campaign-0f5e4c095b0e/r3_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789720848, "small_b": 1789720848, "small_c": 1789720848, "small_d": 1789720847}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/S/r4: {"artifacts": [{"file_sha256": "fa236753525f2ebf45ef8452774ad9723b72bf4bbee4b18d7b888937e6aef81a", "path": "results/hetero/campaign-0f5e4c095b0e/r4_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789726820, "small_b": 1789726820, "small_c": 1789726820, "small_d": 1789726819}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/S/r5: {"artifacts": [{"file_sha256": "211960b06de19745686c01b245288bbaf39892e1bd4da4a67b313bffc649df15", "path": "results/hetero/campaign-0f5e4c095b0e/r5_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789737348, "small_b": 1789737348, "small_c": 1789737348, "small_d": 1789737347}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W1/S/r6: {"artifacts": [{"file_sha256": "ea0f48d0c5e743638f7387b904c9afec7bbbdc7c8f6ac7c2854db0cccfa9cfbd", "path": "results/hetero/campaign-0f5e4c095b0e/r6_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789741723, "small_b": 1789741723, "small_c": 1789741723, "small_d": 1789741722}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/H/r1: {"artifacts": [{"file_sha256": "72139912daf367a6e3812bbcb2445ad28aa355ef1b276e39aeea0077e3941de0", "path": "results/hetero/campaign-0f5e4c095b0e/r1_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789708911, "large_b": 1789708912}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/H/r2: {"artifacts": [{"file_sha256": "e66722fdb19e6af2f0b55481e1b2721069e3ecb732f2feb08d602e8f26d9fa46", "path": "results/hetero/campaign-0f5e4c095b0e/r2_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789714883, "large_b": 1789714883}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/H/r3: {"artifacts": [{"file_sha256": "3388fe6aae365888b07fbf9d877cef0ea8f5bb7aac3dd90648aa4a89b54e95ed", "path": "results/hetero/campaign-0f5e4c095b0e/r3_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789722290, "large_b": 1789722290}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/H/r4: {"artifacts": [{"file_sha256": "9b054725f2ef97f765ab2afeaaaec483faa4f46362ab5a7a770bad570e8a5531", "path": "results/hetero/campaign-0f5e4c095b0e/r4_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789731198, "large_b": 1789731198}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/H/r5: {"artifacts": [{"file_sha256": "540b82c27ee626215b35239477728998f9a785d6efb247791d092fa77238931d", "path": "results/hetero/campaign-0f5e4c095b0e/r5_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789735744, "large_b": 1789735744}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/H/r6: {"artifacts": [{"file_sha256": "a20000766ec0a21f046c32f13f2f7e05e77cc4cbf230d74658e43eccfdcbfc88", "path": "results/hetero/campaign-0f5e4c095b0e/r6_H_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large_a": 1789743183, "large_b": 1789743183}, "layout_id": "H", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/M0/r1: {"artifacts": [{"file_sha256": "d2a0520629ebed2b65b154667c4c67b6b937498b1ec5fc140a41b99a7cef31d8", "path": "results/hetero/campaign-0f5e4c095b0e/r1_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789711969, "small_a": 1789711969, "small_b": 1789711969}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/M0/r2: {"artifacts": [{"file_sha256": "84fcd62bc697dabb6b5df2c74844420157d4336ec46b808c6215231c4da38b7a", "path": "results/hetero/campaign-0f5e4c095b0e/r2_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789716482, "small_a": 1789716482, "small_b": 1789716482}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/M0/r3: {"artifacts": [{"file_sha256": "16c7f14f4b21106f0b318c33371565003cf64f816377b030a787dc2d18a8967b", "path": "results/hetero/campaign-0f5e4c095b0e/r3_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789723894, "small_a": 1789723894, "small_b": 1789723894}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/M0/r4: {"artifacts": [{"file_sha256": "1ad13690c95a9858e6a76d79a43d8fb60a3ec8e6e9cade83913c79ab4cb3e6fa", "path": "results/hetero/campaign-0f5e4c095b0e/r4_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789728271, "small_a": 1789728271, "small_b": 1789728271}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/M0/r5: {"artifacts": [{"file_sha256": "9a915bcfd2aafaa4b61823799017ebf604eb1574831c29358fcef1e90d597dc7", "path": "results/hetero/campaign-0f5e4c095b0e/r5_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789732797, "small_a": 1789732797, "small_b": 1789732797}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/M0/r6: {"artifacts": [{"file_sha256": "c2e0891b07e51a16eb36c95cb08ed1d09538b5a8a6d5e5939c09b7f9c5acc9dc", "path": "results/hetero/campaign-0f5e4c095b0e/r6_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789738801, "small_a": 1789738801, "small_b": 1789738801}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/MR/r1: {"artifacts": [{"file_sha256": "d2a0520629ebed2b65b154667c4c67b6b937498b1ec5fc140a41b99a7cef31d8", "path": "results/hetero/campaign-0f5e4c095b0e/r1_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789711969, "small_a": 1789711969, "small_b": 1789711969}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/MR/r2: {"artifacts": [{"file_sha256": "84fcd62bc697dabb6b5df2c74844420157d4336ec46b808c6215231c4da38b7a", "path": "results/hetero/campaign-0f5e4c095b0e/r2_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789716482, "small_a": 1789716482, "small_b": 1789716482}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/MR/r3: {"artifacts": [{"file_sha256": "16c7f14f4b21106f0b318c33371565003cf64f816377b030a787dc2d18a8967b", "path": "results/hetero/campaign-0f5e4c095b0e/r3_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789723894, "small_a": 1789723894, "small_b": 1789723894}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/MR/r4: {"artifacts": [{"file_sha256": "1ad13690c95a9858e6a76d79a43d8fb60a3ec8e6e9cade83913c79ab4cb3e6fa", "path": "results/hetero/campaign-0f5e4c095b0e/r4_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789728271, "small_a": 1789728271, "small_b": 1789728271}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/MR/r5: {"artifacts": [{"file_sha256": "9a915bcfd2aafaa4b61823799017ebf604eb1574831c29358fcef1e90d597dc7", "path": "results/hetero/campaign-0f5e4c095b0e/r5_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789732797, "small_a": 1789732797, "small_b": 1789732797}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/MR/r6: {"artifacts": [{"file_sha256": "c2e0891b07e51a16eb36c95cb08ed1d09538b5a8a6d5e5939c09b7f9c5acc9dc", "path": "results/hetero/campaign-0f5e4c095b0e/r6_M_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"large": 1789738801, "small_a": 1789738801, "small_b": 1789738801}, "layout_id": "M", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/S/r1: {"artifacts": [{"file_sha256": "b0025f5814dd2e4a09c9dc85e247135e0b8e6453128d8cecb9cb42e388c0e935", "path": "results/hetero/campaign-0f5e4c095b0e/r1_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789710514, "small_b": 1789710514, "small_c": 1789710514, "small_d": 1789710513}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/S/r2: {"artifacts": [{"file_sha256": "f747dfd0704b30d3c4d736b078dc5c662bcb28f3a4be94e029f61694168b4bdc", "path": "results/hetero/campaign-0f5e4c095b0e/r2_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789719400, "small_b": 1789719400, "small_c": 1789719400, "small_d": 1789719399}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/S/r3: {"artifacts": [{"file_sha256": "c6b43f59ad1e0dc6a98d2772cf0caaad1ec0c482ac08aff4535c6d140f4cf660", "path": "results/hetero/campaign-0f5e4c095b0e/r3_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789720848, "small_b": 1789720848, "small_c": 1789720848, "small_d": 1789720847}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/S/r4: {"artifacts": [{"file_sha256": "fa236753525f2ebf45ef8452774ad9723b72bf4bbee4b18d7b888937e6aef81a", "path": "results/hetero/campaign-0f5e4c095b0e/r4_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789726820, "small_b": 1789726820, "small_c": 1789726820, "small_d": 1789726819}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/S/r5: {"artifacts": [{"file_sha256": "211960b06de19745686c01b245288bbaf39892e1bd4da4a67b313bffc649df15", "path": "results/hetero/campaign-0f5e4c095b0e/r5_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789737348, "small_b": 1789737348, "small_c": 1789737348, "small_d": 1789737347}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}
- W2/S/r6: {"artifacts": [{"file_sha256": "ea0f48d0c5e743638f7387b904c9afec7bbbdc7c8f6ac7c2854db0cccfa9cfbd", "path": "results/hetero/campaign-0f5e4c095b0e/r6_S_deploy.json"}], "clock": "deployment.stage_clocks.engine_init_s: one coordinator interval over concurrent engine initialization", "generations": {"small_a": 1789741723, "small_b": 1789741723, "small_c": 1789741723, "small_d": 1789741722}, "layout_id": "S", "scope": "Reused-generation startup plus serving are noncontiguous measured components, not an actual wall interval or amortized initialization. Startup is not summed over workers or charged as repeated initialization.", "status": "matched"}

## Missing observations, failures and exclusions

Missing or excluded scheduled tuples: `[]`.


## Retained history (not selected-cohort conclusions)

All retained run rows, including failed and superseded cohorts, remain in RESULTS.json arms. Original evidence and previous output are not overwritten when --output-dir selects a new report directory.

- `6ffbc8d3fe92a6f2a1f410d7b30a7c3fb2cddddc6c657a61f0cc666a06da7186`: 39 retained runs.
- `8e5b845bae12f65de017faaa1818467c51d816f797dd5f4852f8df54d086f94e`: 2 retained runs.
- `ceff8004d5c412d6e8a1481b90ef650a5d4fd65f1d71533049d108d5a67d3878`: 2 retained runs.
- `unfrozen`: 18 retained runs.

## Goal acceptance audit

Goal complete: **True**. Blocking reason: None

| Acceptance gate | Status | Evidence |
|---|---|---|
| 1. Contiguous subslice allocation and TPU runtime bring-up | passed | results/hetero/20260918/permanent_fp32_v2_gate_H.json, results/hetero/20260918/permanent_fp32_v2_gate_S.json, results/hetero/20260918/permanent_fp32_v2_gate_M.json |
| 2. Repeatable deployment and Ray serving-instance discovery | passed | results/hetero/20260918/permanent_fp32_v2_lifecycle_M.json |
| 3. Continuation and ownership qualified with recompute | passed | results/hetero/20260918/permanent_fp32_v2_continuation.json, results/hetero/20260918/permanent_fp32_v2_lifecycle_M.json, results/hetero/20260918/permanent_fp32_v2_recompute.json |
| 4. Host-staged KV transfer qualified; ICI optional | passed | results/hetero/20260918/permanent_fp32_v2_continuation.json, results/hetero/20260918/permanent_fp32_v2_kv_host.json |
| 5. Predeclared research comparisons | passed | results/hetero/evaluation_fp32_state_20260918_v2_source_frozen.json, results/hetero/20260918/permanent_fp32_v2_mechanism_cost.json |
| 6. Conclude and restore owned state | passed | results/hetero/20260918/permanent_fp32_final_inspect_H.json, results/hetero/20260918/permanent_fp32_final_inspect_S.json, results/hetero/20260918/permanent_fp32_final_inspect_M.json |

- **1. Contiguous subslice allocation and TPU runtime bring-up**: All three layouts require hash-bound raw gates, native precision, concurrent serving and frozen source checks.
- **2. Repeatable deployment and Ray serving-instance discovery**: Full-fabric ExtProc probes, scoped restart, peer preservation and fresh-generation readmission must pass.
- **3. Continuation and ownership qualified with recompute**: Unchanged numerical criteria and actual gateway recompute handoffs; earlier diagnostics do not authorize this cohort.
- **4. Host-staged KV transfer qualified; ICI optional**: Actual released host-KV migrations, positive payload bytes, exact output accounting and ownership ordering; no ICI claim.
- **5. Predeclared research comparisons**: 102/102 exact scheduled tuples eligible. Every scheduled contrast and the matched 18-trial mechanism-cost group are required; a positive speedup is not required.
- **6. Conclude and restore owned state**: Residual owned state per layout: {'H': {'registered_instances': 0, 'live_engine_actors': 0, 'created_owned_placement_groups': 0, 'pending_owned_placement_groups': 0}, 'S': {'registered_instances': 0, 'live_engine_actors': 0, 'created_owned_placement_groups': 0, 'pending_owned_placement_groups': 0}, 'M': {'registered_instances': 0, 'live_engine_actors': 0, 'created_owned_placement_groups': 0, 'pending_owned_placement_groups': 0}}.
