# TPUFlex case studies: experiment design (2026-10-25)

Scope and priorities, as set by the user:

| priority | case study | status |
|---|---|---|
| P0 | Autoscaling: two framings, a trace-driven burst experiment | design below |
| P1 | Concurrent scale-out: GCS vs peer contention | design below |
| P0 | Failure recovery: process crash with a hot-standby gate, and host/slice preemption | design below |
| P1 | Submesh reconfiguration: online trace, more real points, P/D claim | design below |
| – | TorchTPU compatibility: paper text only, no new experiments | bullets in §5 |

Sources:

- **Harness facts**: code reading of `~/hetero-research/hetero/`.
- **Platform facts**: official GKE, Ray and GCS documentation.
- **Methodology**: OSDI/SOSP/ASPLOS/EuroSys papers (links in §6).
- **Numbers**: committed results; see `results/matrix-2x2/PAPER_REPORT.md` for provenance.

Tags:

- **[I]**: inference, not a measurement.
- **[U]**: unverified.

## 0. Facts that change the design

1. **On GKE the failure unit of a multi-host TPU slice is the whole slice.** "If a machine in the group fails or terminates, GKE recreates the entire set of nodes as a new unit" ([About TPUs in GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/tpus)). Consequences:
   - **Same-slice host replacement is impossible.** "Replace one host of a multi-host instance and pull from the surviving host" cannot happen inside one slice: the surviving hosts are recreated too.
   - **State on the slice is gone.** Every host's `/dev/shm` snapshot and Executable Bank on that slice is lost. Recovery state must come from **another slice** (a peer instance) or from **GCS**.
   - **A shared slice fails as a whole.** Our shared 4x8 slice runs four TP8 instances; losing one host kills all four. Experiments that inject host loss need instances on separate slices (dedicated 2x4 node pools, as in the e2e harness), never on the shared 4x8.
   - **Spot notice is short.** TPU spot preemption gives at most 30 s of notice; the 120 s extension does not apply to TPU spot ([TPU Spot](https://cloud.google.com/tpu/docs/spot)).
   - **Ray Serve recreates one replica.** KubeRay deletes the whole multi-host replica when Ray deletes one worker. Atomic recreation after a single pod failure is an open issue ([kuberay#3552](https://github.com/ray-project/kuberay/issues/3552)).
2. **The client is closed-loop today.** `hetero/experiment.py:208-255` dispatches a backlog with a concurrency cap (`ADMITTED_CAP`). There are no arrival times. Responses are non-streaming, but every response carries the engine's `ttft_s`, `started_at_s` and `finished_at_s`, and the router reports its queue wait (`EPP_QUEUE_HEADER`, `experiment.py:157-167`). From these, an open-loop client can compute:
   - TTFT = router queue + engine TTFT;
   - TPOT = (finished − started − TTFT) / (tokens − 1).
   No new engine instrumentation is needed.
3. **Missing harness pieces** (none exist in `hetero/`, `scripts/` or `gateway/`):
   - an autoscaling controller;
   - a fault injector;
   - prefill/decode disaggregation;
   - an open-loop trace client.
4. **Gateway behavior to account for:**
   - health lease 45 s (`gateway/core.py:34`);
   - 429-and-poll admission with a 3600 s timeout (`experiment.py:189-205`, `core.py:36`);
   - the ledger defaults to `/tmp`, which "survives process loss only, not machine replacement" (`gateway/state.py:13-16`), so node loss currently needs an operator step.
   - A 45 s lease would dominate a 19 s recovery, so failure detection must be event-driven (§2).
5. **The shared slice is preemptible right now.** After the 2026-10-24 GKE auto-upgrade, the shared KubeRay workers run on the spot pool `as-v5e-4x8-spot`, so a preemption recreates all 8 hosts. Before long campaigns, move the workers back to the on-demand `tpu-v5e-4x8-d500` pool (pending user decision; needs a nodeSelector on the shared RayCluster).
6. **A citation correction.** The figures "peer 50.2 s vs GCS 53.1 s" mislabel one number. At TP16:
   - peer = 50.2 s (n=2);
   - GCS = 52.2 s (n=3, the paper's number);
   - 53.1 s is **GCS FUSE** delivery (n=1).
   At TP8 (matrix, n=3): pm-peer 51.7 s vs pm-gcs 53.8 s.

## 1. Autoscaling (P0)

### 1.1 Two framings for the paper

| framing | when it applies | what to report | numbers |
|---|---|---|---|
| **New capacity** (GKE allocates new VMs) | no idle capacity; spot recovery without a buffer | end-to-end from the scale-up trigger, honestly | TPU-JAX 580.6 → 264.7 s (**2.19×**); TorchTPU 386.1 → 305.7 s (1.26×). GKE + Ray join ≈ 170–176 s in every arm |
| **Standby capacity** (nodes allocated and joined, engine not running) | warm pools, reservations, overprovisioning buffers, idle pairs of an allocated slice | fresh-host engine time | TPU-JAX 362.6 → 53.8 s (**6.74×**); TorchTPU 3.22×; H100 2.0–2.9× |
| **Hot standby** (standby node with state pre-staged in `/dev/shm` and workers pre-spawned at a gate; CPU and DRAM only, TPU idle) | same as standby, plus a few GB of host RAM per standby host | gate → ready | TPU-JAX **18.9 s** (warm-pm from gate, n=3); TorchTPU 23.9 s. Not yet measured on a *standby* host; the same mechanism measured as a same-host restart |

**Why standby is common in production** (primary sources):

| source | what is kept warm | reason |
|---|---|---|
| [GKE spare capacity](https://cloud.google.com/kubernetes-engine/docs/how-to/capacity-provisioning) | running nodes held by low-priority balloon pods | node boot takes 80–120 s; meant for bursts |
| [GKE capacity buffers](https://cloud.google.com/kubernetes-engine/docs/concepts/capacity-buffer) (Preview) | active (running) and standby (suspended) nodes | cheaper than overprovisioning (TPU support [U]) |
| [Compute Engine reservations](https://cloud.google.com/compute/docs/instances/reservations-overview) | reserved capacity, **billed whether used or not** | accelerator capacity assurance and disaster recovery |
| [AWS EC2 warm pools](https://docs.aws.amazon.com/autoscaling/ec2/userguide/ec2-auto-scaling-warm-pools.html), [Azure standby pools](https://learn.microsoft.com/en-us/azure/virtual-machine-scale-sets/standby-pools-overview) | pre-initialized instances | apps with long boot times |
| [Modal](https://modal.com/blog/truly-serverless-gpus) | "a small buffer of healthy, idle GPUs" with no engine | GPU acquisition can take tens of minutes; closest analogue |
| [Aegaeon §2.2](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf) | production reserved GPU instances per model (17.7% of GPUs serve 1.35% of requests) | reservation is the production default |
| [DeepServe §6.1](https://arxiv.org/abs/2501.14417) | pre-warmed pods and engines, no weights | engine startup dominates |

TPU-specific argument [I]: TPU capacity is scarce, so it is typically reserved or held. A held slice that is under-used between peaks is standby capacity, and so is any idle pair of an allocated 4x8.

### 1.2 Trace-driven burst experiment

**Question.** Given the same reactive scaling policy, how much does boot time change user-visible latency during a burst?

**Setup.**

- **Instances and gateway.** TP8 instances, each on its own new 2x4 v5e slice: dedicated on-demand node pools in a private namespace with a private Ray head, as in the e2e harness. The user approved new capacity on 2026-10-25, and the shared 4x8 slice is not used. Behind one router. The base fleet is 1 instance; standby slices are pre-provisioned, joined and idle, up to the peak count the chosen window needs. Quota on 2026-10-25: on-demand v5e podslice 44/128 chips used, so 84 chips (10 slices of 2x4) are free.
- **Client.** An open-loop trace client (new, §4). Each request is sent at its trace timestamp, with no concurrency cap.
- **Controller (new, §4).** The policy is identical in every arm:
  - trigger a scale-up when the router's waiting-request count, or the instance's KV usage, stays above a threshold for 10 s;
  - one standby pair per trigger;
  - no scale-down within a run.

**Workload.**

- **Calibration first.** For one TP8 instance, measure the maximum rate that meets the SLO, using the chosen trace's length distribution. The SLO follows BlitzScale: TTFT and TPOT each at most 5× their unloaded medians. Write both down before any arm runs.
- **Trace choice (researched 2026-10-25; Azure 2023 is too old).** Candidates, statistics computed from the public files:

  | trace | collected | what it is | burstiness | lengths (in p50 / out p50) | use |
  |---|---|---|---|---|---|
  | **Chutes "A Year in LLM Serving"** ([arXiv 2608.13573](https://arxiv.org/abs/2608.13573), [data](https://github.com/HarvardMadSys/chutes_workload), CC BY 4.0) | one year, about mid-2025 to mid-2026; 6.12B requests, 9,174 models | per request: model, **instance_id**, start/finish, input/output/cached tokens, TTFT | `unsloth/gemma-3-27b-it`, last 67 days: mean 1.96 rps, daily 1-min peak/mean median 5.4 (max 33); **85 burst episodes** where the 5-min rate reaches ≥ 4× the trailing hour (median 8.5×, median 7 min, 25% ≥ 12 min, some ≥ 2 h) | 1902 / 131 (day 330); 7.3% of inputs > 8k | **primary**: same model family and size, recent, real bursts, and real production instance counts |
  | Qwen-Bailian traces ([repo](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon); KVCache ATC'25, LMetric OSDI'26) | Dec 2024 (to-B); 2 h each | cluster-level to-C chat (A), to-B API (B), thinking, coder; multi-turn ids | A: 5.98 rps, inter-arrival CV 1.06, 1-min peak/mean 1.36. B: 24 rps, CV 1.04 | A 1046 / 375; B 574 / 39 | steady background load (failure-recovery runs), non-bursty control |
  | ServeGen ([NSDI'26](https://www.usenix.org/conference/nsdi26/presentation/xiang-servegen), [repo](https://github.com/alibaba/ServeGen)) | Alibaba Model Studio, 12 models, 2025 | generator from per-client rate/CV and length distributions | bursty clients, shifting length mix | per model | simulator sweeps (rate shifts, CV) |
  | not chosen | | Azure 2023/2024 (May 2024 or older), BurstGPT (2023 GPT-3.5/4 data; v2.0 only re-released it in Jan 2026), FreeInference agent traces (contexts around 50k, above our 8192 limit), Azure LMM 2025 (multimodal, Oct 2024) | | | |

- **Trace A (realistic): Chutes gemma-3-27b burst windows.** Windows found in the last 67 days, minute index from trace start:

  | window | base | peak | duration ≥ 2× base | Chutes instances |
  |---|---|---|---|---|
  | 464487 | 21 rpm | 157 rpm (7.4×) | 38 min | 2 → 3 after 4 min |
  | 453710 | 23 rpm | 302 rpm (13×) | 68 min | 2 → 4 after 12 min |
  | 473753 | 22 rpm | 260 rpm (11.7×) | 124 min | 2 → 3 after 26 min |
  | 440318 (scale-from-zero) | 0 rpm | 436 rpm | 124 min | 0 → 6 over about 36 min |

  - **Window choice.** Replay 40–60 min around one window; pick by peak/base so the peak needs at most 3–4 TP8 instances.
  - **Rescaling.** Multiply timestamps by one constant (Arrow) so the pre-burst rate is about 50% of one instance's capacity. Uniform rescaling keeps the peak/base ratio.
  - **Lengths.** Use the trace's own input/output lengths. Inputs over 8192 − output are truncated (report the fraction). Prompts are synthetic token arrays of the exact length (the client already sends token arrays); outputs are forced with `max_tokens` + `ignore_eos`.
  - **Extraction.** DuckDB over HTTP range reads of the 91 GB parquet, which is sorted by `started_at` (6114 row groups of about 1M rows). One model-day takes about 5 s, so there is no download. A local extract of the last 67 days of gemma-3-27b sits in `~/.cache/pm-render/traces/` (`chutes_gemma3_27b_d300plus.parquet`, `gemma3_per_min.parquet`, `gemma3_bursts.csv`).
  - **Motivation data point (from the trace, cause unknown).** On Chutes, the instance count for this model rose 4–26 min after burst onset.
- **Trace B (controlled).** A step burst from 0.5× to 1.6× one instance's capacity, held for 10 min. This makes recovery-time metrics clean.

**Arms** (n=3 each, interleaved, same trace and seed):

1. **Static base** (1 instance, no scaling): lower bound on capacity.
2. **Static peak** (the peak instance count always on): upper bound, i.e. zero boot time.
3. **Stock scale-up**: best stock fresh-host boot on the standby pair (runai-cc, about 363 s).
4. **TPUFlex scale-up from GCS** (about 54 s).
5. **TPUFlex scale-up from a peer** (about 52 s; the peer is the serving instance, so also record the peer's throughput dip while it streams state).
6. **TPUFlex hot standby**: state pre-staged and workers parked; open the gate (about 19 s + detection).
7. **New-capacity framing, real** (the user approved new instances on 2026-10-25): arms 3 and 4 where the trigger scales a GKE 2x4 node pool from 0 and the instance boots on the new VMs. Recreate the pool between runs so every run gets fresh VMs, as the e2e harness does (`scripts/tpu_e2e.py`).

**Metrics** (timeline + aggregates):

- **Timeline**: request rate, instances serving, router queue depth, per-10-s-window TTFT P50/P99 and TPOT P99.
- **Capacity-recovery time**: trigger → first request served by the new instance, and trigger → queue depth back below threshold.
- **Burst-window aggregates**: TTFT P99, TPOT P99, SLO-violation %, max and integrated queue depth, timeouts/429s, and instance-seconds (cost).

The literature reads recovery time off a throughput timeline (BlitzScale Fig 21, λScale Fig 14); we report it as a named metric.

**Scale.**

- 6 arms × 3 runs × about 30 min ≈ 9 h of slice time.
- Uses 2–3 pairs of the shared slice (never pair c).
- It is the only experiment that turns boot latency into user-visible numbers.

**Breadth.**

- Calibrate the existing boot-time → headroom analysis (Oct 22, Google Doc tab "Autoscaling: boot time & GKE") as a trace simulator using measured per-instance capacity and boot-time distributions.
- Validate it against the 3 real arms (report MAPE, as Splitwise does).
- Sweep traces and burst sizes in simulation only.

### 1.3 Concurrent scale-out: bandwidth contention (P1)

Booting N TP8 instances at once (N ∈ {1, 2, 3}) on separate pairs:

| arm | source |
|---|---|
| GCS | all N pull the same PM prefix |
| one peer, fan-out 1→N | one serving instance streams to all N |
| peer per target | each target pulls from a different live instance |
| *(optional)* chain/tree relay | target 1 forwards to target 2 |

- **Metrics**: per-host GB/s, delivery wall, gate wait (does delivery still hide behind spawn?), ready time, and the source peers' serving throughput while they stream.
- **Expectation** [I]:
  - GCS scales: the default regional egress is 200 Gbps per project, and there is no per-object read limit.
  - Peer fan-out divides the source NIC (2×100 Gbps per host is documented for 8-chip v5e hosts; [U] for our 4-chip hosts).
  - At N=1 the delivery is fully hidden, gate wait 0 (state 12–21 s vs spawn about 33 s), so contention matters only once delivery exceeds spawn.
- **Scale**: 3 pairs, n=3, about 2 h.
- **Prior data**: the matrix's three pairs already ran concurrently at times; check their overlap windows first.

## 2. Failure recovery (P0)

### 2.1 Process crash with a hot-standby gate

**Mechanism to build** (Decompose applied to failure):

- Each serving instance keeps a standby set of CPU-only pre-spawned workers parked at the gate on the same hosts.
- On crash, the supervisor opens the gate. The standby restores from the intact `/dev/shm` snapshot and bank, then spawns a new standby in the background.

**Requirements:**

1. **Event-driven detection, target ≤ 1 s.** Use Ray actor-death notification or a pid watcher. The 45 s gateway lease is too slow.
2. **Device release.** The dead process's TPU chips must be released before the standby opens them; reuse the VFIO reset barrier, which the paper measures at 0.84–0.86 s in full TPUFlex (`tab:segments`).
3. **Immediate gateway rebinding.** Withdraw the dead endpoint at once, and bind the new one on health.
4. **In-flight requests.** Re-dispatch them from their token history (re-prefill). The continuation path exists (`hetero/inproc_engine.py:246-247` PREEMPTED, `qualify_continuation.py`); the paper already argues re-prefill beats KV migration.
5. **Memory footprint.** Measure the standby's host-RAM footprint and its effect on foreground throughput. The paper's foreground-throughput section has a template.

**Experiment** (TPU-JAX, TP8, shared slice, steady open-loop load at about 60–70% of fleet capacity on 2–3 instances):

| arm | recovery path | expected (s) |
|---|---|---|
| stock | supervisor restarts stock vLLM with local caches (warm-stock) | about 350 |
| TPUFlex restart | relaunch with the TPUFlex restore (warm-pm from launch) | about 52 |
| TPUFlex hot standby | open the parked standby's gate | about 19 + detection |

- **Injection**: `kill -9` of the engine process (and, separately, of one worker). 5 injections per arm at randomized times.
- **Metrics:**
  - detection latency;
  - recovery time: kill → first token from the new engine, and kill → throughput back to the pre-fault level;
  - requests in flight at the kill: lost vs re-dispatched, plus the extra latency of re-dispatched ones;
  - failures/timeouts;
  - TTFT P99 in the fault window;
  - the **other instances' throughput curves**: they should absorb rerouted load without dipping;
  - availability (% of time at full capacity).
- **Variant (container restart, `/dev/shm` survives)** [I]: the 2026-10-24 Ray head OOM restart wiped `/root` but kept `/dev/shm`, which suggests a container restart within the same pod keeps the snapshot. Measure this as the "pod-level" tier.

### 2.2 Host preemption: slice loss

By §0.1 a host failure removes the whole slice. Run instances on **dedicated 2x4 node pools** (e2e harness), each with its own private Ray head. Then:

**Injection:**

- Delete one node VM of the instance's slice (Compute Engine documents stop/delete as the way to simulate spot preemption). GKE should recreate the slice ([I], to verify).
- Optionally try `simulate-maintenance-event` ([U] on GKE TPU nodes).
- Separately, exercise the 30 s spot notice: drain admission and hand in-flight requests to other instances within the grace period.

**Recovery arms:**

| arm | where it boots | state source | expected |
|---|---|---|---|
| no standby, stock | recreated slice (wait for GKE, about 170 s) | GCS checkpoint + cache | ≈ e2e stock (581 s) |
| no standby, TPUFlex | recreated slice | GCS | ≈ e2e TPUFlex (265 s) |
| standby slice, stock | idle standby slice | GCS | about 363 s + detection |
| standby slice, TPUFlex, GCS | idle standby slice | GCS | about 54 s + detection |
| standby slice, TPUFlex, peer | idle standby slice | a surviving instance on another slice (cross-slice DCN) | about 52 s + detection; measure the peer's throughput dip |
| hot-standby slice | standby slice with state pre-staged, workers parked | local | about 19 s + detection |

**Metrics**: as in §2.1, plus the replica-count timeline. In flight, the spot-grace handling reports lost vs handed-off requests.

**State-source decision** (the user's open question):

| source | when available | cost | interference |
|---|---|---|---|
| local `/dev/shm` | process crash, container restart | none | none |
| peer instance (same layout, same per-rank state; canonical device order is required on TorchTPU) | any healthy instance of the same layout | 51.7 s at TP8 | uses the peer's NIC while it serves: measure |
| GCS | always | 53.8 s at TP8; scales with N (§1.3) | none on serving instances |

Recommendation [I]:

- Default to GCS; it is always available and does not touch serving instances.
- Use a peer only if §1.3 shows GCS degrading under concurrent recoveries.
- Pre-stage state on standby slices (hot standby) whenever DRAM allows.

**Required fixes:**

- **Ledger durability.** Make the gateway ledger survive node loss (GCS or etcd instead of `/tmp`), or document the manual step.
- **Replica recreation.** Handle KubeRay's non-atomic replica recreation.

### 2.3 What not to claim

Reviewers know GKE's slice semantics, so the paper should not describe "replacing one host of a multi-host instance and restoring from its surviving hosts".

## 3. Submesh reconfiguration (P1)

### 3.1 Online trace with unknown phase boundaries

- **Workload**: an open-loop stream whose mix shifts without announcement, built from trace segments:
  - Azure conversation (short), then a long-context segment (GovReport W6 or Mooncake conversation, input median 6.9k), then back to short;
  - boundaries at random times;
  - plus a gradual-mix variant.
- **Planner (new)**:
  - estimate the per-layout rates τ online from observed throughput, and the mix from a rolling window of prompt lengths and queue composition;
  - apply the paper's break-even rule with hysteresis;
  - choose full vs partial switches.
- **Arms**:
  - static S, H, M;
  - online TPUFlex switching;
  - offline oracle (knows the boundaries);
  - vanilla-cost switching (the same planner with C = 419 s / 186 s, executed with stock redeploy; or emulated by blocking for the measured cost, labelled).
- **Metrics:**
  - TTFT and TPOT P50/P99, SLO attainment, goodput;
  - the switch timeline (when the planner decided vs when the phase changed);
  - per-switch disruption: requests delayed by the drain, and the partial switch's effect on instances that keep serving.
  - Precedent: SpotServe Fig 8g/h and DynamoLLM Fig 10 show latency timelines with switch points marked.

### 3.2 More real points for the makespan model

The current backing is thin:

- the draft says the model reproduces "all 10 end-to-end multi-phase runs within max |Δk| ≤ 0.009";
- `results/reconfig/e2e.json` holds 12 job records. Checked on 2026-10-25:
  - static M on W4+W0×4 (2) and reconfigure M→S (2), job walls 1752/1757 vs 1966/1957 s;
  - partial switch on mixA (2);
  - static S on W0×4 (1);
  - static M on W4+W5 (2) and reconfigure M→H (2), 1772/1819 vs 2244/2223 s;
  - one partial smoke.
  None is a W0→W6 or W0→W7 job, and in each pair the reconfigured job is *slower* than static. These runs validate the cost model at older switch costs, not TPUFlex's 27.5/18.8 s;
- `results/reconfig/breakeven_measured.json` (40 rows) uses costs of 395 s and 440 s only.
- **So the paper's break-even table (`tab:breakeven`) and coverage figure (`fig:regions`) at C = 27.5/18.8 s have no real run behind them.** The k* values come from the model alone.

Plan:

- **Grid**: about 20 points stratified on the measured-cost phase diagram: both sides of the k_s* = 0.78 and k_m* = 1.38 boundaries, one or two points deep inside each region (static H, full, partial, static M), and a few at the vanilla-cost boundary.
- **Runs**: at each point run the planner's choice and the best static alternative, n=2.
- **Report**: measured vs model makespan (MAPE, max error) and whether the measured winner matches the predicted one.
- **Cost** [I]: each run is a few manifests of 190–410 s, so about 10–20 min per run; 20 × 2 × 2 ≈ 13 h. Do boundary points first.

### 3.3 Zero-DCN P/D rebalancing claim

There is no P/D disaggregation in the harness or in our TPU stack, and building TPU P/D with KV transfer is a project of its own. Recommendation: **narrow the claim to what can be measured** and drop the P/D-deployment framing unless P/D exists. Two cheap measurements:

1. **NIC bytes during a reconfiguration.** Sample host NIC counters across S↔M and S↔H switches: expect ≈ 0 for TPUFlex, against about 62.5 GB per pair pulled from GCS for a stock redeploy.
2. **Contention test.** During a partial switch, run a synthetic DCN stream at 30 Gbps per host between hosts of the untouched instances (iperf or a JAX cross-host transfer). Show that the stream's throughput and the switch cost are both unchanged, whereas a stock redeploy's GCS pull competes on the NIC.

## 4. Harness work needed (shared by the P0 items)

| piece | basis | used by |
|---|---|---|
| open-loop trace client: arrival timestamps, per-request records (send, router queue, engine TTFT/start/finish, tokens, status, retries) | `experiment._post_epp` + `_dispatch_http` minus the semaphore | 1.2, 2.1, 2.2, 3.1 |
| trace loader + rescaler (Azure 2023/2024, BurstGPT, Mooncake), length capping, prompt synthesis by length | `hetero/manifest.py` | 1.2, 3.1 |
| metrics timeline sampler (router queue, instances, per-instance tokens/s) | gateway/EPP metrics | all |
| autoscaling controller (threshold + window, standby-pair registry, calls deploy / gateway-up) | `hetero_ctl.py deploy`, gateway join | 1.2 |
| hot-standby mode (parked workers per serving instance, gate on a crash event, standby respawn) | warm-pm pre-spawn + gate | 1.2 arm 6, 2.1, 2.2 |
| event-driven failure detector + immediate endpoint withdrawal + re-dispatch of in-flight requests | Ray actor death, continuation path | 2.1, 2.2 |
| fault injector (process kill, worker kill, pod delete, node VM delete) with timestamps | kubectl / gcloud | 2.1, 2.2 |
| online planner | break-even model | 3.1 |
| durable gateway ledger | `gateway/state.py` | 2.2 |

Rough order [I]:

1. client, trace loader, metrics and calibration (≈ 1 day);
2. controller and the §1.2 burst runs (≈ 1–2 days);
3. hot standby, detector and fault injector, then §2.1 (≈ 2 days);
4. dedicated pools and §2.2 (≈ 1–2 days);
5. §1.3;
6. §3.

**Each cluster campaign follows the repo rules**: `rcm_exec` only, leases, never pair c, and approval before mutating runs.

## 5. TorchTPU compatibility paragraph (no new experiments)

Points for the paper agent; all numbers are in PAPER_REPORT.md:

- **All three principles carry over** to vLLM-TorchTPU 0.29.0:
  - fresh host 145.0 → 45.1 s (3.22×);
  - same-host restart 126.6 → 23.9 s from gate (5.29×).
- **The bottleneck moves from weights to compile.** On TorchTPU the stock loader takes 6 s, while compile takes 67.9 s even with the stack's own cache. The dominant mechanism becomes the sampler bank (+48.0 s when removed, the whole config A → B step), not the snapshot (+9.0 s). What Cache must target depends on the stack.
- **Device order matters.** Banked programs carry their device assignment. Chip enumeration differs across VMs, and a bank loaded on another slice produced silently wrong text (±65536 token shifts). This justifies the device-order-preserving bank: key on the rank→device map, use canonical binding, refuse on mismatch.

## 6. Methodology references

| scenario | references | what to follow |
|---|---|---|
| bursty autoscaling | BlitzScale OSDI'25 §6; λScale; ServerlessLLM OSDI'24 §7; Llumnix OSDI'24 §6 | real trace rescaled to ~50% mean load; timeline + CDFs + SLO-violation % (5× unloaded); same policy for every system |
| preemption/failure | SpotServe ASPLOS'24 §6; SkyServe EuroSys'25 §5; NVIDIA Dynamo shadow engine (kill process: 7.3 s vs 283 s) | client retries counted in latency; failure rate; replica timeline |
| online reconfiguration | SpotServe Fig 8g/h; DynamoLLM HPCA'25 Fig 10; Llumnix Fig 10 | latency timeline with switch points; disruption per switch |
| P/D | Splitwise ISCA'24; DistServe OSDI'24; Arrow | SLO attainment vs rate; TTFT and TPOT separately |

Full table with links: research note from agent MethodLitScout (2026-10-25), summarized in §1–§3.
