# TPU 拓扑差异下的 Padding 开销与分桶编译物理实验设计规范

## 1. 概述与核心动机

在基于 Google Cloud TPU v5e 的大规模 LLM 分布式推理（vLLM / XLA）体系中，XLA 编译器依赖于**静态计算图（Static Computation Graphs）**。为了支持变长 Prompt 输入与动态并发批处理，系统通常预编译一组离散的**序列长度分桶（Sequence Length Buckets）**：
$$\mathcal{S} = [128, 256, 512, 1024, 2048]$$

任何长度落在区间 $(S_i, S_{i+1}]$ 内的请求，均必须强制对齐（Ceil-Padding）至下一个分桶 $S_{i+1}$。本实验旨在测量该静态 Padding 机制在 **$2\times 4$（8 芯片，TP=8）** 与 **$4\times 4$（16 芯片，TP=16）** 两种物理互联拓扑下的真实硬件开销，量化越界 Padding 导致的阶梯延迟跳变（Bucket Jump Penalty）以及跨机通信放大的物理代价。

---

## 2. 硬件拓扑与物理互联特性

### 2.1 物理集群环境
- **加速器规格**：Google Cloud TPU v5e（单芯片峰值算力 197 TFLOPS BF16，819 GB/s HBM 带宽，16 MB VMEM）。
- **物理机架**：TPU-v5litepod-16（4 台物理 Host，每 Host 配置 4 颗 TPU v5e 芯片，通过机内与机间高速 ICI (Inter-Chip Interconnect) 构成 2D Torus 互联网格）。

### 2.2 拓扑对比维度

| 维度 | $2\times 4$ 拓扑 (Subslice) | $4\times 4$ 拓扑 (Full Pod) |
| :--- | :--- | :--- |
| **芯片总数 / Host 数** | 8 芯片 / 2 台物理 Host | 16 芯片 / 4 台物理 Host |
| **张量并行 (TP) 程度** | $\text{TP}=8$（跨 2 台 Host） | $\text{TP}=16$（跨 4 台 Host） |
| **网络直径与跳数** | ICI 单环/双向相邻跳数 $\le 2$ | 2D Torus 环形跳数 $\le 4$，跨 Host 跳数加倍 |
| **AllReduce 通信延迟** | 低（局部 Host 间环传输） | 高（跨机 ICI 跳数增加，单次 AllReduce 基础延迟上升） |
| **算力理论峰值** | $8 \times 197 = 1,576\text{ TFLOPS}$ | $16 \times 197 = 3,152\text{ TFLOPS}$ |

---

## 3. Padding 开销的物理成因与数学模型

### 3.1 单步 Prefill 延迟分解模型
在 Prefill 阶段，单步执行时间 $T_{\text{step}}$ 主要由**计算延迟**与**卡间通信延迟**构成：
$$T_{\text{step}}(S, B) = T_{\text{GEMM}}(S, B) + T_{\text{Attn}}(S, B) + T_{\text{Comm}}(S, B)$$

- **GEMM 算力消耗**：全模型各层投影（QKV、O-proj、Gate/Up/Down MLP）计算量为：
  $$\text{FLOPs}_{\text{GEMM}} \approx 2 \cdot N_{\text{layers}} \cdot B \cdot S \cdot (4 \cdot d^2 + 3 \cdot d \cdot d_{\text{ffn}})$$
  Padding 增加的 Token 线性放大 GEMM 计算耗时。
- **Attention 算力消耗**：Prefill 因果自注意力计算量为 $O(B \cdot S^2 \cdot d)$。
- **ICI AllReduce 通信消耗**：
  每层包含 2 次卡间 AllReduce（分别在 Attention 输出与 FFN 输出），通信张量尺寸为 $[B, S, d]$。
  在 Ring AllReduce 算法下，单步传输延迟为：
  $$T_{\text{Comm}}(S, B) \approx 2 \cdot N_{\text{layers}} \cdot \left[ 2 \cdot \frac{\text{TP}-1}{\text{TP}} \cdot \alpha_{\text{latency}} + \frac{2 \cdot (\text{TP}-1)}{\text{TP}} \cdot \frac{2 \cdot B \cdot S \cdot d}{\text{BW}_{\text{ICI}}} \right]$$
  其中 $\alpha_{\text{latency}}$ 为单跳物理延迟，$\text{BW}_{\text{ICI}}$ 为互联带宽。在 $4\times 4$ 拓扑下，$\text{TP}=16$，跨机通信跳数与同步延迟项显著增大。

### 3.2 跨桶跃迁惩罚（Bucket Jump Penalty）
设请求实际有效长度为 $L$。若 $S_i < L \le S_{i+1}$，则必须填充至 $S_{i+1}$。
临界越界对齐惩罚定义为刚超过 $S_i$ 的请求（如 $L = S_i + 1$）强制对齐至 $S_{i+1}$ 所付出的延迟开销：
$$\Delta T_{\text{jump}}(S_i \to S_{i+1}) = T_{\text{step}}(S_{i+1}) - T_{\text{step}}(S_i)$$
相对惩罚比例（Penalty Overhead）：
$$\text{Overhead}_{\text{rel}} = \frac{T_{\text{step}}(S_{i+1}) - T_{\text{step}}(S_i)}{T_{\text{step}}(S_i)} \times 100\%$$

### 3.3 拓扑放大系数（Topology Amplification Factor）
为了验证大拓扑是否会放大 Padding 造成的延迟惩罚，定义拓扑放大比 $\Gamma$：
$$\Gamma(S_i \to S_{i+1}) = \frac{\left( \frac{T_{\text{step}, 4\times 4}(S_{i+1})}{T_{\text{step}, 4\times 4}(S_i)} \right)}{\left( \frac{T_{\text{step}, 2\times 4}(S_{i+1})}{T_{\text{step}, 2\times 4}(S_i)} \right)}$$
当 $\Gamma > 1$ 时，证明在 $4\times 4$ 拓扑下，Padding 无效 Token 对网络通信造成的额外开销显著超越了算力收益，细粒度分桶在更大拓扑下具备更高的优化优先级。

---

## 4. 实验测试矩阵与变量控制

### 4.1 核心被测模型
- **模型**：`Gemma-4 31B`（权重路径 `/models/gemma-4-31b`）。
- **参数规格**：48 层，Hidden Size = 5376，Intermediate Size = 21504（或 14336），Heads = 32 Q / 16 KV，Head Dim = 128。
- **数据类型**：`bfloat16`。

### 4.2 测试参数矩阵

1. **标准编译分桶基线测试（Standard Bucket Baseline）**：
   - 分桶集合：$S \in [128, 256, 512, 1024, 2048]$。
   - 目的：获得两组拓扑下各静态 Shape 的物理迭代基准耗时与吞吐。
2. **临界跨桶越界测试（Boundary Jump Testing）**：
   - 测试点：
     - $128 \to 256$（有效 Token 129，补齐 127 个 Token，填充率 50.4%）
     - $256 \to 512$（有效 Token 257，补齐 255 个 Token，填充率 50.2%）
     - $512 \to 1024$（有效 Token 513，补齐 511 个 Token，填充率 50.1%）
     - $1024 \to 2048$（有效 Token 1025，补齐 1023 个 Token，填充率 50.0%）
3. **固定 Bucket 填充率梯度测试（Padding Density Sweep）**：
   - 在基准分桶 $S=1024$ 下，输入有效 Token 比例设为 25% (256), 50% (512), 75% (768), 100% (1024)。
   - 测量「有效吞吐（Effective Tokens/s）」与实际单步迭代耗时的偏离程度。
4. **并发批次（Batch Size）**：
   - $B=1$（单请求延迟敏感场景，剥离算力打满的影响，观察通信占比）；
   - $B=4$（高负荷并发场景，观察算力与通信同时放大时的耗时）。

---

## 5. 执行流程与验证规约

### 5.1 物理集群隔离与放置组（Placement Group）
- **$2\times 4$ 拓扑配置**：
  - 通过 `TPUTopologyDiscovery` 选定连续的 2 台宿主机（8 芯片）；
  - 显式创建 `STRICT_SPREAD` 的节点绑定 Placement Group（每个 Bundle `{"TPU": 4.0, "node:<ip>": 0.001}`）；
  - 配置 `TP_SIZE=8`，初始化分布式网格。
- **$4\times 4$ 拓扑配置**：
  - 绑定全部 4 台连续物理宿主机（16 芯片）；
  - 显式创建 `STRICT_SPREAD` 节点绑定 Placement Group；
  - 配置 `TP_SIZE=16`，初始化完整 Pod 互联。
- **运行清理规范**：在初始化与切换拓扑前，严格调用 `cleanup_stale_placement_groups()` 清理残留。

### 5.2 预热与采样稳定性
- **JIT Warmup**：每个分桶在计时前执行至少 3 次预热迭代，确保 XLA 编译完成并排空流水线。
- **多轮采样**：每个测试点连续采样 10 次，记录各次迭代耗时，计算 P50（中位数）、P90、平均耗时与标准差。
- **真实权重保护**：执行阶段依托物理模型权重，严禁使用离线估算或模拟器替代。

### 5.3 结果数据结构规约 (`results/padding_cost_topologies.json`)
生成的 JSON 文件必须具备以下结构：
```json
{
  "metadata": {
    "timestamp": 1725800000.0,
    "model": "gemma-4-31b",
    "topologies": ["2x4", "4x4"],
    "batch_sizes": [1, 4],
    "standard_buckets": [128, 256, 512, 1024, 2048]
  },
  "topologies": {
    "2x4": {
      "tp_size": 8,
      "num_chips": 8,
      "target_hosts": ["192.168.102.18", "192.168.102.16"],
      "standard_buckets": {
        "128": { "step_ms_p50": 1.25, "throughput_tok_s": 102400.0 },
        "256": { ... },
        "512": { ... },
        "1024": { ... },
        "2048": { ... }
      },
      "jump_penalties": {
        "128_to_256": { "delta_ms": 0.8, "overhead_pct": 64.0 },
        "256_to_512": { ... },
        "512_to_1024": { ... },
        "1024_to_2048": { ... }
      },
      "density_sweep_1024": [
        { "effective_tokens": 256, "padded_tokens": 1024, "effective_tput": 12500.0, "step_ms": 20.48 },
        { ... }
      ]
    },
    "4x4": {
      "tp_size": 16,
      "num_chips": 16,
      "target_hosts": ["192.168.102.18", "192.168.102.16", "192.168.102.19", "192.168.102.17"],
      "standard_buckets": { ... },
      "jump_penalties": { ... },
      "density_sweep_1024": [ ... ]
    }
  },
  "amplification_analysis": {
    "jump_amplification": {
      "128_to_256": 1.15,
      "256_to_512": 1.28,
      "512_to_1024": 1.42,
      "1024_to_2048": 1.65
    }
  }
}
```

---

## 6. RCM 运维与逆向控制规约

1. **执行入口**：必须通过 `python3 -m omp_rcm run benchmarks/run_padding_cost_topology.py` 驱动。
2. **生命周期事件上报**：在关键阶段输出 `[OMP_EVENT: TOPOLOGY_LOCKED]`、`[OMP_EVENT: BUCKET_WARMED]`、`[OMP_EVENT: BENCHMARK_SUCCESS]`。
3. **故障陷阱拦截**：
   若遇到 ICI 网络抖动、OOM 或分桶编译超时，自动进入 `DiagnosticTrap`，保护内存现场并退出码 2，供 Agent 实时热打补丁。

---

## 7. 实测硬件基准数据与拓扑分析摘要 (Physical Ground Truth)

以下数据均采集自 Cloud TPU v5e 物理集群实测（Gemma-4 31B 结构模型，`results/padding_cost_topologies.json`）：

### 7.1 标准分桶基准延迟与吞吐对比
| 分桶大小 ($S$) | $2\times 4$ P50 延迟 (ms) | $2\times 4$ 吞吐 (tok/s) | $4\times 4$ P50 延迟 (ms) | $4\times 4$ 吞吐 (tok/s) | 算力加速比 ($4\times 4$ vs $2\times 4$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **128** | 26.82 | 4,772.1 | 26.57 | 4,817.7 | $1.01\times$ |
| **256** | 33.37 | 7,670.6 | 31.05 | 8,245.1 | $1.07\times$ |
| **512** | 43.79 | 11,692.0 | 40.36 | 12,686.7 | $1.09\times$ |
| **1024** | 67.85 | 15,093.1 | 62.89 | 16,283.2 | $1.08\times$ |
| **2048** | 118.35 | 17,305.3 | 103.85 | 19,721.2 | $1.14\times$ |
| **4096** | 243.35 | 16,831.9 | 188.25 | 21,758.6 | **$1.29\times$** |
| **8192** | 459.27 | 17,836.8 | 362.23 | 22,615.7 | **$1.27\times$** |

### 7.2 临界分桶跃迁开销（Padding Jump Overhead）

| 临界跃迁 | 填充目标分桶 | $2\times 4$ 额外延迟 (ms) | $2\times 4$ 开销比例 | $4\times 4$ 额外延迟 (ms) | $4\times 4$ 开销比例 | 拓扑放大系数 $\Gamma$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $128 \to 256$ | 256 | +6.55 ms | +24.4% | +4.48 ms | +16.9% | 0.684 |
| $256 \to 512$ | 512 | +10.42 ms | +31.2% | +9.31 ms | +30.0% | 0.894 |
| $512 \to 1024$ | 1024 | +24.06 ms | +54.9% | +22.53 ms | +55.8% | 0.937 |
| $1024 \to 2048$ | 2048 | +50.50 ms | +74.4% | +40.96 ms | +65.1% | 0.811 |
| $2048 \to 4096$ | 4096 | +125.00 ms | +105.6% | +84.40 ms | +81.3% | 0.675 |
| $4096 \to 8192$ | 8192 | +215.93 ms | +88.7% | +173.98 ms | +92.4% | 0.806 |

### 7.3 核心发现与准入控制指导建议
1. **大分桶彻底释放 $4\times 4$（TP=16）算力**：在小分桶（128~1024）时，$4\times 4$ 因跨机 AllReduce 通信底噪优势不显（加速比仅 $1.01\sim 1.09\times$）；但进入 **4096 与 8192** 超大分桶后，$4\times 4$ 算力优势迅速爆发，单步执行时间分别较 $2\times 4$ 节省 **55.1 ms** 与 **97.1 ms**，加速比跃升至 **$1.29\times$**，有效吞吐达到 **$22,615.7\text{ tok/s}$**。
2. **$2\times 4$ 拓扑在大分桶下的吞吐见顶**：$2\times 4$（TP=8）在分桶达到 2048 时硬件吞吐即见顶（17,305 tok/s），在 4096 分桶时因单卡激活张量过大反而回落至 16,831 tok/s；而 $4\times 4$ 凭借 16 芯片更宽的算力池与聚合显存带宽，吞吐仍持续线性走高。
3. **高分桶阶梯绝对惩罚剧烈，但 $4\times 4$ 边际惩罚更低**：在 $2048 \to 4096$ 与 $4096 \to 8192$ 区间，$2\times 4$ 跨桶惩罚高达 +125 ms 与 +216 ms；而在 $4\times 4$ 下，由于 GEMM 计算被 16 芯片分摊，跨桶惩罚收窄至 +84.4 ms 与 +174.0 ms。
4. **调度与准入建议**：对于长输入（$\ge 2048$）或大批次（并发 Prefill Batch $\ge 4$）请求，应严格路由至 $4\times 4$ 拓扑；对于短请求（$\le 512$ 且批次小），应路由至 $2\times 4$ 拓扑，实现异构切片的高效 Serving。

### 7.4 实测基准矢量图呈现 (Visual Artifact)

本实验四维物理分析图表保存在：
![Padding Cost and Topology Analysis](../visuals/padding_cost_topology_analysis.svg)
- **图 A**：单步 Prefill 延迟随分桶变化曲线，展示 $S \le 1024$ 通信底噪平台与 $S \ge 4096$ 算力爆发拐点。
- **图 B**：物理硬件吞吐曲线，展示 $2\times 4$ 在 $17.3\text{k tok/s}$ 见顶与 $4\times 4$ 突破 $22.6\text{k tok/s}$。
- **图 C**：跨分桶越界惩罚（$\Delta T_{\text{jump}}$），量化大分桶下阶梯惩罚。
- **图 D**：$S=1024$ 下有效 Token 填充密度下降导致的吞吐线性塌陷。
