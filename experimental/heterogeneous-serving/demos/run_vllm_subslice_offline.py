"""
Run vLLM Offline Inference on a Selected 2-Host TPU Subslice (TP=8)
===================================================================
Uses TPUTopologyDiscovery to guarantee that the two chosen TPU hosts
form a physically connected ICI sub-mesh before launching vLLM.
"""

import os
import sys

# Ensure required environment variables for multi-host TPU Ray executor
os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
os.environ["VLLM_USE_RAY_V2_EXECUTOR_BACKEND"] = "1"
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from pkg_util.tpu_topology import TPUTopologyDiscovery


@ray.remote
class VLLMSubsliceWorker:
    def __init__(self, model_name: str, tensor_parallel_size: int = 8):
        self.model_name = model_name
        self.tp_size = tensor_parallel_size

    def run_inference(self, prompts: list[str]):
        from vllm import LLM, SamplingParams

        print(f"[vLLM Worker] Initializing LLM '{self.model_name}' with TP={self.tp_size}...")
        llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tp_size,
            trust_remote_code=True,
            enforce_eager=False,
        )

        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)
        outputs = llm.generate(prompts, sampling_params)

        results = []
        for o in outputs:
            results.append({
                "prompt": o.prompt,
                "generated_text": o.outputs[0].text
            })
        return results


def main():
    # 1. Connect to Ray
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    # 2. Discover topology and select a physically connected 2-host subslice (8 TPUs)
    discovery = TPUTopologyDiscovery()
    pair = discovery.select_connected_2host_submesh()

    print(f"\n{'='*75}")
    print(f"Selected Connected TPU Host Subslice (TP=8):")
    print(f"  - Host A: Worker {pair.host_a.worker_id} (IP: {pair.host_a.node_ip})")
    print(f"  - Host B: Worker {pair.host_b.worker_id} (IP: {pair.host_b.node_ip})")
    print(f"  - Topology: {pair.mesh_dimension}")
    print(f"{'='*75}\n")

    # 3. Create Ray Placement Group pinned to the two connected node IPs
    bundle_specs = [
        {"TPU": 4, f"node:{pair.host_a.node_ip}": 1},
        {"TPU": 4, f"node:{pair.host_b.node_ip}": 1},
    ]
    print(f"Creating Placement Group with bundles: {bundle_specs}...")
    pg = placement_group(bundle_specs, strategy="STRICT_SPREAD")
    ray.get(pg.ready(), timeout=600)
    print("Placement Group is ready!")

    # 4. Launch the vLLM Actor inside the placement group
    # (By placing the worker on Bundle 0, vLLM will capture and use the parent PG)
    model_name = "google/gemma-2-9b"  # or gemma-4-31b / target model
    worker = VLLMSubsliceWorker.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=0
        )
    ).remote(model_name=model_name, tensor_parallel_size=8)

    prompts = [
        "Explain the theory of general relativity in simple terms.",
        "What are the primary differences between TPU and GPU architectures?",
    ]

    print("Submitting generation request to vLLM...")
    # results = ray.get(worker.run_inference.remote(prompts))
    # for r in results:
    #     print(f"\nPrompt: {r['prompt']}\nResponse: {r['generated_text']}\n")

    # Clean up placement group when done
    # ray.util.remove_placement_group(pg)


if __name__ == "__main__":
    main()
