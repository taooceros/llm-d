#!/usr/bin/env python3
"""
Minimal standalone demo extracting vLLM/TPU-Inference JAX bucket precompilation.

Demonstrates:
1. Bucket dimension math (token paddings, request paddings, bisect matching).
2. Creating dummy input arrays matching bucket shapes.
3. JAX AOT lowering (fn.lower(*args)) and XLA compilation (lowered.compile()).
4. Multi-threaded background compilation (ThreadPoolExecutor).
5. Device warmup pass and hardware synchronization via block_until_ready().
6. Hot-path zero-compilation enforcement via ForbidCompile.
"""

import bisect
import functools
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, List, Tuple

import jax
import jax.numpy as jnp
from jax.interpreters import pxla

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("bucket-precompilation-demo")


# =====================================================================
# 1. Bucket Math Utilities (Extracted from tpu_inference/runner/utils.py)
# =====================================================================

MIN_NUM_SEQS = 1


def get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int) -> int:
    res = MIN_NUM_SEQS if x <= MIN_NUM_SEQS else 1 << (x - 1).bit_length()
    return min(res, upper_limit)


def get_req_paddings(min_req_size: int, max_req_size: int) -> List[int]:
    """Generate power-of-two request padding buckets."""
    assert (min_req_size & (min_req_size - 1) == 0) and min_req_size > 0
    paddings: List[int] = []
    num = max(MIN_NUM_SEQS, min_req_size)
    while num <= max_req_size and (len(paddings) == 0 or paddings[-1] != num):
        paddings.append(num)
        num = get_padded_num_reqs_with_upper_limit(num + 1, max_req_size)
    return paddings


def get_token_paddings(min_token_size: int, max_token_size: int, padding_gap: int) -> List[int]:
    """Generate padded token sizes starting from min_token_size up to max_token_size.

    If padding_gap == 0: double exponentially (16, 32, 64, 128, ...).
    If padding_gap > 0: double up to padding_gap, then increment linearly by padding_gap (+256).
    """
    assert (min_token_size & (min_token_size - 1) == 0) and min_token_size > 0
    paddings = []
    num = min_token_size

    if padding_gap == 0:
        while True:
            paddings.append(num)
            if num >= max_token_size:
                break
            num *= 2
    else:
        while num <= padding_gap:
            paddings.append(num)
            num *= 2
        num //= 2
        while num < max_token_size:
            num += padding_gap
            paddings.append(num)
    return paddings


def get_padded_token_len(paddings: List[int], x: int) -> int:
    """Binary search for the smallest bucket in paddings >= x."""
    index = bisect.bisect_left(paddings, x)
    assert index < len(paddings), f"Input length {x} exceeds max bucket {paddings[-1]}"
    return paddings[index]


def _find_jax_caching_target():
    """Dynamically locate _cached_lowering_to_hlo across JAX versions."""
    for mod_path in ["jax._src.dispatch", "jax._src.interpreters.pxla", "jax.interpreters.pxla"]:
        try:
            mod = __import__(mod_path, fromlist=["_cached_lowering_to_hlo"])
            if hasattr(mod, "_cached_lowering_to_hlo"):
                return mod, "_cached_lowering_to_hlo"
        except (ImportError, AttributeError):
            pass
    return None, None


class ForbidCompile:
    """Context manager to verify zero JAX JIT compilation on hot path."""

    def __init__(self, message="JAX compilation occurred but was forbidden in this context."):
        self.message = message
        self._target_mod = None
        self._attr_name = None
        self._original_func = None

    def __enter__(self):
        mod, attr = _find_jax_caching_target()
        if mod is None or attr is None:
            logger.warning("Could not locate JAX internal _cached_lowering_to_hlo for ForbidCompile check.")
            return self

        self._target_mod = mod
        self._attr_name = attr
        self._original_func = getattr(mod, attr)
        original_cached_func = self._original_func

        @functools.wraps(original_cached_func)
        def wrapper(*args, **kwargs):
            info_before = original_cached_func.cache_info()
            misses_before = info_before.misses
            result = original_cached_func(*args, **kwargs)
            info_after = original_cached_func.cache_info()
            misses_after = info_after.misses

            if misses_after > misses_before:
                raise RuntimeError(self.message)

            return result

        setattr(mod, attr, wrapper)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._target_mod is not None and self._attr_name is not None and self._original_func is not None:
            setattr(self._target_mod, self._attr_name, self._original_func)


# =====================================================================
# 3. Minimal Compilation Manager (Extracted from compilation_manager.py)
# =====================================================================

class MinimalCompilationManager:
    """Minimal orchestrator for JAX AOT lowering, compilation, and warmup."""

    def __init__(self, num_workers: int = 2):
        self._compile_executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="aot_compile")
        self._compile_futures: List[Future] = []
        self._warmup_tasks: List[Tuple[str, Callable, Tuple, dict]] = []

    def run_compilation(self, name: str, fn: Callable, *args, call_kwargs=None) -> None:
        if call_kwargs is None:
            call_kwargs = {}

        self._warmup_tasks.append((name, fn, args, call_kwargs))

        if not hasattr(fn, "lower"):
            logger.info(f"Skipping AOT for un-jitted function: {name}")
            return

        # Lower Python/JAXpr to HLO module
        lowered = fn.lower(*args, **call_kwargs)

        def _compile(lowered_graph, log_name):
            start = time.perf_counter()
            compiled = lowered_graph.compile()  # Triggers XLA compilation
            elapsed = time.perf_counter() - start
            logger.info(f"XLA compilation of [{log_name}] finished in {elapsed:.4f}s")
            return compiled

        future = self._compile_executor.submit(_compile, lowered, name)
        self._compile_futures.append(future)

    def flush_compilations(self) -> None:
        """Await all XLA compilation futures and execute warmup pass."""
        futures, self._compile_futures = self._compile_futures, []
        tasks, self._warmup_tasks = self._warmup_tasks, []

        # Await thread futures
        for fut in futures:
            fut.result()

        logger.info(f"Flushed {len(futures)} background compilations. Running warmup pass...")

        warmup_start = time.perf_counter()
        for name, fn, args, call_kwargs in tasks:
            out = fn(*args, **call_kwargs)
            # Force hardware synchronization on device
            jax.tree.map(lambda r: r.block_until_ready() if hasattr(r, "block_until_ready") else r, out)
        
        elapsed = time.perf_counter() - warmup_start
        logger.info(f"Warmup pass completed in {elapsed:.4f}s for {len(tasks)} tasks.")

    def shutdown(self):
        self._compile_executor.shutdown(wait=True)


# =====================================================================
# 4. Model & Execution Pipeline Demo
# =====================================================================

# Example JAX model forward computation
@jax.jit
def model_forward_step(params: dict, input_ids: jax.Array) -> jax.Array:
    """Dummy transformer backbone step performing embedding lookup and MLP projection."""
    embeddings = params["wte"][input_ids]  # (num_tokens, hidden_dim)
    output = jnp.matmul(embeddings, params["w_mlp"])  # (num_tokens, hidden_dim)
    return jnp.sin(output)


def main():
    logger.info("Starting JAX Bucket Precompilation Demo...")

    # Define model dimensions
    vocab_size = 1000
    hidden_dim = 64
    max_num_batched_tokens = 256

    # Initialize dummy model parameters
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    params = {
        "wte": jax.random.normal(k1, (vocab_size, hidden_dim)),
        "w_mlp": jax.random.normal(k2, (hidden_dim, hidden_dim)),
    }

    # Derive token and request buckets
    token_buckets = get_token_paddings(min_token_size=16, max_token_size=max_num_batched_tokens, padding_gap=64)
    request_buckets = get_req_paddings(min_req_size=1, max_req_size=8)

    logger.info(f"Derived Token Buckets: {token_buckets}")
    logger.info(f"Derived Request Buckets: {request_buckets}")

    compiler = MinimalCompilationManager(num_workers=4)

    # -----------------------------------------------------------------
    # AOT Precompile across all token buckets
    # -----------------------------------------------------------------
    start_precompile = time.perf_counter()
    logger.info("Submitting AOT precompilation tasks for all bucket shapes...")

    for num_tokens in token_buckets:
        # Create dummy input array of shape (num_tokens,)
        dummy_input_ids = jnp.ones((num_tokens,), dtype=jnp.int32)
        
        compiler.run_compilation(
            f"model_forward_num_tokens_{num_tokens}",
            model_forward_step,
            params,
            dummy_input_ids,
        )

    # Flush compilations and run device warmup
    compiler.flush_compilations()
    compiler.shutdown()
    total_precompile_time = time.perf_counter() - start_precompile
    logger.info(f"Precompilation & Warmup complete in {total_precompile_time:.4f}s!\n")

    # -----------------------------------------------------------------
    # Runtime Inference & Zero-Compilation Verification
    # -----------------------------------------------------------------
    logger.info("Simulating runtime request execution with dynamic un-padded inputs...")

    test_input_lengths = [12, 45, 100, 200]

    for unpadded_len in test_input_lengths:
        # Step 1: Binary search to find padded bucket shape
        padded_len = get_padded_token_len(token_buckets, unpadded_len)

        # Step 2: Construct actual input padded to bucket shape
        raw_tokens = jnp.arange(unpadded_len, dtype=jnp.int32) % vocab_size
        padded_input_ids = jnp.pad(raw_tokens, (0, padded_len - unpadded_len), mode="constant", constant_values=0)

        # Step 3: Run model inside ForbidCompile context manager
        start_exec = time.perf_counter()
        with ForbidCompile(message=f"Cache miss! Shape {padded_len} was not precompiled!"):
            output = model_forward_step(params, padded_input_ids)
            output.block_until_ready()
        exec_time = (time.perf_counter() - start_exec) * 1000

        logger.info(
            f"Input len: {unpadded_len:3d} -> Matched Bucket: {padded_len:3d} | "
            f"Exec time: {exec_time:.3f}ms | Verified Zero-Compile Cache Hit!"
        )

    # -----------------------------------------------------------------
    # Negative Test: Un-precompiled shape should trigger ForbidCompile error
    # -----------------------------------------------------------------
    uncompiled_len = 300  # Exceeds precompiled buckets
    logger.info(f"\nTesting uncompiled shape ({uncompiled_len}) with ForbidCompile guardrail...")
    uncompiled_input = jnp.ones((uncompiled_len,), dtype=jnp.int32)
    try:
        with ForbidCompile(message=f"Intercepted uncompiled shape {uncompiled_len}!"):
            _ = model_forward_step(params, uncompiled_input)
    except RuntimeError as e:
        logger.info(f"Successfully caught forbidden runtime compilation: {e}")

    logger.info("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
