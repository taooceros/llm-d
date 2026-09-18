# JAX Bucket Precompilation Minimal Demo

This directory contains an extracted, minimal standalone demonstration of how `vllm` / `tpu-inference` precompiles different input batch sizes and token lengths to avoid runtime XLA JIT compilation overhead.

## Directory Structure

```
demos/bucket-precompilation/
├── README.md
└── main.py
```

## Extracted Minimal Components

1. **Bucket Math Utilities**:
   - `get_token_paddings(min_token_size, max_token_size, padding_gap)`: Computes token padding bucket sizes (exponential powers of 2 or linear gaps).
   - `get_req_paddings(min_req_size, max_req_size)`: Computes power-of-two request batch size buckets.
   - `get_padded_token_len(paddings, x)`: Binary searches (`bisect_left`) the target bucket shape for dynamic input lengths.

2. **Minimal Compilation Manager (`MinimalCompilationManager`)**:
   - Manages AOT lowering (`fn.lower(*args)`) into JAXpr/HLO IR.
   - Compiles HLO graphs into executable binaries (`lowered.compile()`) across background threads (`ThreadPoolExecutor`).
   - Runs a device warmup pass and forces hardware synchronization via `jax.tree.map(lambda r: r.block_until_ready(), out)`.

3. **Hot-Path Zero-Compilation Guardrail (`ForbidCompile`)**:
   - Context manager that intercepts JAX's internal `pxla._cached_lowering_to_hlo`.
   - Intercepts cache misses at runtime and throws a `RuntimeError` if an un-warmed shape triggers compilation.

## Running the Demo

Execute the demo script using Python (requires `jax` and `jaxlib`):

```bash
python3 main.py
```

### Expected Output Summary

1. Computes bucket lists (e.g. `[16, 32, 64, 128, 192, 256]`).
2. Submits parallel AOT compilation tasks for each bucket shape.
3. Flushes compilation futures and executes the warmup pass.
4. Executes dynamic inputs mapped to padded buckets inside `ForbidCompile` context, verifying sub-millisecond execution with **Zero-Compile Cache Hits**.
5. Catches and logs a forbidden runtime compilation error when an uncompiled shape is attempted.
