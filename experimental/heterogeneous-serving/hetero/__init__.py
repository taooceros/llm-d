"""Static heterogeneous TPU serving fabric for llm-d.

Packages the reproducible layout resolver, scoped placement-group ownership,
per-instance TPU runtime environment contract, serving-instance registry and
gateway integration used by the heterogeneous bulk-inference testbed.
"""

__all__ = [
    "evidence",
    "layout",
    "placement",
    "slice_inventory",
]
