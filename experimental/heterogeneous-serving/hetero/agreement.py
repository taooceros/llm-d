"""Cross-layout greedy agreement gate (Q0 bf16-KV).

Within a layout, replicas must be token-for-token identical. Across layouts
(different TP sizes) bf16 KV perturbs logits, so greedy outputs may diverge. A
layout pair passes when, for every first divergence, both chosen tokens sit in
the top-32 table at the shared prefix within `tie_gap_max_nats` of the argmax
(a near-tie), at least `exact_prompts_min` of 32 prompts match exactly, and at
most `early_divergences_max` prompts diverge before generated index
`early_divergence_index`. Stdlib only; pure functions.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

CROSS_LAYOUT_CRITERIA = {
    "provenance": "Q0 bf16-KV gate, bulk_reconfiguration_plan.md decision log 2026-10-05",
    "top_k": 32,                    # next_token_diagnostic returns top-32 logprobs
    "tie_gap_max_nats": 1.0,        # covers the worst measured TP8/TP16 bf16-KV perturbation (0.92 nats)
    "exact_prompts_min": 16,        # of 32
    "early_divergence_index": 4,    # divergence at generated index < 4 counts as early
    "early_divergences_max": 2,
}


def first_divergence(a: List[int], b: List[int]) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def replica_equality(outputs: Dict[str, List[List[int]]], tp_sizes: Dict[str, int]) -> Dict[str, Any]:
    by_tp: Dict[int, List[str]] = {}
    for name in sorted(outputs):
        by_tp.setdefault(int(tp_sizes[name]), []).append(name)
    groups = {}
    for tp, names in by_tp.items():
        groups[str(tp)] = {"engines": names,
                           "passed": all(outputs[n] == outputs[names[0]] for n in names)}
    return {"groups": groups, "passed": all(g["passed"] for g in groups.values())}


def canonical_engine(outputs: Dict[str, Any]) -> str:
    return sorted(outputs)[0]


def divergence_points(self_outputs: List[List[int]], other_outputs: List[List[int]]) -> List[Dict[str, Any]]:
    if len(self_outputs) != len(other_outputs):
        raise ValueError("output lists differ in length")
    points = []
    for i, (s, o) in enumerate(zip(self_outputs, other_outputs)):
        d = first_divergence(s, o)
        if d is None:
            continue
        points.append({"prompt_index": i, "d": d, "len_self": len(s), "len_other": len(o),
                       "token_self": s[d] if d < len(s) else None,
                       "token_other": o[d] if d < len(o) else None})
    return points


def tie_check(top_logprobs: Dict, token_self: Optional[int], token_other: Optional[int],
              criteria: Dict[str, Any] = CROSS_LAYOUT_CRITERIA) -> Dict[str, Any]:
    lp = {int(k): float(v) for k, v in top_logprobs.items()}
    finite = all(math.isfinite(v) for v in lp.values())
    if token_self is None and token_other is None:
        return {"in_top_self": None, "in_top_other": None, "argmax": None, "argmax_is_self": None,
                "gap_nats": 0.0, "finite": finite, "passed": True}
    in_self = token_self is not None and int(token_self) in lp
    in_other = token_other is not None and int(token_other) in lp
    argmax = max(lp, key=lp.get) if lp else None
    gap = None
    if in_self and in_other:
        gap = max(lp.values()) - min(lp[int(token_self)], lp[int(token_other)])
    passed = bool(finite and in_self and in_other and gap is not None
                  and gap <= criteria["tie_gap_max_nats"])
    return {"in_top_self": in_self, "in_top_other": in_other, "argmax": argmax,
            "argmax_is_self": argmax is not None and token_self is not None and argmax == int(token_self),
            "gap_nats": gap, "finite": finite, "passed": passed}


def evaluate_side(self_outputs: List[List[int]], other_outputs: List[List[int]],
                  diagnostics: List[Dict[str, Any]],
                  criteria: Dict[str, Any] = CROSS_LAYOUT_CRITERIA) -> Dict[str, Any]:
    points = divergence_points(self_outputs, other_outputs)
    by_key = {(r["prompt_index"], r["d"]): r for r in diagnostics}
    keys = {(p["prompt_index"], p["d"]) for p in points}
    error = None
    if set(by_key) != keys:
        error = (f"diagnostics keys differ from divergence points: missing {sorted(keys - set(by_key))}, "
                 f"extra {sorted(set(by_key) - keys)}")
    rows = []
    for p in points:
        diag = by_key.get((p["prompt_index"], p["d"]))
        if diag is None:
            rows.append({**p, "passed": False, "error": "missing diagnostic"})
            continue
        rows.append({**p, **tie_check(diag["top_logprobs"], p["token_self"], p["token_other"], criteria)})
    n = len(self_outputs)
    checks = {
        "ties": error is None and all(r["passed"] for r in rows),
        "exact_prompts": n - len(points) >= criteria["exact_prompts_min"],
        "early_divergences": sum(1 for p in points if p["d"] < criteria["early_divergence_index"])
                             <= criteria["early_divergences_max"],
    }
    result = {"passed": all(checks.values()), "checks": checks, "exact_prompts": n - len(points),
              "prompts": n, "divergences": rows,
              "first_divergence_min": min((p["d"] for p in points), default=None),
              "criteria": criteria}
    if error:
        result["error"] = error
    return result


def compare_distributions(a, b):
    """Compare two top-logprob tables (str or int keys) of the same next-token position."""
    a = {int(k): v for k, v in a.items()}
    b = {int(k): v for k, v in b.items()}
    argmax_a = max(a, key=a.get)
    argmax_b = max(b, key=b.get)
    shared = a.keys() & b.keys()
    return {"argmax_a": argmax_a, "argmax_b": argmax_b, "argmax_equal": argmax_a == argmax_b,
            "shared": len(shared),
            "max_abs_dlogprob_shared": max((abs(a[k] - b[k]) for k in shared), default=None),
            "top1_a_lp_in_b": b.get(argmax_a)}
