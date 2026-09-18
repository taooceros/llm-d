"""
Reverse Control Lab (Demos, Benchmarks, and Verification Suites).
SDK resides in `omp_rcm`.
"""

from omp_rcm import (
    AbortExecution,
    DiagnosticTrap,
    JointTimingLogEngine,
    SkipStage,
    StageMetadata,
    StageRunner,
    StateStore,
    SteeringReceiver,
    TrapInterrupt,
    WatchdogTimer,
    stage,
)

__all__ = [
    "AbortExecution",
    "DiagnosticTrap",
    "JointTimingLogEngine",
    "SkipStage",
    "StageMetadata",
    "StageRunner",
    "StateStore",
    "SteeringReceiver",
    "TrapInterrupt",
    "WatchdogTimer",
    "stage",
]
