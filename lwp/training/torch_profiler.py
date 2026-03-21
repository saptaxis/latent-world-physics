"""torch.profiler integration for GPU kernel-level Chrome traces.

Generates traces viewable at chrome://tracing or perfetto.dev.
Activated via --torch-profile flag. Traces a configurable window of
training steps to keep output manageable.

Usage in train.py:
    tp = make_torch_profiler(enabled=args.torch_profile,
                             trace_dir=str(run_dir / "torch_trace"))
    for epoch in ...:
        for batch in ...:
            ...
            if tp is not None:
                tp.step()
    if tp is not None:
        tp.stop()
"""
from __future__ import annotations

from pathlib import Path

import torch.profiler


def make_torch_profiler(
    enabled: bool = False,
    trace_dir: str = "./torch_trace",
    wait_steps: int = 5,
    warmup_steps: int = 1,
    active_steps: int = 10,
) -> torch.profiler.profile | None:
    """Create a torch.profiler.profile instance or None if disabled.

    Args:
        enabled: whether to create the profiler
        trace_dir: directory for Chrome trace output files
        wait_steps: steps to skip before warmup (lets model settle)
        warmup_steps: warmup steps (profiler overhead stabilizes)
        active_steps: steps to actively trace
    """
    if not enabled:
        return None

    Path(trace_dir).mkdir(parents=True, exist_ok=True)

    schedule = torch.profiler.schedule(
        wait=wait_steps,
        warmup=warmup_steps,
        active=active_steps,
        repeat=1,
    )

    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    return profiler
