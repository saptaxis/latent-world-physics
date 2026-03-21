from lwp.training.torch_profiler import make_torch_profiler


def test_make_torch_profiler_returns_none_when_disabled():
    result = make_torch_profiler(enabled=False, trace_dir="/tmp/traces")
    assert result is None


def test_make_torch_profiler_returns_profiler_when_enabled():
    import torch.profiler
    result = make_torch_profiler(enabled=True, trace_dir="/tmp/traces",
                                  wait_steps=2, warmup_steps=1, active_steps=3)
    assert result is not None
    assert isinstance(result, torch.profiler.profile)


def test_make_torch_profiler_default_schedule():
    """Default schedule: wait=5, warmup=1, active=10."""
    result = make_torch_profiler(enabled=True, trace_dir="/tmp/traces")
    assert result is not None
