import json
import time
from lwp.training.profiler import ProfileLogger


def test_logger_writes_jsonl(tmp_path):
    path = tmp_path / "profile.jsonl"
    log = ProfileLogger(path)
    with log.phase("forward", step=1, epoch=0):
        time.sleep(0.001)
    with log.phase("backward", step=1, epoch=0):
        time.sleep(0.001)
    log.close()

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert entry["phase"] == "forward"
    assert entry["step"] == 1
    assert entry["epoch"] == 0
    assert entry["dur_ms"] > 0
    assert "ts" in entry


def test_logger_disabled_writes_nothing(tmp_path):
    path = tmp_path / "profile.jsonl"
    log = ProfileLogger(None)  # disabled
    with log.phase("forward", step=1, epoch=0):
        time.sleep(0.001)
    log.close()
    assert not path.exists()


def test_logger_hierarchical_keys(tmp_path):
    path = tmp_path / "profile.jsonl"
    log = ProfileLogger(path)
    with log.phase("cb/ValidationCallback", step=5, epoch=0):
        time.sleep(0.001)
    with log.phase("cb/GradNormCallback", step=5, epoch=0):
        time.sleep(0.001)
    log.close()

    lines = path.read_text().strip().split("\n")
    phases = [json.loads(l)["phase"] for l in lines]
    assert "cb/ValidationCallback" in phases
    assert "cb/GradNormCallback" in phases


def test_logger_flushes_per_event(tmp_path):
    """Each event is readable immediately, even before close()."""
    path = tmp_path / "profile.jsonl"
    log = ProfileLogger(path)
    with log.phase("forward", step=1, epoch=0):
        pass
    # Read before close — should already be on disk
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["phase"] == "forward"
    log.close()


def test_logger_many_events(tmp_path):
    path = tmp_path / "profile.jsonl"
    log = ProfileLogger(path)
    for step in range(100):
        with log.phase("forward", step=step, epoch=0):
            pass
        with log.phase("backward", step=step, epoch=0):
            pass
    log.close()

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 200


def test_logger_enabled_property(tmp_path):
    path = tmp_path / "profile.jsonl"
    log = ProfileLogger(path)
    assert log.enabled is True
    log.close()

    disabled = ProfileLogger(None)
    assert disabled.enabled is False
    disabled.close()


def test_logger_log_event(tmp_path):
    """log_event() writes a single event without context manager."""
    path = tmp_path / "profile.jsonl"
    log = ProfileLogger(path)
    log.log_event("dataloader", dur_ms=1.234, step=5, epoch=1)
    log.close()

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["phase"] == "dataloader"
    assert entry["dur_ms"] == 1.234
    assert entry["step"] == 5
    assert entry["epoch"] == 1


def test_logger_log_event_disabled():
    """log_event() is a no-op when disabled."""
    log = ProfileLogger(None)
    log.log_event("forward", dur_ms=1.0)  # should not raise
    log.close()


def test_logger_context_manager(tmp_path):
    """ProfileLogger can be used as a context manager."""
    path = tmp_path / "profile.jsonl"
    with ProfileLogger(path) as log:
        with log.phase("forward", step=1, epoch=0):
            pass
    # File should be closed after exiting context
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1
