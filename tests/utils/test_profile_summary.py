import json
from scripts.world_models.profile_summary import load_profile, summarize, format_table


def _write_log(path, events):
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def test_load_profile(tmp_path):
    path = tmp_path / "profile.jsonl"
    events = [
        {"ts": 1.0, "step": 1, "epoch": 0, "phase": "forward", "dur_ms": 0.5},
        {"ts": 1.1, "step": 1, "epoch": 0, "phase": "backward", "dur_ms": 0.3},
    ]
    _write_log(path, events)
    loaded = load_profile(path)
    assert len(loaded) == 2
    assert loaded[0]["phase"] == "forward"


def test_summarize_aggregates():
    events = [
        {"ts": 1.0, "step": 1, "epoch": 0, "phase": "forward", "dur_ms": 1.0},
        {"ts": 1.1, "step": 2, "epoch": 0, "phase": "forward", "dur_ms": 2.0},
        {"ts": 1.2, "step": 1, "epoch": 0, "phase": "backward", "dur_ms": 0.5},
    ]
    summary = summarize(events)
    assert summary["forward"]["count"] == 2
    assert summary["forward"]["total_ms"] == 3.0
    assert summary["forward"]["mean_ms"] == 1.5
    assert summary["backward"]["count"] == 1


def test_summarize_with_step_range():
    events = [
        {"ts": 1.0, "step": 1, "epoch": 0, "phase": "forward", "dur_ms": 1.0},
        {"ts": 1.1, "step": 5, "epoch": 0, "phase": "forward", "dur_ms": 2.0},
        {"ts": 1.2, "step": 10, "epoch": 0, "phase": "forward", "dur_ms": 3.0},
    ]
    summary = summarize(events, step_min=5, step_max=10)
    assert summary["forward"]["count"] == 2
    assert summary["forward"]["total_ms"] == 5.0


def test_format_table_output():
    events = [
        {"ts": 1.0, "step": 1, "epoch": 0, "phase": "forward", "dur_ms": 10.0},
        {"ts": 1.1, "step": 1, "epoch": 0, "phase": "backward", "dur_ms": 5.0},
    ]
    summary = summarize(events)
    table = format_table(summary)
    assert "forward" in table
    assert "backward" in table
    # forward is slower, should appear first
    assert table.index("forward") < table.index("backward")


def test_summarize_per_callback():
    events = [
        {"ts": 1.0, "step": 1, "epoch": 0, "phase": "cb/ValidationCallback", "dur_ms": 800.0},
        {"ts": 1.1, "step": 1, "epoch": 0, "phase": "cb/GradNormCallback", "dur_ms": 0.1},
        {"ts": 1.2, "step": 2, "epoch": 0, "phase": "cb/ValidationCallback", "dur_ms": 750.0},
    ]
    summary = summarize(events)
    assert summary["cb/ValidationCallback"]["count"] == 2
    assert summary["cb/GradNormCallback"]["count"] == 1
