from lwp.training.scheduling import curriculum_schedule, sampling_schedule


def test_curriculum_schedule_starts_at_min():
    k = curriculum_schedule(epoch=0, total_epochs=100, k_min=1, k_max=20)
    assert k == 1


def test_curriculum_schedule_ends_at_max():
    k = curriculum_schedule(epoch=100, total_epochs=100, k_min=1, k_max=20)
    assert k == 20


def test_curriculum_schedule_monotonic():
    ks = [curriculum_schedule(e, 100, 1, 20) for e in range(101)]
    for i in range(len(ks) - 1):
        assert ks[i + 1] >= ks[i]


def test_curriculum_schedule_midpoint():
    k = curriculum_schedule(epoch=50, total_epochs=100, k_min=1, k_max=20)
    assert 9 <= k <= 11  # approximately halfway


def test_sampling_schedule_starts_at_start():
    p = sampling_schedule(epoch=0, total_epochs=100, start=0.0, end=0.5)
    assert p == 0.0


def test_sampling_schedule_ends_at_end():
    p = sampling_schedule(epoch=100, total_epochs=100, start=0.0, end=0.5)
    assert abs(p - 0.5) < 1e-6


def test_sampling_schedule_monotonic():
    ps = [sampling_schedule(e, 100, 0.0, 0.5) for e in range(101)]
    for i in range(len(ps) - 1):
        assert ps[i + 1] >= ps[i]


def test_sampling_schedule_midpoint():
    p = sampling_schedule(epoch=50, total_epochs=100, start=0.0, end=0.5)
    assert abs(p - 0.25) < 1e-6
