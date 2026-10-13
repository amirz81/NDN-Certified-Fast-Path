from ndnhw.pit import PITInterval, pit_feasibility


def test_pit_occupancy():
    intervals = [PITInterval(0, 10), PITInterval(1, 9), PITInterval(2, 8)]
    report = pit_feasibility(intervals, K=2, U=10)
    assert report.max_occupancy == 3
    assert not report.occupancy_feasible


def test_pit_update_burst():
    intervals = [PITInterval(0, 5), PITInterval(0, 5), PITInterval(0, 5)]
    report = pit_feasibility(intervals, K=10, U=2, window=1)
    assert not report.update_feasible
