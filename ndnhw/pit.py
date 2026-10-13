from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict


@dataclass(frozen=True)
class PITInterval:
    arrival: int
    departure: int

    def __post_init__(self):
        if self.departure < self.arrival:
            raise ValueError("departure must be >= arrival")


@dataclass(frozen=True)
class PITFeasibilityReport:
    max_occupancy: int
    occupancy_feasible: bool
    max_updates_per_window: int
    update_feasible: bool
    feasible: bool

    def as_dict(self) -> dict:
        return {
            "max_occupancy": self.max_occupancy,
            "occupancy_feasible": self.occupancy_feasible,
            "max_updates_per_window": self.max_updates_per_window,
            "update_feasible": self.update_feasible,
            "feasible": self.feasible,
        }


def pit_feasibility(
    intervals: list[PITInterval],
    K: int,
    U: int,
    window: int = 1,
) -> PITFeasibilityReport:
    """Check PIT dynamic feasibility.

    K: maximum active PIT entries.
    U: maximum insert/delete updates per time window.
    window: integer window size.
    """

    events: list[tuple[int, int]] = []
    updates_by_time = defaultdict(int)

    for item in intervals:
        events.append((item.arrival, +1))
        events.append((item.departure, -1))
        updates_by_time[item.arrival] += 1
        updates_by_time[item.departure] += 1

    events.sort(key=lambda x: (x[0], -x[1]))

    active = 0
    max_occupancy = 0
    for _, delta in events:
        active += delta
        max_occupancy = max(max_occupancy, active)

    if not updates_by_time:
        max_updates = 0
    else:
        times = sorted(updates_by_time)
        min_t, max_t = min(times), max(times)
        max_updates = 0
        for start in range(min_t, max_t + 1):
            end = start + window
            count = sum(c for t, c in updates_by_time.items() if start <= t < end)
            max_updates = max(max_updates, count)

    occupancy_feasible = max_occupancy <= K
    update_feasible = max_updates <= U

    return PITFeasibilityReport(
        max_occupancy=max_occupancy,
        occupancy_feasible=occupancy_feasible,
        max_updates_per_window=max_updates,
        update_feasible=update_feasible,
        feasible=occupancy_feasible and update_feasible,
    )
