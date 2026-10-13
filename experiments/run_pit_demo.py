from __future__ import annotations

from ndnhw.pit import PITInterval, pit_feasibility


def main() -> None:
    intervals = [
        PITInterval(0, 10),
        PITInterval(1, 9),
        PITInterval(2, 8),
        PITInterval(3, 7),
        PITInterval(4, 6),
    ]

    print("Case A: capacity too small")
    print(pit_feasibility(intervals, K=3, U=10, window=1).as_dict())

    bursty = [
        PITInterval(0, 10),
        PITInterval(0, 10),
        PITInterval(0, 10),
        PITInterval(0, 10),
    ]

    print("Case B: update burst too large")
    print(pit_feasibility(bursty, K=10, U=2, window=1).as_dict())


if __name__ == "__main__":
    main()
