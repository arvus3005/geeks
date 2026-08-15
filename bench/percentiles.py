import statistics
from collections.abc import Sequence


def compute_percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot compute percentiles of empty sequence")
    s = sorted(values)
    n = len(s)

    def pct(p):
        idx = int(p / 100 * n)
        return s[min(idx, n - 1)]

    return {
        "p50": pct(50),
        "p70": pct(70),
        "p95": pct(95),
        "p99": pct(99),
        "p100": s[-1],  # P100 is the actual maximum
        "mean": statistics.mean(s),
        "count": n,
    }
