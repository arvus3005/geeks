import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class RequestTimer:
    stages: dict[str, float] = field(default_factory=dict)
    _start: float = field(default_factory=time.monotonic, init=False)

    @contextmanager
    def stage(self, name: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.stages[name] = (time.monotonic() - t0) * 1000.0

    def total_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000.0

    def deadline(self, budget_s: float) -> float:
        """Absolute monotonic deadline: this request's start time plus a
        total budget -- lets a downstream stage (e.g. the reranker) know how
        much of the WHOLE request's time is left, not just its own fixed
        local allowance. Without this, a stage that already ran long
        upstream (slow retrieval, GC pause, etc.) still hands the next stage
        a fresh full budget on top, which is exactly how a single-stage
        timeout can still blow the end-to-end target."""
        return self._start + budget_s
