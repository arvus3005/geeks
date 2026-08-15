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
