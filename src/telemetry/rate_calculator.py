from dataclasses import dataclass

@dataclass
class CounterSample:
    value: int
    timestamp_s: float

class RateCalculator:
    def __init__(self):
        self._previous = {}

    def update(self, key: str, total: int, timestamp_s: float) -> float:
        previous = self._previous.get(key)
        self._previous[key] = CounterSample(total, timestamp_s)
        if previous is None:
            return 0.0
        dt = timestamp_s - previous.timestamp_s
        delta = total - previous.value
        if dt <= 0 or delta < 0:
            return 0.0
        return delta / dt
