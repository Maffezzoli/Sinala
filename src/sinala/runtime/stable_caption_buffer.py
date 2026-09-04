from collections import Counter, deque
from time import monotonic


class StableCaptionBuffer:
    def __init__(self, confidence_threshold: float = 0.75, window_size: int = 5, minimum_consensus: int = 4, cooldown_seconds: float = 0.8) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if window_size < 1 or minimum_consensus < 1 or minimum_consensus > window_size:
            raise ValueError("minimum_consensus must be between 1 and window_size")
        if cooldown_seconds < 0.0:
            raise ValueError("cooldown_seconds cannot be negative")
        self.confidence_threshold = confidence_threshold
        self.window_size = window_size
        self.minimum_consensus = minimum_consensus
        self.cooldown_seconds = cooldown_seconds
        self._candidates: deque[str | None] = deque(maxlen=window_size)
        self._current_label: str | None = None
        self._last_accept_time = float("-inf")

    @property
    def current_label(self) -> str | None:
        return self._current_label

    def update(self, label: str | None, confidence: float, hands_detected: bool, now: float | None = None) -> str | None:
        current_time = monotonic() if now is None else now
        accepted_candidate = label if hands_detected and confidence >= self.confidence_threshold else None
        self._candidates.append(accepted_candidate)
        counts = Counter(candidate for candidate in self._candidates if candidate is not None)
        if counts:
            candidate, count = counts.most_common(1)[0]
            if count >= self.minimum_consensus and candidate != self._current_label and current_time - self._last_accept_time >= self.cooldown_seconds:
                self._current_label = candidate
                self._last_accept_time = current_time
        return self._current_label


__all__ = ["StableCaptionBuffer"]
