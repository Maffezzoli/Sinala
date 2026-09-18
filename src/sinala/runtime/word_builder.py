"""Real-time word builder and debouncer for Libras fingerspelling."""

from __future__ import annotations

import time
from typing import Any


class WordBuilder:
    """Accumulates stable fingerspelled letters into words and sentences."""

    def __init__(
        self,
        stability_frames: int = 12,
        confidence_threshold: float = 0.70,
        repeat_cooldown_seconds: float = 1.2,
    ) -> None:
        self.stability_frames = stability_frames
        self.confidence_threshold = confidence_threshold
        self.repeat_cooldown_seconds = repeat_cooldown_seconds

        self.current_candidate: str | None = None
        self.consecutive_count: int = 0
        self.awaiting_release: bool = False
        self.last_committed_letter: str | None = None
        self.last_commit_time: float = 0.0

        self.word_buffer: list[str] = []
        self.committed_words: list[str] = []

    def update(self, letter: str, confidence: float) -> str | None:
        """Process a frame prediction and commit only after release/rearm."""
        now = time.monotonic()
        if letter == "?" or confidence < self.confidence_threshold:
            self.consecutive_count = 0
            self.current_candidate = None
            self.awaiting_release = False
            return None

        if letter == self.current_candidate:
            if self.awaiting_release:
                return None
            self.consecutive_count += 1
        else:
            self.current_candidate = letter
            self.consecutive_count = 1
            self.awaiting_release = False

        if self.consecutive_count >= self.stability_frames:
            self.word_buffer.append(letter)
            self.last_committed_letter = letter
            self.last_commit_time = now
            self.consecutive_count = 0
            self.awaiting_release = True
            return letter

        return None

    def get_progress(self) -> float:
        """Return completion progress towards committing the current candidate (0.0 to 1.0)."""
        if not self.current_candidate or self.awaiting_release or self.stability_frames <= 0:
            return 0.0
        return min(1.0, self.consecutive_count / self.stability_frames)

    def add_space(self) -> None:
        """Finish the current word and advance to the next word."""
        if self.word_buffer:
            self.committed_words.append("".join(self.word_buffer))
            self.word_buffer = []
        self.last_committed_letter = None
        self.current_candidate = None
        self.consecutive_count = 0
        self.awaiting_release = False

    def add_word(self, word: str) -> None:
        """Directly commit an entire recognized word or greeting into the text buffer."""
        word_clean = word.strip().upper()
        if not word_clean:
            return
        if self.word_buffer:
            self.committed_words.append("".join(self.word_buffer))
            self.word_buffer = []
        self.committed_words.append(word_clean)
        self.current_candidate = None
        self.consecutive_count = 0
        self.awaiting_release = False
        self.last_committed_letter = None
        self.last_commit_time = time.monotonic()

    def backspace(self) -> str | None:
        """Remove the last letter or last word."""
        if self.word_buffer:
            removed = self.word_buffer.pop()
            self.last_committed_letter = None
            self.current_candidate = None
            self.consecutive_count = 0
            self.awaiting_release = False
            return removed
        if self.committed_words:
            removed_word = self.committed_words.pop()
            self.word_buffer = list(removed_word)
            self.current_candidate = None
            self.consecutive_count = 0
            self.awaiting_release = False
            return " "
        return None

    def clear(self) -> None:
        """Clear all buffers."""
        self.word_buffer.clear()
        self.committed_words.clear()
        self.current_candidate = None
        self.consecutive_count = 0
        self.awaiting_release = False
        self.last_committed_letter = None

    @property
    def current_word(self) -> str:
        """Return the word currently being spelled."""
        return "".join(self.word_buffer)

    @property
    def full_text(self) -> str:
        """Return the entire composed sentence so far."""
        words = list(self.committed_words)
        if self.word_buffer:
            words.append("".join(self.word_buffer))
        return " ".join(words)

    def to_dict(self) -> dict[str, Any]:
        """Debug state dictionary."""
        return {
            "current_candidate": self.current_candidate,
            "progress": self.get_progress(),
            "current_word": self.current_word,
            "full_text": self.full_text,
        }
