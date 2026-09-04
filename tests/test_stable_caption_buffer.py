from sinala.runtime.stable_caption_buffer import StableCaptionBuffer


def test_stable_caption_buffer_when_consensus_reaches_threshold_then_accepts_label() -> None:
    buffer = StableCaptionBuffer(confidence_threshold=0.7, window_size=5, minimum_consensus=4, cooldown_seconds=0.0)

    for timestamp in range(4):
        result = buffer.update("banco", 0.9, True, now=float(timestamp))

    assert result == "banco"


def test_stable_caption_buffer_when_confidence_is_low_then_keeps_previous_label() -> None:
    buffer = StableCaptionBuffer(confidence_threshold=0.7, window_size=3, minimum_consensus=2, cooldown_seconds=0.0)
    buffer.update("banco", 0.9, True, now=0.0)
    buffer.update("banco", 0.9, True, now=1.0)

    result = buffer.update("vacina", 0.6, True, now=2.0)

    assert result == "banco"


def test_stable_caption_buffer_when_no_hand_is_detected_then_keeps_previous_label() -> None:
    buffer = StableCaptionBuffer(confidence_threshold=0.7, window_size=2, minimum_consensus=1, cooldown_seconds=0.0)
    buffer.update("banco", 0.9, True, now=0.0)

    result = buffer.update(None, 0.0, False, now=1.0)

    assert result == "banco"


def test_stable_caption_buffer_when_same_label_repeats_then_does_not_change_label() -> None:
    buffer = StableCaptionBuffer(confidence_threshold=0.7, window_size=2, minimum_consensus=1, cooldown_seconds=0.0)
    first = buffer.update("banco", 0.9, True, now=0.0)
    second = buffer.update("banco", 0.9, True, now=1.0)

    assert first == second == "banco"
