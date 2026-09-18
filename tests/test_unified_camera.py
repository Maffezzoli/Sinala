"""Unit tests for UnifiedCameraApp initialization and components."""

from __future__ import annotations

from pathlib import Path

from sinala.runtime.unified_camera_app import UnifiedCameraApp


def test_unified_camera_app_init() -> None:
    greetings_ckpt = Path("artifacts/sinala_greetings.pt")
    alphabet_ckpt = Path("artifacts/sinala_alphabet.pt")
    model_asset = Path("models/hand_landmarker.task")

    app = UnifiedCameraApp(
        greetings_checkpoint_path=greetings_ckpt,
        alphabet_checkpoint_path=alphabet_ckpt,
        model_asset_path=model_asset,
        camera_index=0,
        alphabet_threshold=0.70,
        greetings_threshold=0.65,
        stability_frames=12,
        sequence_length=32,
    )

    assert app.alphabet_model is not None
    assert app.greetings_model is not None
    assert len(app.greetings_classes) == 9
    assert app.word_builder.full_text == ""
    assert "Oi" in app.greetings_classes.values()
