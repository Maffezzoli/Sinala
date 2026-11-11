"""Unit tests for the DuoLibrasRequestHandler component."""

import io
import threading
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock

from sinala.runtime.duolibras_request_handler import DuoLibrasRequestHandler
from sinala.runtime.duolibras_web_server import DuoLibrasWebServer


def test_request_handler_when_end_headers_then_sends_cors_and_security_headers() -> None:
    handler = DuoLibrasRequestHandler.__new__(DuoLibrasRequestHandler)
    handler.request_version = "HTTP/1.1"
    handler._headers_buffer = []
    handler.wfile = io.BytesIO()
    handler.send_header = MagicMock()

    DuoLibrasRequestHandler.end_headers(handler)

    sent_headers = {call.args[0]: call.args[1] for call in handler.send_header.call_args_list}
    assert sent_headers["Access-Control-Allow-Origin"] == "*"
    assert sent_headers["Cross-Origin-Opener-Policy"] == "same-origin"
    assert sent_headers["Cross-Origin-Embedder-Policy"] == "credentialless"
    assert sent_headers["X-Content-Type-Options"] == "nosniff"
    assert sent_headers["X-Frame-Options"] == "SAMEORIGIN"
    assert sent_headers["Cache-Control"] == "no-cache"


def test_request_handler_when_list_directory_then_sends_forbidden() -> None:
    handler = DuoLibrasRequestHandler.__new__(DuoLibrasRequestHandler)
    handler.send_error = MagicMock()

    result = DuoLibrasRequestHandler.list_directory(handler, "/some/path")

    assert result is None
    handler.send_error.assert_called_once_with(403, "Directory listing forbidden")


def test_request_handler_when_log_message_then_formats_without_raising() -> None:
    handler = DuoLibrasRequestHandler.__new__(DuoLibrasRequestHandler)
    handler.address_string = MagicMock(return_value="127.0.0.1")

    handler.log_message("GET /test/{id} %s", "200")


def test_request_handler_serves_training_video_with_range_and_blocks_traversal(tmp_path: Path) -> None:
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "index.html").write_text("ok", encoding="utf-8")

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    payload = b"0123456789ABCDEF" * 64
    (videos_dir / "Por favor_Articulador1.mp4").write_bytes(payload)
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")

    server = DuoLibrasWebServer(web_dir=web_dir, host="127.0.0.1", port=0, training_videos_dir=videos_dir)
    server.start(blocking=False)
    assert server.server is not None
    port = server.server.server_address[1]
    thread = threading.Thread(target=server.server.serve_forever, daemon=True)
    thread.start()

    try:
        url = f"http://127.0.0.1:{port}/training-videos/Por%20favor_Articulador1.mp4"
        req = urllib.request.Request(url, headers={"Range": "bytes=10-25"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 206
            assert resp.headers["Content-Type"] == "video/mp4"
            assert resp.headers["Accept-Ranges"] == "bytes"
            assert resp.headers["Content-Range"] == f"bytes 10-25/{len(payload)}"
            assert resp.read() == payload[10:26]

        traversal_url = f"http://127.0.0.1:{port}/training-videos/..%2Fsecret.txt"
        try:
            urllib.request.urlopen(traversal_url, timeout=5)
            raise AssertionError("Expected 404 for traversal")
        except urllib.error.HTTPError as err:
            assert err.code == 404
    finally:
        server.shutdown()
        thread.join(timeout=2)
