"""Unit tests for the DuoLibrasRequestHandler component."""

import io
from unittest.mock import MagicMock

from sinala.runtime.duolibras_request_handler import DuoLibrasRequestHandler


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
