"""Unit tests for the DuoLibrasWebServer runtime component."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sinala.runtime.duolibras_web_server import DuoLibrasWebServer


def test_web_server_when_started_then_initializes_and_serves_forever(tmp_path: Path) -> None:
    web_dir = tmp_path / "web"
    web_dir.mkdir()

    server = DuoLibrasWebServer(web_dir=web_dir, host="127.0.0.1", port=8000)

    mock_http_server = MagicMock()
    with patch("sinala.runtime.duolibras_web_server.ThreadingHTTPServer", return_value=mock_http_server) as mock_cls:
        server.start(blocking=True)

        mock_cls.assert_called_once()
        args, _ = mock_cls.call_args
        assert args[0] == ("127.0.0.1", 8000)
        mock_http_server.serve_forever.assert_called_once()
        mock_http_server.shutdown.assert_called_once()
        mock_http_server.server_close.assert_called_once()


def test_web_server_when_shutdown_called_then_closes_socket(tmp_path: Path) -> None:
    web_dir = tmp_path / "web"
    web_dir.mkdir()

    server = DuoLibrasWebServer(web_dir=web_dir)
    mock_http_server = MagicMock()
    server.server = mock_http_server

    server.shutdown()

    mock_http_server.shutdown.assert_called_once()
    mock_http_server.server_close.assert_called_once()
    assert server.server is None


def test_web_server_when_missing_dir_then_raises_file_not_found(tmp_path: Path) -> None:
    non_existent = tmp_path / "missing_dir"
    server = DuoLibrasWebServer(web_dir=non_existent, port=8000)
    with pytest.raises(FileNotFoundError, match="Diretorio web nao encontrado"):
        server.start(blocking=False)
