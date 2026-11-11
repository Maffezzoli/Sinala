"""Lightweight local web server for the DuoLibras interactive web application."""

from __future__ import annotations

import functools
from http.server import ThreadingHTTPServer
from pathlib import Path

from loguru import logger

from sinala.runtime.duolibras_request_handler import DuoLibrasRequestHandler


class DuoLibrasWebServer:
    """Serves the static web application with proper MIME types for WASM and ONNX."""

    def __init__(
        self,
        web_dir: str | Path = "web",
        host: str = "127.0.0.1",
        port: int = 8000,
        training_videos_dir: str | Path = "data/raw/greetings/videos",
    ) -> None:
        self.web_dir = Path(web_dir).resolve()
        self.training_videos_dir = Path(training_videos_dir).resolve()
        self.host = host
        self.port = port
        self.server: ThreadingHTTPServer | None = None

    def start(self, blocking: bool = True) -> None:
        """Start the HTTP server on configured host and port."""
        if not self.web_dir.exists():
            raise FileNotFoundError(f"Diretorio web nao encontrado: {self.web_dir}")

        handler_factory = functools.partial(
            DuoLibrasRequestHandler,
            directory=str(self.web_dir),
            training_videos_dir=str(self.training_videos_dir),
        )
        self.server = ThreadingHTTPServer((self.host, self.port), handler_factory)
        logger.info(f"DuoLibras Web Server rodando em http://{self.host}:{self.port} (diretorio: {self.web_dir})")

        if blocking:
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                logger.info("Encerrando DuoLibras Web Server...")
            finally:
                self.shutdown()

    def shutdown(self) -> None:
        """Gracefully stop and close the server."""
        server = self.server
        self.server = None
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
            logger.info("DuoLibras Web Server finalizado.")
