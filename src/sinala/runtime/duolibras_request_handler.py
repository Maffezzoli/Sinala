"""HTTP request handler for the DuoLibras web interface."""

from __future__ import annotations

import mimetypes
from http.server import SimpleHTTPRequestHandler
from typing import Any

from loguru import logger


class DuoLibrasRequestHandler(SimpleHTTPRequestHandler):
    """Custom request handler with CORS, security headers, and MIME mappings."""

    def __init__(self, *args: Any, directory: str | None = None, **kwargs: Any) -> None:
        mimetypes.add_type("application/wasm", ".wasm")
        mimetypes.add_type("application/octet-stream", ".onnx")
        mimetypes.add_type("application/json", ".json")
        mimetypes.add_type("application/javascript", ".js")
        mimetypes.add_type("text/css", ".css")
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self) -> None:
        """Inject required CORS and cross-origin isolation headers for ONNX WASM."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def list_directory(self, path: str) -> None:
        """Disable directory listing for security."""
        self.send_error(403, "Directory listing forbidden")
        return None

    def log_message(self, format: str, *args: Any) -> None:
        """Forward access logs to Loguru safely avoiding format injection."""
        logger.opt(colors=False).debug("DuoLibras Web: {} - {}", self.address_string(), format % args)
