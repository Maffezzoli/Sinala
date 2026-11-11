"""HTTP request handler for the DuoLibras web interface."""

from __future__ import annotations

import io
import mimetypes
import os
import re
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, BinaryIO

from loguru import logger

_RANGE_HEADER_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
_TRAINING_VIDEOS_PREFIX = "/training-videos/"


class DuoLibrasRequestHandler(SimpleHTTPRequestHandler):
    """Custom request handler with CORS, security headers, HTTP Range, and training video routing."""

    def __init__(
        self,
        *args: Any,
        directory: str | None = None,
        training_videos_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        mimetypes.add_type("application/wasm", ".wasm")
        mimetypes.add_type("application/octet-stream", ".onnx")
        mimetypes.add_type("application/json", ".json")
        mimetypes.add_type("application/javascript", ".js")
        mimetypes.add_type("text/css", ".css")
        mimetypes.add_type("video/mp4", ".mp4")
        self.training_videos_dir = Path(training_videos_dir).resolve() if training_videos_dir is not None else Path("data/raw/greetings/videos").resolve()
        self._byte_range: tuple[int, int] | None = None
        super().__init__(*args, directory=directory, **kwargs)

    def translate_path(self, path: str) -> str:
        """Map `/training-videos/<file>` safely to `training_videos_dir` without traversal."""
        raw_url_path = urllib.parse.urlsplit(path).path
        decoded_path = urllib.parse.unquote(raw_url_path)

        if decoded_path.startswith(_TRAINING_VIDEOS_PREFIX):
            rel_part = decoded_path[len(_TRAINING_VIDEOS_PREFIX) :].lstrip("/")
            if not rel_part or "/" in rel_part or "\\" in rel_part or rel_part in {".", ".."}:
                return str(self.training_videos_dir / "__forbidden__")

            candidate = self.training_videos_dir / rel_part
            if candidate.is_symlink():
                return str(self.training_videos_dir / "__forbidden__")

            resolved = candidate.resolve()
            if not resolved.is_relative_to(self.training_videos_dir):
                return str(self.training_videos_dir / "__forbidden__")
            return str(resolved)

        return super().translate_path(path)

    def send_head(self) -> io.BytesIO | BinaryIO | None:
        """Serve static files with RFC 7233 single byte-range support for HTML5 video."""
        self._byte_range = None
        local_path = self.translate_path(self.path)

        range_header = self.headers.get("Range") if hasattr(self, "headers") and self.headers else None
        if os.path.isfile(local_path):
            ctype = self.guess_type(local_path)
            try:
                f = open(local_path, "rb")
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return None

            try:
                fs = os.fstat(f.fileno())
                file_size = fs.st_size

                if range_header:
                    parsed = self._parse_byte_range(range_header, file_size)
                    if parsed is None:
                        f.close()
                        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                        self.send_header("Content-Range", f"bytes */{file_size}")
                        self.end_headers()
                        return None

                    start, end = parsed
                    self._byte_range = (start, end)
                    self.send_response(HTTPStatus.PARTIAL_CONTENT)
                    self.send_header("Content-type", ctype)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(end - start + 1))
                    self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
                    self.end_headers()
                    return f

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(file_size))
                self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
                self.end_headers()
                return f
            except Exception:
                f.close()
                raise

        return super().send_head()

    def copyfile(self, source: BinaryIO, outputfile: BinaryIO) -> None:
        """Stream either the requested byte range or the complete file."""
        byte_range = getattr(self, "_byte_range", None)
        if byte_range is None:
            super().copyfile(source, outputfile)
            return

        start, end = byte_range
        remaining = end - start + 1
        source.seek(start)
        chunk_size = 64 * 1024
        while remaining > 0:
            chunk = source.read(min(chunk_size, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    @staticmethod
    def _parse_byte_range(range_header: str, file_size: int) -> tuple[int, int] | None:
        """Parse a single `bytes=start-end` header against `file_size`."""
        if file_size <= 0:
            return None
        match = _RANGE_HEADER_RE.match(range_header.strip())
        if not match:
            return None

        start_str, end_str = match.groups()
        if not start_str and not end_str:
            return None

        if not start_str:
            suffix_len = int(end_str)
            if suffix_len <= 0:
                return None
            start = max(0, file_size - suffix_len)
            end = file_size - 1
        else:
            start = int(start_str)
            end = int(end_str) if end_str else file_size - 1
            end = min(end, file_size - 1)

        if start < 0 or start > end or start >= file_size:
            return None
        return (start, end)

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
