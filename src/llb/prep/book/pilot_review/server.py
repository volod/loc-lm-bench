"""Loopback-only HTTP interface for the page worksheet. No external assets or services."""

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
from urllib.parse import parse_qs, urlsplit

from llb.core.contracts.data_prep.book_blocks import BLOCK_KINDS
from llb.prep.book.pilot_review.session import Review, SaveRequest, StaleRevision

_ASSETS = {
    "/": ("worksheet.html", "text/html; charset=utf-8"),
    "/worksheet.css": ("worksheet.css", "text/css; charset=utf-8"),
    "/worksheet.js": ("worksheet.js", "text/javascript; charset=utf-8"),
}
_BODY_LIMIT = 8 * 1024 * 1024


class _ReviewServer(HTTPServer):
    def __init__(self, review: Review, port: int) -> None:
        self.review = review
        super().__init__(("127.0.0.1", port), WorksheetHandler)


class WorksheetHandler(BaseHTTPRequestHandler):
    server: _ReviewServer

    def do_GET(self) -> None:
        if not self._local_request():
            return
        try:
            self._get()
        except (FileNotFoundError, KeyError):
            self._json(HTTPStatus.NOT_FOUND, {"error": "Page or revision not found."})
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def do_POST(self) -> None:
        if not self._local_request():
            return
        try:
            self._save()
        except StaleRevision as exc:
            self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except (FileNotFoundError, KeyError):
            self._json(HTTPStatus.NOT_FOUND, {"error": "Page or revision not found."})
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _local_request(self) -> bool:
        host = f"127.0.0.1:{self.server.server_port}"
        if self.headers.get("Host") == host and self.headers.get("Origin") in (
            None,
            f"http://{host}",
        ):
            return True
        self._json(HTTPStatus.FORBIDDEN, {"error": "Open the printed loopback URL."})
        return False

    def _get(self) -> None:
        url = urlsplit(self.path)
        if url.path in _ASSETS:
            name, media = _ASSETS[url.path]
            self._reply(HTTPStatus.OK, media, files(__package__).joinpath(name).read_bytes())
        elif url.path == "/api/pages":
            self._json(HTTPStatus.OK, {"pages": self.server.review.status(), "kinds": BLOCK_KINDS})
        elif url.path.startswith("/api/page/"):
            page = url.path.removeprefix("/api/page/")
            version = parse_qs(url.query).get("version", [])
            if version:
                self._json(HTTPStatus.OK, self.server.review.earlier_page(page, int(version[0])))
            else:
                self._json(HTTPStatus.OK, self.server.review.worksheet(page))
        elif url.path.startswith("/scan/"):
            page = url.path.removeprefix("/scan/")
            self._reply(HTTPStatus.OK, "image/png", self.server.review.scan(page).read_bytes())
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown worksheet URL."})

    def _save(self) -> None:
        if not self.path.startswith("/api/page/"):
            raise ValueError("unknown worksheet write URL")
        if self.headers.get_content_type() != "application/json":
            raise ValueError("page revisions must be JSON")
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= _BODY_LIMIT:
            raise ValueError("page revision exceeds the request size limit or is empty")
        page = self.path.removeprefix("/api/page/")
        request = SaveRequest.model_validate_json(self.rfile.read(length))
        self._json(HTTPStatus.OK, {"version": self.server.review.save(page, request)})

    def _reply(self, status: HTTPStatus, media: str, content: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", media)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; frame-ancestors 'none'; base-uri 'none'",
        )
        self.end_headers()
        self.wfile.write(content)

    def _json(self, status: HTTPStatus, content: object) -> None:
        self._reply(status, "application/json", json.dumps(content).encode("utf-8"))


def serve(review: Review, port: int) -> None:
    """Serve one review locally until interrupted."""
    with _ReviewServer(review, port) as server:
        logging.getLogger(__name__).warning(
            "Open http://127.0.0.1:%d in a browser. Press Ctrl+C here to stop.",
            server.server_port,
        )
        server.serve_forever()
