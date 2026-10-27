import importlib.util
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

# Load the helper without running qd_build's interpreter-restarting bootstrap.
_path = Path(__file__).resolve().parents[2] / ".github/workflows/scripts/qd_build/download.py"
_spec = importlib.util.spec_from_file_location("build_download", _path)
download = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(download)


@pytest.fixture
def delays(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr(download.time, "sleep", sleep)
    return sleep


def test_interrupted_body_restarts_download(tmp_path, delays):
    body = b"complete archive" * 2048
    attempts = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            attempts.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body[:9000] if len(attempts) == 1 else body)
            self.close_connection = True

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = tmp_path / "archive"
    try:
        download.download_file(f"http://127.0.0.1:{server.server_port}/archive", destination, "archive")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert destination.read_bytes() == body
    assert len(attempts) == 2
    delays.assert_called_once_with(1)
    assert not (tmp_path / "archive.download").exists()


@pytest.mark.parametrize(
    "error_type", [requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError]
)
def test_exhaustion_preserves_cache_and_removes_partial(tmp_path, monkeypatch, delays, error_type):
    destination = tmp_path / "archive"
    destination.write_bytes(b"old cache")
    partial = tmp_path / "archive.download"

    def fail(*args, **kwargs):
        partial.write_bytes(b"incomplete")
        raise error_type("interrupted")

    get = Mock(side_effect=fail)
    monkeypatch.setattr(download.requests, "get", get)
    with pytest.raises(error_type, match="interrupted"):
        download.download_file("https://example.com/archive", destination, "archive")
    assert get.call_count == 5
    assert [call.args[0] for call in delays.call_args_list] == [1, 2, 4, 8]
    assert destination.read_bytes() == b"old cache"
    assert not partial.exists()


@pytest.mark.parametrize(
    "status, attempts", [(404, 1), (403, 1), (408, 5), (429, 5), (500, 5), (502, 5), (503, 5), (504, 5)]
)
def test_http_status_retry_policy(tmp_path, monkeypatch, delays, status, attempts):
    response = requests.Response()
    response.status_code = status
    get = Mock(side_effect=requests.HTTPError("HTTP failure", response=response))
    monkeypatch.setattr(download.requests, "get", get)
    with pytest.raises(requests.HTTPError):
        download.download_file("https://example.com/archive", tmp_path / "archive", "archive")
    assert get.call_count == attempts
    assert delays.call_count == attempts - 1
    assert not (tmp_path / "archive").exists()


def test_disk_error_is_not_retried(tmp_path, monkeypatch, delays):
    get = Mock(side_effect=OSError("disk failure"))
    monkeypatch.setattr(download.requests, "get", get)
    with pytest.raises(OSError, match="disk failure"):
        download.download_file("https://example.com/archive", tmp_path / "archive", "archive")
    get.assert_called_once()
    delays.assert_not_called()
