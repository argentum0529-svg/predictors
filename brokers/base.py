"""Curl client for the brokers service (local, or ssh then curl)."""

from __future__ import annotations

import json
import shlex
import socket
import subprocess
from typing import Any
from urllib.parse import urlencode


class BrokerError(RuntimeError):
    """curl or ssh to the brokers service failed."""


class BrokerClient:
    """GET JSON from the brokers service on loopback.

    If this machine's short hostname matches ``host``, curl locally.
    Otherwise ssh to ``host`` and curl loopback there. The brokers
    process binds 127.0.0.1 and is not reachable from another host.
    """

    def __init__(self, host: str, port: int, timeout: int = 120) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    def on_host(self) -> bool:
        name = socket.gethostname().split(".")[0]
        return name.lower() == self.host.lower()

    def curl_argv(self, path: str, params: dict[str, str] | None = None) -> list[str]:
        query = urlencode(params or {})
        url = f"http://127.0.0.1:{self.port}{path}"
        if query:
            url = f"{url}?{query}"
        curl = ["curl", "-fsS", "--max-time", str(self.timeout), url]
        if self.on_host():
            return curl
        remote = " ".join(shlex.quote(part) for part in curl)
        return ["ssh", "-o", "BatchMode=yes", self.host, remote]

    def get(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        argv = self.curl_argv(path, params)
        result = subprocess.run(argv, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise BrokerError(f"broker request failed ({result.returncode}): {detail[:500]}")
        try:
            body = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise BrokerError(f"broker response was not JSON: {result.stdout[:300]}") from exc
        if not isinstance(body, dict):
            raise BrokerError("broker response was not a JSON object")
        return body
