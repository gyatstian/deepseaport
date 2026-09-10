"""Minimal MCP stdio client for Obscura (stdlib only)."""
import json
import subprocess
import threading


class McpClient:
    def __init__(self, binary, profile):
        self.proc = subprocess.Popen(
            [binary, "--storage-dir", profile, "mcp"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._id = 0
        self._lock = threading.Lock()
        self._pending = {}
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._init()

    def _read_loop(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            mid = msg.get("id")
            if mid is not None and mid in self._pending:
                self._pending[mid].append(msg)

    def _send(self, method, params=None):
        with self._lock:
            self._id += 1
            mid = self._id
            box = []
            self._pending[mid] = box
            self.proc.stdin.write(json.dumps(
                {"jsonrpc": "2.0", "id": mid, "method": method,
                 "params": params or {}}) + "\n")
            self.proc.stdin.flush()
        import time
        t0 = time.time()
        while time.time() - t0 < 120:
            if box:
                del self._pending[mid]
                msg = box[0]
                if "error" in msg:
                    raise RuntimeError(f"mcp {method}: {msg['error']}")
                return msg.get("result")
            time.sleep(0.1)
        del self._pending[mid]
        raise TimeoutError(f"mcp {method} timed out")

    def _init(self):
        self._send("initialize", {"protocolVersion": "2024-11-05",
                                  "capabilities": {},
                                  "clientInfo": {"name": "deepseaport", "version": "0.1"}})
        self.proc.stdin.write(json.dumps(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        self.proc.stdin.flush()

    def tools(self):
        return self._send("tools/list")

    def call(self, name, arguments):
        return self._send("tools/call", {"name": name, "arguments": arguments})

    def close(self):
        try:
            self.proc.terminate()
        except Exception:
            pass
