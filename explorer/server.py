#!/usr/bin/env python3
"""PurePacketPPSeq Model Explorer Server

Lightweight, dependency-free local web server for interactive exploration of
point-process motif runs, continuous data streams, and multi-unit spike rasters.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import sys
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any

EXPLORER_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPLORER_DIR.parent
RUNS_DIR = REPO_ROOT / "runs"


class ExplorerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(EXPLORER_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # 1. Main visualizer page
        if path in ("/", "/index.html", "/explorer"):
            self.send_file(EXPLORER_DIR / "index.html", "text/html; charset=utf-8")
            return

        # 2. List all available runs in runs/
        if path == "/api/runs":
            runs_list = []
            if RUNS_DIR.exists():
                for entry in sorted(RUNS_DIR.iterdir()):
                    if entry.is_dir() and (entry / "manifest.json").exists():
                        try:
                            with open(entry / "manifest.json", "r", encoding="utf-8") as f:
                                meta = json.load(f)
                            runs_list.append({
                                "id": meta.get("run_id", entry.name),
                                "name": meta.get("run_name", entry.name),
                                "session": meta.get("session", ""),
                                "duration": meta.get("duration", 0),
                                "n_units": meta.get("n_units", 0),
                                "n_spikes": meta.get("n_spikes", 0),
                                "n_motifs": meta.get("n_motifs", 0),
                                "n_events": meta.get("n_events", 0),
                                "streams": meta.get("active_streams", [])
                            })
                        except Exception as e:
                            print(f"[Explorer] Error reading manifest in {entry.name}: {e}")
            self.send_json(runs_list)
            return

        # 3. Serve run data: /api/run/<run_id>/<file>
        if path.startswith("/api/run/"):
            parts = [p for p in path.split("/") if p]
            # Expected format: ['api', 'run', run_id, endpoint]
            if len(parts) >= 4:
                run_id = parts[2]
                endpoint = parts[3]
                run_dir = RUNS_DIR / run_id

                if not run_dir.exists():
                    self.send_error(404, f"Run not found: {run_id}")
                    return

                if endpoint == "manifest":
                    self.send_file(run_dir / "manifest.json", "application/json")
                    return
                elif endpoint == "events":
                    self.send_file(run_dir / "events.json", "application/json")
                    return
                elif endpoint == "streams":
                    self.send_file(run_dir / "streams.json", "application/json")
                    return
                elif endpoint == "spikes":
                    spikes_path = run_dir / "spikes.json"
                    if not spikes_path.exists():
                        self.send_error(404, "spikes.json not found")
                        return

                    t0_str = query.get("t0", [None])[0]
                    t1_str = query.get("t1", [None])[0]
                    if t0_str is not None and t1_str is not None:
                        try:
                            t0 = float(t0_str)
                            t1 = float(t1_str)
                            with open(spikes_path, "r", encoding="utf-8") as f:
                                spk = json.load(f)
                            ts = spk.get("t", [])
                            cs = spk.get("c", [])
                            filtered_t = []
                            filtered_c = []
                            for t_val, c_val in zip(ts, cs):
                                if t0 <= t_val <= t1:
                                    filtered_t.append(t_val)
                                    filtered_c.append(c_val)
                            self.send_json({"t": filtered_t, "c": filtered_c, "total": len(filtered_t)})
                            return
                        except Exception as e:
                            print(f"[Explorer] Error slicing spikes: {e}")

                    # Default: send full spikes overview
                    self.send_file(spikes_path, "application/json")
                    return

        # 4. Fallback to default static file handling
        super().do_GET()

    def send_file(self, filepath: Path, content_type: str) -> None:
        if not filepath.exists():
            self.send_error(404, f"File not found: {filepath.name}")
            return
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def send_json(self, data: Any) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep server logging clean
        if "/api/run/" in args[0] and "200" in args[1]:
            return
        super().log_message(format, *args)


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description="PurePacketPPSeq Model Explorer Server")
    parser.add_argument("--port", "-p", type=int, default=8050, help="Port to listen on (default: 8050)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    port = args.port
    url = f"http://localhost:{port}"

    print("=" * 75)
    print(" PurePacketPPSeq Model Explorer")
    print(f" Web Interface:  {url}")
    print(f" Runs Directory: {RUNS_DIR}")
    print("=" * 75)

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    with ReusableTCPServer(("", port), ExplorerHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down Model Explorer server.")


if __name__ == "__main__":
    main()
