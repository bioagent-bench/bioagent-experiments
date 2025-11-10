#!/usr/bin/env python3
"""
Minimal OTLP gRPC log sink for Codex.
Accepts OTLP/gRPC logs on <host>:<port> and appends newline-delimited JSON.
This is NOT a full collector; it's perfect for local capture/analysis.

Run as a module:
  python -m otel --host 127.0.0.1 --port 4317 --path /path/to/out.ndjson
"""

from __future__ import annotations

from concurrent import futures
from datetime import datetime, timezone
import argparse
import json
import os
import signal
import sys
from typing import Any, Dict

import grpc
from google.protobuf.json_format import MessageToDict

from opentelemetry.proto.collector.logs.v1 import logs_service_pb2_grpc, logs_service_pb2


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _msg_to_dict(msg) -> Dict[str, Any]:
    """Compatibility wrapper: protobuf has used both singular/plural kw names."""
    return MessageToDict(msg, preserving_proto_field_name=True)


def _extract_quick_fields(body_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull a few helpful fields to top level for quick jq/duckdb queries.
    Safe if structure changes—everything else stays under 'body'.
    """
    out: Dict[str, Any] = {}
    try:
        rlogs = body_dict.get("resource_logs", [])
        if not rlogs:
            return out
        res = rlogs[0].get("resource", {})
        attrs = {
            a.get("key"): a.get("value", {}).get(list(a.get("value", {}).keys())[0])
            for a in res.get("attributes", [])
            if "key" in a and "value" in a
        }
        out["service.name"] = attrs.get("service.name")
        out["env"] = attrs.get("env")

        scope_logs = rlogs[0].get("scope_logs", [])
        if scope_logs:
            logs = scope_logs[0].get("log_records", [])
            if logs:
                lr = logs[0]
                lattrs = {
                    a.get("key"): a.get("value", {}).get(list(a.get("value", {}).keys())[0])
                    for a in lr.get("attributes", [])
                    if "key" in a and "value" in a
                }
                out["event.name"] = lattrs.get("event.name") or lattrs.get("event")
                out["model"] = lattrs.get("model")
                out["originator"] = lattrs.get("originator")
    except Exception:
        pass
    return out


class LogSinkServicer(logs_service_pb2_grpc.LogsServiceServicer):
    def __init__(self, out_path: str):
        self.out_path = out_path
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    def Export(self, request: logs_service_pb2.ExportLogsServiceRequest, context):
        body = _msg_to_dict(request)
        envelope = {
            "received_at": _now_iso(),
            "kind": "otlp_logs_export",
            **_extract_quick_fields(body),
            "body": body,
        }
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(envelope, ensure_ascii=False) + "\n")
        return logs_service_pb2.ExportLogsServiceResponse()


def run_server(host: str, ndjson_path: str):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ],
    )
    logs_service_pb2_grpc.add_LogsServiceServicer_to_server(LogSinkServicer(ndjson_path), server)
    bind_addr = f"{host}"
    server.add_insecure_port(bind_addr)
    server.start()
    print(f"[mini-otel] listening for OTLP/gRPC logs on {bind_addr}", file=sys.stderr)
    print(f"[mini-otel] writing NDJSON to {ndjson_path}", file=sys.stderr)
    return server


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal OTLP gRPC log sink (NDJSON).")
    parser.add_argument("--host")
    parser.add_argument("--path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    server = run_server(args.host, args.path)

    def _graceful(signum, frame):
        print("[mini-otel] shutting down…", file=sys.stderr)
        server.stop(grace=None)
        sys.exit(0)

    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
