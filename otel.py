#!/usr/bin/env python3
"""
Minimal OTLP gRPC log sink for Codex.
Accepts OTLP/gRPC logs on <host>:<port> and appends newline-delimited JSON.
This is NOT a full collector; it's perfect for local capture/analysis.

Run as a module (single run):
  python -m otel --host 127.0.0.1:4317 --path /path/to/out.ndjson

Multi-run (per run_hash file in a directory):
  python -m otel --host 127.0.0.1:4317 --path /path/to/otel-dir --mode multi
"""

from __future__ import annotations

from concurrent import futures
from datetime import datetime, timezone
import argparse
import json
import os
import signal
import sys
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, Tuple

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
            if "key" in a and "value" in a and a.get("value")
        }
        out["service.name"] = attrs.get("service.name")
        out["env"] = attrs.get("env")
        if attrs.get("run_hash"):
            out["run_hash"] = attrs.get("run_hash")

        scope_logs = rlogs[0].get("scope_logs", [])
        if scope_logs:
            logs = scope_logs[0].get("log_records", [])
            if logs:
                lr = logs[0]
                lattrs = {
                    a.get("key"): a.get("value", {}).get(list(a.get("value", {}).keys())[0])
                    for a in lr.get("attributes", [])
                    if "key" in a and "value" in a and a.get("value")
                }
                out["event.name"] = lattrs.get("event.name") or lattrs.get("event")
                out["model"] = lattrs.get("model")
                out["originator"] = lattrs.get("originator")
                if not out.get("run_hash") and lattrs.get("run_hash"):
                    out["run_hash"] = lattrs.get("run_hash")
    except Exception:
        # Best-effort — never break ingestion.
        pass
    return out


# token aggregation

def _to_int(x: Any) -> int:
    """Coerce potential numeric strings/floats to int safely; returns 0 if not numeric."""
    try:
        if isinstance(x, bool):
            return 0
        if isinstance(x, (int,)):
            return int(x)
        if isinstance(x, float):
            return int(x)
        if isinstance(x, str):
            # Allow things like "123", "123.0"
            if x.strip().isdigit():
                return int(x.strip())
            return int(float(x.strip()))
    except Exception:
        return 0
    return 0


def _iter_log_records(envelope: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield log_record dicts from a decoded envelope line."""
    body = envelope.get("body") or {}
    for r in body.get("resource_logs", []) or []:
        for s in r.get("scope_logs", []) or []:
            for lr in s.get("log_records", []) or []:
                yield lr


def _attrs_to_dict(attrs_list: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten OTel attribute list into a simple dict."""
    out: Dict[str, Any] = {}
    for a in attrs_list or []:
        key = a.get("key")
        val_obj = a.get("value") or {}
        # 'value' has one of: string_value, int_value, double_value, bool_value, bytes_value
        if key and isinstance(val_obj, dict) and val_obj:
            # grab the only value present
            val = val_obj.get(next(iter(val_obj)))
            out[key] = val
    return out


def _extract_run_hash(body_dict: Dict[str, Any]) -> str | None:
    """Try to pull a run identifier from resource or log attributes."""

    def _from_attrs(attrs: Iterable[Dict[str, Any]] | None) -> str | None:
        values = _attrs_to_dict(attrs or [])
        for key in ("run_hash", "run.id", "runId", "codex.run_hash", "codex_run_hash"):
            raw = values.get(key)
            if raw is None:
                continue
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
            return str(raw)
        return None

    for resource_log in body_dict.get("resource_logs", []) or []:
        resource = resource_log.get("resource") or {}
        run_hash = _from_attrs(resource.get("attributes"))
        if run_hash:
            return run_hash
        for scope_log in resource_log.get("scope_logs", []) or []:
            for log_record in scope_log.get("log_records", []) or []:
                run_hash = _from_attrs(log_record.get("attributes"))
                if run_hash:
                    return run_hash
    return None


def _sanitize_run_hash(value: str | None) -> str:
    if not value:
        return "unknown"
    safe = [c if c.isalnum() or c in ("-", "_") else "_" for c in value]
    cleaned = "".join(safe).strip("_")
    return cleaned or "unknown"


def _extract_usage_from_record(lr: Dict[str, Any]) -> Tuple[int, int]:
    """
    Attempt to extract (input_tokens, output_tokens) from a single log record,
    being resilient to different key shapes:
      - attributes: input_token_count / output_token_count
      - attributes: input_tokens / output_tokens
      - attributes: prompt_tokens / completion_tokens
      - attributes: usage.prompt_tokens / usage.completion_tokens (flattened)
      - body.kv-like payloads that contain 'usage' dicts (rare but supported)
    """
    in_tok = out_tok = 0

    lattrs = _attrs_to_dict(lr.get("attributes", []))

    flat = {str(k): lattrs[k] for k in lattrs}

    candidates = [
        ("input_token_count", "output_token_count")
    ]

    for k_in, k_out in candidates:
        if k_in in flat or k_out in flat:
            in_tok += _to_int(flat.get(k_in, 0))
            out_tok += _to_int(flat.get(k_out, 0))

    lr_body = lr.get("body")
    if isinstance(lr_body, dict):
        if "string_value" in lr_body:
            try:
                maybe = json.loads(lr_body.get("string_value") or "{}")
                if isinstance(maybe, dict):
                    usage = maybe.get("usage") or maybe.get("token_usage") or {}
                    in_tok += _to_int(usage.get("prompt_tokens"))
                    out_tok += _to_int(usage.get("completion_tokens"))
                    in_tok += _to_int(maybe.get("input_token_count"))
                    out_tok += _to_int(maybe.get("output_token_count"))
            except Exception:
                pass
        elif "kvlist_value" in lr_body:
            try:
                kvvals = lr_body["kvlist_value"].get("values", [])
                inner = _attrs_to_dict(kvvals)
                usage = inner.get("usage") if isinstance(inner.get("usage"), dict) else {}
                if isinstance(usage, dict):
                    in_tok += _to_int(usage.get("prompt_tokens"))
                    out_tok += _to_int(usage.get("completion_tokens"))
                in_tok += _to_int(inner.get("input_token_count"))
                out_tok += _to_int(inner.get("output_token_count"))
                in_tok += _to_int(inner.get("input_tokens"))
                out_tok += _to_int(inner.get("output_tokens"))
            except Exception:
                pass

    return in_tok, out_tok


def sum_token_counts(ndjson_path: str | Path) -> Tuple[int, int]:
    """
    Scan the NDJSON log produced by this sink and return the total
    (input_tokens, output_tokens) observed across the conversation.

    This is resilient to different key names and payload structures.
    """
    ndjson_path = Path(ndjson_path)
    if not ndjson_path.exists():
        return 0, 0

    total_in = 0
    total_out = 0

    with ndjson_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                envelope = json.loads(line)
            except Exception:
                continue

            for lr in _iter_log_records(envelope):
                i, o = _extract_usage_from_record(lr)
                total_in += i
                total_out += o

    return total_in, total_out


# gRPC server

class LogSinkServicer(logs_service_pb2_grpc.LogsServiceServicer):
    def __init__(self, out_path: str, multi_run: bool = False):
        self.out_path = Path(out_path)
        self.multi_run = multi_run
        if self.multi_run:
            self.out_path.mkdir(parents=True, exist_ok=True)
        else:
            os.makedirs(self.out_path.parent if self.out_path.parent else Path("."), exist_ok=True)
        self._lock = Lock()
        self._file_locks: dict[Path, Lock] = {}

    def _get_lock(self, path: Path) -> Lock:
        with self._lock:
            lock = self._file_locks.get(path)
            if lock is None:
                lock = Lock()
                self._file_locks[path] = lock
            return lock

    def _target_path(self, run_hash: str | None) -> Path:
        if not self.multi_run:
            return self.out_path
        safe_hash = _sanitize_run_hash(run_hash)
        return self.out_path / f"otlp-{safe_hash}.ndjson"

    def Export(self, request: logs_service_pb2.ExportLogsServiceRequest, context):
        body = _msg_to_dict(request)
        run_hash = _extract_run_hash(body)
        envelope = {
            "received_at": _now_iso(),
            "kind": "otlp_logs_export",
            **_extract_quick_fields(body),
            "body": body,
        }
        if run_hash:
            envelope.setdefault("run_hash", run_hash)
        target = self._target_path(run_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        lock = self._get_lock(target)
        with lock:
            with open(target, "a", encoding="utf-8") as f:
                f.write(json.dumps(envelope, ensure_ascii=False) + "\n")
        return logs_service_pb2.ExportLogsServiceResponse()


def run_server(host: str, ndjson_path: str, mode: str = "single"):
    multi_run = mode.lower() == "multi"
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ],
    )
    logs_service_pb2_grpc.add_LogsServiceServicer_to_server(
        LogSinkServicer(ndjson_path, multi_run=multi_run), server
    )
    bind_addr = f"{host}"
    server.add_insecure_port(bind_addr)
    server.start()
    mode_label = "multi-run" if multi_run else "single-run"
    print(f"[mini-otel] listening for OTLP/gRPC logs on {bind_addr}", file=sys.stderr)
    print(
        f"[mini-otel] {mode_label} mode -> writing NDJSON to {ndjson_path}", file=sys.stderr
    )
    return server


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal OTLP gRPC log sink (NDJSON).")
    parser.add_argument("--host")
    parser.add_argument("--path")
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    server = run_server(args.host, args.path, mode=args.mode)

    def _graceful(signum, frame):
        print("[mini-otel] shutting down…", file=sys.stderr)
        server.stop(grace=None)
        sys.exit(0)

    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
