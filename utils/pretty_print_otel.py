#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional


STANDARD_ATTRS = {
    "event.name",
    "event.timestamp",
    "event.kind",
    "conversation.id",
    "app.version",
    "terminal.type",
    "model",
    "slug",
    "duration_ms",
    "tool_name",
    "call_id",
    "arguments",
    "success",
    "output",
    "decision",
    "source",
    "http.response.status_code",
    "attempt",
    "prompt",
    "prompt_length",
    "provider_name",
    "reasoning_effort",
    "reasoning_summary",
    "approval_policy",
    "sandbox_policy",
    "mcp_servers",
    "active_profile",
}

RESPONSE_KEYS = (
    "message",
    "content",
    "delta",
    "data",
    "text",
    "response",
    "assistant",
)


def decode_attr_value(value: Dict[str, Any]) -> Any:
    if "string_value" in value:
        return value["string_value"]
    if "int_value" in value:
        return value["int_value"]
    if "bool_value" in value:
        return value["bool_value"]
    if "double_value" in value:
        return value["double_value"]
    if value:
        return next(iter(value.values()))
    return None


def attrs_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    attrs = {}
    for attr in record.get("attributes", []):
        attrs[attr.get("key")] = decode_attr_value(attr.get("value", {}))
    return attrs


def parse_json_objects(raw: Optional[str]) -> List[Any]:
    if not raw:
        return []
    raw = raw.strip()
    if not raw:
        return []
    decoder = json.JSONDecoder()
    idx = 0
    objs = []
    while idx < len(raw):
        try:
            obj, end = decoder.raw_decode(raw, idx)
        except json.JSONDecodeError:
            break
        objs.append(obj)
        idx = end
        while idx < len(raw) and raw[idx].isspace():
            idx += 1
    if not objs:
        return []
    return objs


def format_command(cmd: Any) -> str:
    if isinstance(cmd, list):
        return " ".join(shlex.quote(str(part)) for part in cmd)
    return str(cmd)


def extract_response_content(attrs: Dict[str, Any]) -> Optional[str]:
    for key in RESPONSE_KEYS:
        value = attrs.get(key)
        if value:
            return str(value)
    extra_keys = [k for k in attrs.keys() if k not in STANDARD_ATTRS]
    if len(extra_keys) == 1:
        return str(attrs.get(extra_keys[0]))
    return None


def iter_log_records(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for resource in obj.get("body", {}).get("resource_logs", []):
                for scope in resource.get("scope_logs", []):
                    for record in scope.get("log_records", []):
                        yield record


def build_report(path: str) -> Dict[str, Any]:
    responses = []
    prompt_text = None
    tool_summary = defaultdict(lambda: Counter())
    tool_calls = []
    commands = []

    for record in iter_log_records(path):
        attrs = attrs_from_record(record)
        event_name = attrs.get("event.name")
        timestamp = attrs.get("event.timestamp")

        if event_name == "codex.user_prompt":
            prompt_text = attrs.get("prompt")
            continue

        if event_name == "codex.sse_event":
            content = extract_response_content(attrs)
            if content:
                responses.append({"timestamp": timestamp, "content": content})
            continue

        if event_name == "codex.tool_decision":
            tool_name = attrs.get("tool_name") or "unknown"
            tool_summary[tool_name]["decisions"] += 1
            decision = attrs.get("decision")
            if decision:
                tool_summary[tool_name][f"decision_{decision}"] += 1
            continue

        if event_name == "codex.tool_result":
            tool_name = attrs.get("tool_name") or "unknown"
            tool_summary[tool_name]["calls"] += 1
            success = attrs.get("success")
            if success == "true" or success is True:
                tool_summary[tool_name]["success"] += 1
            elif success == "false" or success is False:
                tool_summary[tool_name]["failure"] += 1

            args_raw = attrs.get("arguments")
            parsed_args = parse_json_objects(args_raw)
            call_id = attrs.get("call_id")
            tool_calls.append(
                {
                    "timestamp": timestamp,
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "success": success,
                    "arguments": parsed_args if parsed_args else args_raw,
                }
            )

            if tool_name == "shell":
                for arg in parsed_args or []:
                    cmd = arg.get("command") if isinstance(arg, dict) else None
                    workdir = arg.get("workdir") if isinstance(arg, dict) else None
                    commands.append(
                        {
                            "timestamp": timestamp,
                            "command": format_command(cmd) if cmd else None,
                            "workdir": workdir,
                            "success": success,
                            "call_id": call_id,
                        }
                    )
            continue

    summary = {name: dict(counter) for name, counter in tool_summary.items()}
    return {
        "prompt": prompt_text,
        "responses": responses,
        "tool_summary": summary,
        "tool_calls": tool_calls,
        "commands": commands,
    }


def render_text_report(
    report: Dict[str, Any], include_tool_output: bool, include_prompt: bool
) -> str:
    responses = report["responses"]
    prompt = report["prompt"]
    tool_summary = report["tool_summary"]
    tool_calls = report["tool_calls"]
    commands = report["commands"]

    lines = []
    lines.append("Model responses")
    if responses:
        for idx, item in enumerate(responses, 1):
            ts = item.get("timestamp") or "unknown_time"
            lines.append(f"[{idx}] {ts} {item.get('content')}")
    else:
        lines.append("No response content found in this OTLP file.")

    if include_prompt and prompt:
        lines.append("")
        lines.append("User prompt")
        lines.append(prompt)

    lines.append("")
    lines.append("Tools used (summary)")
    if tool_summary:
        for tool_name in sorted(tool_summary.keys()):
            summary = tool_summary[tool_name]
            calls = summary.get("calls", 0)
            success = summary.get("success", 0)
            failure = summary.get("failure", 0)
            decisions = summary.get("decisions", 0)
            approved = summary.get("decision_approved", 0)
            denied = summary.get("decision_denied", 0)
            parts = [f"{calls} calls", f"{success} success", f"{failure} failure"]
            if decisions:
                parts.append(f"{approved} approved")
                parts.append(f"{denied} denied")
            lines.append(f"- {tool_name}: " + ", ".join(parts))
    else:
        lines.append("No tool usage found.")

    if commands:
        lines.append("")
        lines.append("Commands run (shell tool)")
        for entry in commands:
            ts = entry.get("timestamp") or "unknown_time"
            cmd = entry.get("command") or "<unparsed command>"
            workdir = entry.get("workdir") or "<unknown workdir>"
            success = entry.get("success")
            status = "success" if success == "true" or success is True else "failure"
            lines.append(f"- {ts} [{status}] {cmd} (workdir={workdir})")

    if include_tool_output:
        lines.append("")
        lines.append("Tool calls (details)")
        for call in tool_calls:
            ts = call.get("timestamp") or "unknown_time"
            tool_name = call.get("tool_name") or "unknown"
            call_id = call.get("call_id") or "unknown"
            success = call.get("success")
            status = "success" if success == "true" or success is True else "failure"
            lines.append(f"- {ts} [{status}] {tool_name} call_id={call_id}")
            args = call.get("arguments")
            if args is not None:
                lines.append(f"  arguments: {args}")
    return "\n".join(lines) + "\n"


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretty-print Codex OTLP NDJSON to show responses, tools, and commands."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="utils/otlp-322882de-0a2e-4f68-ab0e-f67a019bbf11.ndjson",
        help="Path to OTLP NDJSON file or a directory of NDJSON files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output a JSON report instead of text.",
    )
    parser.add_argument(
        "--out",
        help="Write report to this file (defaults to <input>.pretty.txt or .pretty.json).",
    )
    parser.add_argument(
        "--out-dir",
        help="Output directory for reports (defaults to <input_dir>/prettied).",
    )
    parser.add_argument(
        "--include-tool-output",
        action="store_true",
        help="Include detailed tool call arguments in text output.",
    )
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="Include the captured user prompt in text output.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    input_path = args.path
    suffix = ".pretty.json" if args.json else ".pretty.txt"

    def build_content(path: str) -> str:
        report = build_report(path)
        if args.json:
            return json.dumps(report, indent=2, sort_keys=True) + "\n"
        return render_text_report(
            report,
            include_tool_output=args.include_tool_output,
            include_prompt=args.include_prompt,
        )

    if os.path.isdir(input_path):
        output_dir = args.out or args.out_dir or os.path.join(input_path, "prettied")
        os.makedirs(output_dir, exist_ok=True)
        processed = 0
        for name in sorted(os.listdir(input_path)):
            if not name.endswith(".ndjson"):
                continue
            src_path = os.path.join(input_path, name)
            if not os.path.isfile(src_path):
                continue
            content = build_content(src_path)
            out_path = os.path.join(output_dir, f"{name}{suffix}")
            with open(out_path, "w", encoding="utf-8") as handle:
                handle.write(content)
            processed += 1
        print(f"Wrote {processed} report(s) to {output_dir}")
        return 0

    content = build_content(input_path)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        filename = os.path.basename(input_path) + suffix
        output_path = os.path.join(args.out_dir, filename)
    else:
        output_path = args.out or f"{input_path}{suffix}"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    print(f"Wrote report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
