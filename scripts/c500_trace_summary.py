#!/usr/bin/env python3
"""Summarize bounded Chrome/Kineto traces for the four C500 runtime phases.

The parser runs only on a live C500 target, never imports project/vendor Python
packages, never copies trace arguments, and never synchronizes the device.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


LOCAL_SOURCE_ROOT = Path("/home/brave/flagos")
C500_PATTERN = re.compile(r"(?i)(?:\bmetax\b.*\bc500\b|\bmxc?500\b|\bc500\b)")
SAFE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")

CATEGORY_PATTERNS = (
    (
        "sort",
        re.compile(
            r"(?i)(?:aten::)?sort\b|radix|histogram|compute_global_hist|(?:^|\W)sweep(?:$|\W)"
        ),
    ),
    ("synchronization", re.compile(r"(?i)(synchroni[sz]|device.?to.?host|to\.list|\.item\b)")),
    ("memory_copy", re.compile(r"(?i)(memcpy|memory copy|device.*pageable|dtoh|htod|copy_)")),
    ("allocation", re.compile(r"(?i)(alloc|empty|new_zeros|zeros|malloc|free)")),
    ("attention", re.compile(r"(?i)(attention|flash_attn|paged_attn|mha_|fmha)")),
    ("gemm", re.compile(r"(?i)(gemm|matmul|mm\b|linear)")),
    ("rope", re.compile(r"(?i)(rotary|rope)")),
    ("rmsnorm", re.compile(r"(?i)(rms.?norm)")),
    ("silu_gate", re.compile(r"(?i)(silu|swiglu|sigmoid.*mul)")),
    ("scheduler_metadata", re.compile(r"(?i)(scheduler|metadata|cumsum|cu_seq|prefix)")),
)
DIAGNOSTIC_PATTERNS = {
    "device_to_pageable": re.compile(r"(?i)(device.*pageable|dtoh|device.?to.?host)"),
    "cumsum": re.compile(r"(?i)cumsum"),
    "tolist_or_item": re.compile(r"(?i)(to\.list|\.item\b)"),
    "explicit_sync": re.compile(r"(?i)synchroni[sz]"),
    "graph_capture_or_replay": re.compile(r"(?i)(graph.*(?:capture|replay)|(?:capture|replay).*graph)"),
    "graph_break": re.compile(r"(?i)graph.?break"),
}


class TraceError(RuntimeError):
    pass


def die(message: str) -> "NoReturn":
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def absolute_existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    if not path.is_file() or path.is_symlink():
        raise argparse.ArgumentTypeError(f"file is missing or is a symlink: {path}")
    return path


def absolute_existing_dir(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    if not path.is_dir() or path.is_symlink():
        raise argparse.ArgumentTypeError(f"directory is missing or is a symlink: {path}")
    return path


def absolute_new_file(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    if path.exists() or path.is_symlink():
        raise argparse.ArgumentTypeError(f"output already exists: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise argparse.ArgumentTypeError(f"output parent is missing or is a symlink: {path.parent}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize explicit bounded C500 traces using explicit four-phase windows."
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--repo", required=True, type=absolute_existing_dir)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--device-evidence", required=True, type=absolute_existing_file)
    parser.add_argument("--trace", required=True, action="append", type=absolute_existing_file)
    parser.add_argument("--windows", required=True, action="append", type=absolute_existing_file)
    parser.add_argument("--output", required=True, type=absolute_new_file)
    args = parser.parse_args()
    if not SAFE_ID_PATTERN.fullmatch(args.experiment_id):
        parser.error("--experiment-id must use only letters, digits, dot, underscore, and dash")
    args.expected_commit = args.expected_commit.lower()
    if not COMMIT_PATTERN.fullmatch(args.expected_commit):
        parser.error("--expected-commit must be a full 40-character lowercase Git SHA")
    if len(args.trace) != len(args.windows):
        parser.error("provide one --windows file for each --trace, in matching order")
    return args


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def run_readonly(command: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def verify_target_and_source(args: argparse.Namespace) -> str:
    if is_relative_to(Path(__file__).resolve(), LOCAL_SOURCE_ROOT) or is_relative_to(
        Path.cwd().resolve(), LOCAL_SOURCE_ROOT
    ):
        raise TraceError("this target-side script must be copied to and run on the rented C500")
    if args.device_evidence.stat().st_size > 4 * 1024 * 1024:
        raise TraceError("device evidence is unexpectedly large")
    evidence = args.device_evidence.read_text(encoding="utf-8", errors="replace")
    if not C500_PATTERN.search(evidence):
        raise TraceError("device evidence does not identify a MetaX C500")
    live_tool = None
    for name in ("mx-smi", "metax-smi", "maca-smi"):
        executable = shutil.which(name)
        if executable is None:
            continue
        result = run_readonly([executable])
        if result.returncode == 0:
            live_tool = name
            break
    if live_tool is None:
        raise TraceError("no working MetaX device status tool was found on this host")
    if not Path("/opt/maca").is_dir() and not os.environ.get("MACA_PATH"):
        raise TraceError("MACA runtime identity is absent")

    status = run_readonly(["git", "-C", str(args.repo), "status", "--porcelain"])
    if status.returncode != 0 or status.stdout.strip():
        raise TraceError("source worktree is unreadable or dirty; clean-worktree guard refused the run")
    head = run_readonly(["git", "-C", str(args.repo), "rev-parse", "HEAD"])
    actual = head.stdout.strip().lower() if head.returncode == 0 else ""
    if actual != args.expected_commit:
        raise TraceError(
            f"source HEAD mismatch: expected {args.expected_commit}, found {actual or 'UNKNOWN'}"
        )
    return live_tool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TraceError(f"invalid number for {label}") from exc
    if not math.isfinite(number):
        raise TraceError(f"non-finite number for {label}")
    return number


def parse_windows(path: Path) -> list[dict[str, float | str]]:
    import csv

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        formats = (
            ("start_ns", "end_ns", 1_000.0),
            ("start_us", "end_us", 1.0),
            ("start_s", "end_s", 0.000001),
        )
        chosen = next((item for item in formats if {"phase", item[0], item[1]} <= fields), None)
        if chosen is None:
            raise TraceError("window CSV needs phase plus start/end columns in ns, us, or s")
        start_key, end_key, divisor = chosen
        windows = []
        for line_number, row in enumerate(reader, start=2):
            phase = (row.get("phase") or "").strip().lower()
            if phase not in {"prefill", "mixed", "decode", "tail"}:
                raise TraceError(f"invalid phase at {path}:{line_number}: {phase}")
            start = finite_number(row[start_key], f"{path}:{line_number}:{start_key}") / divisor
            end = finite_number(row[end_key], f"{path}:{line_number}:{end_key}") / divisor
            if end <= start:
                raise TraceError(f"non-positive window at {path}:{line_number}")
            windows.append({"phase": phase, "start_us": start, "end_us": end})
    counts = Counter(str(window["phase"]) for window in windows)
    if set(counts) != {"prefill", "mixed", "decode", "tail"} or any(
        value != 1 for value in counts.values()
    ):
        raise TraceError("window CSV must contain each phase exactly once")
    windows.sort(key=lambda item: float(item["start_us"]))
    for previous, current in zip(windows, windows[1:]):
        if float(current["start_us"]) < float(previous["end_us"]):
            raise TraceError("phase windows overlap")
    return windows


def load_trace(path: Path) -> Any:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


def trace_events(document: Any) -> Iterator[dict[str, Any]]:
    events = document.get("traceEvents") if isinstance(document, dict) else document
    if not isinstance(events, list):
        raise TraceError("trace must be a JSON event list or contain traceEvents")
    for event in events:
        if isinstance(event, dict):
            yield event


def category_for(name: str, category: str) -> str:
    combined = f"{category} {name}"
    for label, pattern in CATEGORY_PATTERNS:
        if pattern.search(combined):
            return label
    return "other"


def event_interval(event: dict[str, Any]) -> tuple[float, float] | None:
    if event.get("ph") != "X" or "ts" not in event or "dur" not in event:
        return None
    start = finite_number(event["ts"], "trace event ts")
    duration = finite_number(event["dur"], "trace event dur")
    if duration < 0:
        raise TraceError("trace event has negative duration")
    return start, start + duration


def summarize_trace(path: Path, windows_path: Path) -> dict[str, Any]:
    windows = parse_windows(windows_path)
    per_phase: dict[str, dict[str, Any]] = {
        str(window["phase"]): {
            "window_start_us": round(float(window["start_us"]), 6),
            "window_end_us": round(float(window["end_us"]), 6),
            "window_duration_us": round(
                float(window["end_us"]) - float(window["start_us"]), 6
            ),
            "event_count": 0,
            "summed_event_duration_us": 0.0,
            "categories": defaultdict(lambda: {"count": 0, "duration_us": 0.0}),
            "diagnostic_counts": Counter(),
        }
        for window in windows
    }
    document = load_trace(path)
    complete_events = 0
    for event in trace_events(document):
        interval = event_interval(event)
        if interval is None:
            continue
        complete_events += 1
        start, end = interval
        name = str(event.get("name", ""))
        raw_category = str(event.get("cat", ""))
        label = category_for(name, raw_category)
        combined = f"{raw_category} {name}"
        for window in windows:
            overlap = max(
                0.0,
                min(end, float(window["end_us"]))
                - max(start, float(window["start_us"])),
            )
            if overlap <= 0:
                continue
            phase = str(window["phase"])
            summary = per_phase[phase]
            summary["event_count"] += 1
            summary["summed_event_duration_us"] += overlap
            summary["categories"][label]["count"] += 1
            summary["categories"][label]["duration_us"] += overlap
            for diagnostic, pattern in DIAGNOSTIC_PATTERNS.items():
                if pattern.search(combined):
                    summary["diagnostic_counts"][diagnostic] += 1

    rendered_phases = []
    for window in windows:
        phase = str(window["phase"])
        summary = per_phase[phase]
        rendered_phases.append(
            {
                "phase": phase,
                "window_start_us": summary["window_start_us"],
                "window_end_us": summary["window_end_us"],
                "window_duration_us": summary["window_duration_us"],
                "event_count": summary["event_count"],
                "summed_event_duration_us": round(summary["summed_event_duration_us"], 6),
                "categories": {
                    key: {
                        "count": value["count"],
                        "duration_us": round(value["duration_us"], 6),
                    }
                    for key, value in sorted(summary["categories"].items())
                },
                "diagnostic_counts": dict(sorted(summary["diagnostic_counts"].items())),
            }
        )
    return {
        "trace": str(path),
        "trace_sha256": sha256_file(path),
        "windows": str(windows_path),
        "windows_sha256": sha256_file(windows_path),
        "complete_events": complete_events,
        "phases": rendered_phases,
    }


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    old_umask = os.umask(0o077)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
    finally:
        os.umask(old_umask)


def main() -> None:
    args = parse_args()
    try:
        live_tool = verify_target_and_source(args)
        report = {
            "schema_version": 1,
            "experiment_id": args.experiment_id,
            "platform": "MetaX C500 (rented diagnostic target)",
            "source": {
                "repo": str(args.repo),
                "commit": args.expected_commit,
                "clean_worktree": True,
            },
            "target_guard": {
                "device_evidence": str(args.device_evidence),
                "device_evidence_sha256": sha256_file(args.device_evidence),
                "live_status_tool": live_tool,
            },
            "traces": [
                summarize_trace(trace, windows)
                for trace, windows in zip(args.trace, args.windows)
            ],
            "duration_note": (
                "Category durations are overlap-clipped sums of complete events; nested or concurrent "
                "events can make them exceed wall-clock window duration. Do not divide these sums by "
                "window duration or use them as end-to-end attribution percentages."
            ),
            "privacy": (
                "Trace args and raw event names are never emitted; output contains only fixed "
                "categories, fixed diagnostic counters, explicit file identities, and hashes."
            ),
        }
        write_json_exclusive(args.output, report)
    except (TraceError, OSError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        die(str(exc))
    print(f"WROTE: {args.output}")


if __name__ == "__main__":
    main()
