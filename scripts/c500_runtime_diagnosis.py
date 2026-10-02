#!/usr/bin/env python3
"""Summarize one C500 diagnostic arm from already captured evidence.

This script is intentionally target-side only.  It imports no FlagOS, vLLM,
Torch, or vendor Python modules, runs no workload, and reads only paths named on
the command line.  It also refuses a dirty or unexpected source checkout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LOCAL_SOURCE_ROOT = Path("/home/brave/flagos")
C500_PATTERN = re.compile(r"(?i)(?:\bmetax\b.*\bc500\b|\bmxc?500\b|\bc500\b)")
SAFE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
PHASES = ("prefill", "mixed", "decode", "tail")
OFFICIAL_RUNS = 4
SKIP_FIRST = 1

BENCHMARK_NUMBER_COLUMNS = (
    "Benchmark Duration (s)",
    "Total Input Tokens",
    "Total Output Tokens",
    "Req/s",
    "Output tok/s",
    "Peak Output tok/s",
    "Total tok/s",
    "Mean TTFT (ms)",
    "Median TTFT (ms)",
    "P99 TTFT (ms)",
    "Mean TPOT (ms)",
    "Median TPOT (ms)",
    "P99 TPOT (ms)",
    "Mean ITL (ms)",
    "Median ITL (ms)",
    "P99 ITL (ms)",
)
BENCHMARK_ID_COLUMNS = ("Prefill", "Decode", "Conc", "Num Prompts")
SCHEDULER_KEYS = {
    "num_running_reqs",
    "num_waiting_reqs",
    "num_prefills",
    "num_decodes",
    "num_batched_tokens",
    "num_scheduled_tokens",
    "scheduled_tokens",
    "kv_cache_usage",
    "gpu_cache_usage",
    "num_cached_tokens",
}


class DiagnosisError(RuntimeError):
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
        description="Summarize captured C500 benchmark, scheduler, cgroup, and phase evidence."
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--arm",
        required=True,
        choices=("unmodified-before", "m1", "unmodified-after", "trace"),
    )
    parser.add_argument("--repo", required=True, type=absolute_existing_dir)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--device-evidence", required=True, type=absolute_existing_file)
    parser.add_argument("--benchmark-raw", action="append", type=absolute_existing_file)
    parser.add_argument("--scheduler-data", action="append", type=absolute_existing_file)
    parser.add_argument("--cgroup-before", type=absolute_existing_file)
    parser.add_argument("--cgroup-after", type=absolute_existing_file)
    parser.add_argument("--timestamps", type=absolute_existing_file)
    parser.add_argument("--output", required=True, type=absolute_new_file)
    args = parser.parse_args()
    if not SAFE_ID_PATTERN.fullmatch(args.experiment_id):
        parser.error("--experiment-id must use only letters, digits, dot, underscore, and dash")
    args.expected_commit = args.expected_commit.lower()
    if not COMMIT_PATTERN.fullmatch(args.expected_commit):
        parser.error("--expected-commit must be a full 40-character lowercase Git SHA")
    if bool(args.cgroup_before) != bool(args.cgroup_after):
        parser.error("--cgroup-before and --cgroup-after must be provided together")
    if not any((args.benchmark_raw, args.scheduler_data, args.cgroup_before, args.timestamps)):
        parser.error("provide at least one captured evidence input")
    return args


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def read_limited_text(path: Path, limit: int = 4 * 1024 * 1024) -> str:
    if path.stat().st_size > limit:
        raise DiagnosisError(f"text input is unexpectedly large: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


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
    script_path = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    if is_relative_to(script_path, LOCAL_SOURCE_ROOT) or is_relative_to(cwd, LOCAL_SOURCE_ROOT):
        raise DiagnosisError(
            "this target-side script must be copied to and run on the rented C500"
        )
    evidence = read_limited_text(args.device_evidence)
    if not C500_PATTERN.search(evidence):
        raise DiagnosisError("device evidence does not identify a MetaX C500")

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
        raise DiagnosisError("no working MetaX device status tool was found on this host")
    if not Path("/opt/maca").is_dir() and not os.environ.get("MACA_PATH"):
        raise DiagnosisError("MACA runtime identity is absent")

    status = run_readonly(["git", "-C", str(args.repo), "status", "--porcelain"])
    if status.returncode != 0:
        raise DiagnosisError("cannot read source worktree status")
    if status.stdout.strip():
        raise DiagnosisError("source worktree is dirty; clean-worktree guard refused the run")
    head = run_readonly(["git", "-C", str(args.repo), "rev-parse", "HEAD"])
    if head.returncode != 0:
        raise DiagnosisError("cannot read source HEAD")
    actual_commit = head.stdout.strip().lower()
    if actual_commit != args.expected_commit:
        raise DiagnosisError(
            f"source HEAD mismatch: expected {args.expected_commit}, found {actual_commit}"
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
        raise DiagnosisError(f"invalid number for {label}") from exc
    if not math.isfinite(number):
        raise DiagnosisError(f"non-finite number for {label}")
    return number


def integer_value(value: Any, label: str) -> int:
    number = finite_number(value, label)
    if not number.is_integer():
        raise DiagnosisError(f"non-integer value for {label}")
    return int(number)


def metric_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "stdev": None, "cv_percent": None}
    average = statistics.fmean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    cv = deviation / average * 100.0 if average != 0 else None
    return {
        "count": len(values),
        "mean": round(average, 6),
        "stdev": round(deviation, 6),
        "cv_percent": round(cv, 6) if cv is not None else None,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def summarize_benchmark(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = set(BENCHMARK_ID_COLUMNS) | {
            "Successful Requests",
            "Run Status",
            *BENCHMARK_NUMBER_COLUMNS,
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise DiagnosisError(f"benchmark CSV is missing columns: {', '.join(missing)}")
        grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
        for line_number, row in enumerate(reader, start=2):
            case = tuple(integer_value(row[key], f"{path}:{line_number}:{key}") for key in BENCHMARK_ID_COLUMNS)
            normalized: dict[str, Any] = {
                key: finite_number(row[key], f"{path}:{line_number}:{key}")
                for key in BENCHMARK_NUMBER_COLUMNS
            }
            normalized["Successful Requests"] = integer_value(
                row["Successful Requests"], f"{path}:{line_number}:Successful Requests"
            )
            normalized["Run Status"] = (row["Run Status"] or "").strip().upper()
            grouped[case].append(normalized)
    if not grouped:
        raise DiagnosisError(f"benchmark CSV contains no rows: {path}")

    cases = []
    for case, rows in sorted(grouped.items()):
        if len(rows) != OFFICIAL_RUNS:
            raise DiagnosisError(
                f"benchmark case {case} has {len(rows)} runs; expected {OFFICIAL_RUNS}"
            )
        effective = rows[SKIP_FIRST:]
        expected_requests = case[3]
        valid = all(
            row["Run Status"] == "SUCCESS"
            and row["Successful Requests"] == expected_requests
            for row in effective
        )
        cases.append(
            {
                "shape": dict(zip(BENCHMARK_ID_COLUMNS, case)),
                "runs": rows,
                "skip_first": SKIP_FIRST,
                "effective_runs_valid": valid,
                "effective_statistics": {
                    key: metric_stats([row[key] for row in effective])
                    for key in BENCHMARK_NUMBER_COLUMNS
                },
            }
        )
    return {
        "file": str(path),
        "sha256": sha256_file(path),
        "cases": cases,
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    text = read_limited_text(path, limit=128 * 1024 * 1024)
    if suffix == ".json":
        value = json.loads(text)
        if isinstance(value, dict):
            value = value.get("records", value.get("samples"))
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise DiagnosisError(f"scheduler JSON must contain a list of objects: {path}")
        return value
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DiagnosisError(f"invalid scheduler JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise DiagnosisError(f"scheduler JSONL row is not an object at {path}:{line_number}")
        records.append(value)
    return records


def whitelisted_numeric_leaves(value: Any) -> dict[str, float]:
    selected: dict[str, float] = {}

    def visit(item: Any, key: str = "") -> None:
        if isinstance(item, dict):
            for child_key, child in item.items():
                visit(child, str(child_key))
        elif key in SCHEDULER_KEYS and not isinstance(item, bool):
            selected[key] = finite_number(item, f"scheduler:{key}")

    visit(value)
    return selected


def scheduler_phase(sample: dict[str, float]) -> str:
    prefills = sample.get("num_prefills", 0)
    decodes = sample.get("num_decodes", 0)
    running = sample.get("num_running_reqs")
    if prefills > 0 and decodes > 0:
        return "mixed"
    if prefills > 0:
        return "prefill"
    if decodes > 0:
        return "decode"
    if running is not None and running <= 8:
        return "tail"
    return "unclassified"


def summarize_scheduler(path: Path) -> dict[str, Any]:
    records = load_records(path)
    metrics: dict[str, list[float]] = defaultdict(list)
    phases: Counter[str] = Counter()
    usable = 0
    for record in records:
        selected = whitelisted_numeric_leaves(record)
        if not selected:
            continue
        usable += 1
        phases[scheduler_phase(selected)] += 1
        for key, value in selected.items():
            metrics[key].append(value)
    if not records:
        raise DiagnosisError(f"scheduler data contains no records: {path}")
    if usable == 0:
        raise DiagnosisError(f"scheduler data has no whitelisted numeric fields: {path}")
    return {
        "file": str(path),
        "sha256": sha256_file(path),
        "records": len(records),
        "usable_records": usable,
        "phase_counts": dict(sorted(phases.items())),
        "metrics": {key: metric_stats(values) for key, values in sorted(metrics.items())},
    }


def parse_cgroup_snapshot(path: Path) -> dict[str, int]:
    text = read_limited_text(path)
    stripped = text.strip()
    if not stripped:
        raise DiagnosisError(f"empty cgroup snapshot: {path}")
    if stripped.startswith("{"):
        raw = json.loads(stripped)
        if not isinstance(raw, dict):
            raise DiagnosisError(f"cgroup JSON must be an object: {path}")
        return {str(key): integer_value(value, f"{path}:{key}") for key, value in raw.items()}
    values: dict[str, int] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[=\s]+", line, maxsplit=1)
        if len(parts) != 2 or not parts[0]:
            raise DiagnosisError(f"invalid cgroup row at {path}:{line_number}")
        key = parts[0]
        if key in values:
            raise DiagnosisError(f"duplicate cgroup key at {path}:{line_number}: {key}")
        values[key] = integer_value(parts[1], f"{path}:{line_number}:{key}")
    if not values:
        raise DiagnosisError(f"cgroup snapshot contains no counters: {path}")
    return values


def summarize_cgroup(before_path: Path, after_path: Path) -> dict[str, Any]:
    before = parse_cgroup_snapshot(before_path)
    after = parse_cgroup_snapshot(after_path)
    common = sorted(before.keys() & after.keys())
    if not common:
        raise DiagnosisError("cgroup snapshots have no common counters")
    delta = {key: after[key] - before[key] for key in common}
    oom_delta = {
        key: value
        for key, value in delta.items()
        if key in {"oom", "oom_kill", "oom_group_kill", "failcnt"}
    }
    return {
        "before_file": str(before_path),
        "before_sha256": sha256_file(before_path),
        "after_file": str(after_path),
        "after_sha256": sha256_file(after_path),
        "before": before,
        "after": after,
        "delta": delta,
        "oom_or_failure_delta": oom_delta,
        "oom_or_failure_increased": any(value > 0 for value in oom_delta.values()),
    }


def parse_phase_windows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        units = [
            ("ns", "start_ns", "end_ns", 1_000.0),
            ("us", "start_us", "end_us", 1.0),
            ("s", "start_s", "end_s", 0.000001),
        ]
        chosen = next((item for item in units if {"phase", item[1], item[2]} <= fields), None)
        if chosen is None:
            raise DiagnosisError(
                "timestamp CSV needs phase plus start/end columns in ns, us, or s"
            )
        unit, start_key, end_key, divisor = chosen
        windows = []
        for line_number, row in enumerate(reader, start=2):
            phase = (row.get("phase") or "").strip().lower()
            if phase not in PHASES:
                raise DiagnosisError(f"invalid phase at {path}:{line_number}: {phase}")
            start = finite_number(row[start_key], f"{path}:{line_number}:{start_key}") / divisor
            end = finite_number(row[end_key], f"{path}:{line_number}:{end_key}") / divisor
            if end <= start:
                raise DiagnosisError(f"non-positive phase window at {path}:{line_number}")
            windows.append(
                {
                    "phase": phase,
                    "start_us": round(start, 6),
                    "end_us": round(end, 6),
                    "duration_us": round(end - start, 6),
                    "source_unit": unit,
                }
            )
    counts = Counter(window["phase"] for window in windows)
    if set(counts) != set(PHASES) or any(counts[phase] != 1 for phase in PHASES):
        raise DiagnosisError("timestamp CSV must contain each of prefill, mixed, decode, and tail once")
    ordered = sorted(windows, key=lambda item: item["start_us"])
    for previous, current in zip(ordered, ordered[1:]):
        if current["start_us"] < previous["end_us"]:
            raise DiagnosisError("phase windows overlap")
    return ordered


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
            "arm": args.arm,
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
            "benchmark": [summarize_benchmark(path) for path in (args.benchmark_raw or [])],
            "scheduler": [summarize_scheduler(path) for path in (args.scheduler_data or [])],
            "privacy": (
                "Only whitelisted numeric fields and explicit file identities are emitted; "
                "raw logs, environment variables, prompts, headers, and credentials are not copied."
            ),
        }
        if args.cgroup_before:
            report["cgroup"] = summarize_cgroup(args.cgroup_before, args.cgroup_after)
        if args.timestamps:
            report["phase_windows"] = {
                "file": str(args.timestamps),
                "sha256": sha256_file(args.timestamps),
                "windows": parse_phase_windows(args.timestamps),
            }
        write_json_exclusive(args.output, report)
    except (DiagnosisError, OSError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        die(str(exc))
    print(f"WROTE: {args.output}")


if __name__ == "__main__":
    main()
