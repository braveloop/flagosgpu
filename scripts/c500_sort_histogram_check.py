#!/usr/bin/env python3
"""C500-only differential and batched timing for the sort histogram candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path


LOCAL_ROOT = Path("/home/brave/flagos")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
CURRENT_STAGE = "startup"


def fail(message):
    raise RuntimeError(message)


def set_stage(name):
    global CURRENT_STAGE
    CURRENT_STAGE = name
    print(f"STAGE: {name}", flush=True)


def absolute_path(value):
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("paths must be absolute")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode:
        fail("Git source-identity check failed")
    return result.stdout.rstrip("\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flaggems-root", required=True, type=absolute_path)
    parser.add_argument("--expected-base-commit", required=True)
    parser.add_argument("--expected-sort-sha256", required=True)
    parser.add_argument("--output", required=True, type=absolute_path)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.expected_base_commit = args.expected_base_commit.lower()
    args.expected_sort_sha256 = args.expected_sort_sha256.lower()
    if not COMMIT_RE.fullmatch(args.expected_base_commit):
        parser.error("--expected-base-commit must be a full lowercase Git SHA")
    if not HASH_RE.fullmatch(args.expected_sort_sha256):
        parser.error("--expected-sort-sha256 must be a lowercase SHA-256")
    if args.warmup < 1 or args.repeats < 2:
        parser.error("--warmup must be >=1 and --repeats must be >=2")
    return args


def verify_source(args):
    script = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    if LOCAL_ROOT in script.parents or LOCAL_ROOT == cwd or LOCAL_ROOT in cwd.parents:
        fail("copy this script to and run it only on the target C500")
    root = args.flaggems_root.resolve()
    sort_path = root / "src/flag_gems/ops/sort.py"
    if not root.is_dir() or not sort_path.is_file():
        fail("FlagGems root or sort.py is missing")
    if git(root, "rev-parse", "HEAD") != args.expected_base_commit:
        fail("FlagGems base commit mismatch")
    dirty = []
    for line in git(root, "status", "--porcelain").splitlines():
        if line:
            dirty.append(line[3:])
    if dirty != ["src/flag_gems/ops/sort.py"]:
        fail("dirty manifest must contain only src/flag_gems/ops/sort.py")
    if sha256(sort_path) != args.expected_sort_sha256:
        fail("sort.py hash does not match the explicit candidate manifest")
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        fail("output must be a new file under an existing directory")
    return root, sort_path


def synchronize(torch):
    torch.cuda.synchronize()


def launch_histogram(torch, triton, sort_module, kernel, tensor, descending=False):
    n = tensor.shape[-1]
    m = tensor.numel() // n
    tile_n = 1024
    tiles_n_per_cta = 8
    num_passes = 8
    output = torch.zeros((m, num_passes, 16), device=tensor.device, dtype=torch.int32)
    grid = (m * triton.cdiv(n, tile_n * tiles_n_per_cta), 1, 1)
    kernel[grid](
        tensor,
        output,
        num_passes,
        m,
        n,
        tiles_n_per_cta,
        tile_n,
        16,
        4,
        descending,
    )
    return output


def candidate_sort(sort_module, tensor, enabled, descending=False):
    candidate = sort_module.compute_global_hist_kernel_metax_fp32_k4
    if not enabled:
        sort_module.compute_global_hist_kernel_metax_fp32_k4 = (
            sort_module.compute_global_hist_kernel
        )
    try:
        return sort_module.radix_sort(tensor, k_bits=4, descending=descending)
    finally:
        sort_module.compute_global_hist_kernel_metax_fp32_k4 = candidate


def candidate_public_sort(sort_module, tensor, enabled, descending=False):
    candidate = sort_module.compute_global_hist_kernel_metax_fp32_k4
    if not enabled:
        sort_module.compute_global_hist_kernel_metax_fp32_k4 = (
            sort_module.compute_global_hist_kernel
        )
    try:
        return sort_module.sort(tensor, dim=-1, descending=descending)
    finally:
        sort_module.compute_global_hist_kernel_metax_fp32_k4 = candidate


def assert_exact(torch, expected, actual, label):
    expected_values, expected_indices = expected
    actual_values, actual_indices = actual
    if not torch.equal(expected_values.view(torch.int32), actual_values.view(torch.int32)):
        fail(f"{label}: sorted value bits differ")
    if not torch.equal(expected_indices, actual_indices):
        fail(f"{label}: sorted indices differ")


def timed_sort(torch, sort_module, tensor, enabled, warmup, repeats):
    for _ in range(warmup):
        candidate_sort(sort_module, tensor, enabled, descending=False)
    synchronize(torch)
    started = time.perf_counter()
    for _ in range(repeats):
        candidate_sort(sort_module, tensor, enabled, descending=False)
    synchronize(torch)
    return (time.perf_counter() - started) * 1000.0 / repeats


def timed_histogram(torch, triton, sort_module, tensor, kernel, warmup, repeats):
    for _ in range(warmup):
        launch_histogram(torch, triton, sort_module, kernel, tensor)
    synchronize(torch)
    started = time.perf_counter()
    for _ in range(repeats):
        launch_histogram(torch, triton, sort_module, kernel, tensor)
    synchronize(torch)
    return (time.perf_counter() - started) * 1000.0 / repeats


def top_p_and_sample(flag_gems, sampler_module, sort_module, logits, top_p, enabled):
    candidate = sort_module.compute_global_hist_kernel_metax_fp32_k4
    if not enabled:
        sort_module.compute_global_hist_kernel_metax_fp32_k4 = (
            sort_module.compute_global_hist_kernel
        )
    try:
        with flag_gems.use_gems():
            filtered = sampler_module.apply_top_k_top_p_pytorch(
                logits.clone(), k=None, p=top_p
            )
            probabilities = filtered.softmax(dim=-1, dtype=filtered.dtype)
            token = sampler_module.random_sample(
                probabilities, generators={}, use_fp64_gumbel=False
            )
        return filtered, token
    finally:
        sort_module.compute_global_hist_kernel_metax_fp32_k4 = candidate


def main():
    set_stage("parse arguments and verify source manifest")
    args = parse_args()
    root, sort_path = verify_source(args)

    set_stage("import target modules")
    import torch
    import triton
    import flag_gems

    sort_module = importlib.import_module("flag_gems.ops.sort")
    sampler_module = importlib.import_module(
        "vllm.v1.sample.ops.topk_topp_sampler"
    )

    module_path = Path(sort_module.__file__).resolve()
    package_path = Path(flag_gems.__file__).resolve()
    if root not in module_path.parents or root not in package_path.parents:
        fail("FlagGems did not import from the explicit candidate worktree")
    if flag_gems.vendor_name != "metax" or "C500" not in torch.cuda.get_device_name(0):
        fail("target is not a MetaX C500")

    set_stage("verify selector and fallback routing")
    for n in (1, 1023):
        if sort_module._select_global_hist_kernel(torch.float32, 4, n) is not (
            sort_module.compute_global_hist_kernel
        ):
            fail(f"short FP32/k_bits=4 shape did not use fallback for n={n}")
    for n in (1024, 1025, 130560):
        if sort_module._select_global_hist_kernel(torch.float32, 4, n) is not (
            sort_module.compute_global_hist_kernel_metax_fp32_k4
        ):
            fail(f"FP32/k_bits=4 did not select the candidate for n={n}")
    for dtype, k_bits in (
        (torch.float16, 4),
        (torch.bfloat16, 4),
        (torch.int32, 4),
        (torch.float32, 1),
    ):
        if sort_module._select_global_hist_kernel(dtype, k_bits, 130560) is not (
            sort_module.compute_global_hist_kernel
        ):
            fail(f"fallback selection failed for dtype={dtype}, k_bits={k_bits}")

    set_stage("compare fallback sort paths")
    fallback_results = []
    # Match the service registration context for original and candidate alike.
    with flag_gems.use_gems():
        for dtype in (torch.float16, torch.bfloat16, torch.int32):
            if dtype == torch.int32:
                tensor = torch.randint(-32, 33, (4, 4097), dtype=dtype, device="cuda")
            else:
                tensor = torch.randn((4, 4097), dtype=dtype, device="cuda")
            original = candidate_sort(sort_module, tensor, False)
            candidate_available = candidate_sort(sort_module, tensor, True)
            synchronize(torch)
            if not torch.equal(original[0], candidate_available[0]) or not torch.equal(
                original[1], candidate_available[1]
            ):
                fail(f"fallback changed while candidate was present for dtype={dtype}")
            fallback_results.append(
                {"dtype": str(dtype), "original_path_unchanged": True}
            )

    set_stage("compare original and candidate histograms")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20261002)
    histogram_cases = (
        (1, 1),
        (1, 1023),
        (1, 1024),
        (1, 1025),
        (1, 4097),
        (1, 8191),
        (1, 8192),
        (1, 8193),
        (8, 130560),
        (64, 130560),
        (1, 131072),
    )
    histogram_results = []
    for m, n in histogram_cases:
        tensor = torch.randn((m, n), dtype=torch.float32, device="cuda", generator=generator)
        if n >= 6:
            tensor[0, :6] = torch.tensor(
                [float("-inf"), -0.0, 0.0, float("inf"), float("nan"), 1.0],
                device="cuda",
            )
        original = launch_histogram(
            torch, triton, sort_module, sort_module.compute_global_hist_kernel, tensor
        )
        candidate = launch_histogram(
            torch,
            triton,
            sort_module,
            sort_module.compute_global_hist_kernel_metax_fp32_k4,
            tensor,
        )
        synchronize(torch)
        if not torch.equal(original, candidate):
            fail(f"histogram mismatch for shape {(m, n)}")
        histogram_results.append(
            {
                "shape": [m, n],
                "integer_exact": True,
                "special_values": n >= 6,
            }
        )

    set_stage("compare repeated-value histograms")
    repeated = torch.full((64, 130560), 0.5, dtype=torch.float32, device="cuda")
    repeated_original = launch_histogram(
        torch, triton, sort_module, sort_module.compute_global_hist_kernel, repeated
    )
    repeated_candidate = launch_histogram(
        torch,
        triton,
        sort_module,
        sort_module.compute_global_hist_kernel_metax_fp32_k4,
        repeated,
    )
    synchronize(torch)
    if not torch.equal(repeated_original, repeated_candidate):
        fail("histogram mismatch for repeated-value worst distribution")
    histogram_results.append(
        {"shape": [64, 130560], "distribution": "repeated", "integer_exact": True}
    )

    set_stage("compare complete radix sort outputs")
    correctness = []
    with flag_gems.use_gems():
        for m, n, descending in (
            (1, 8191, False),
            (1, 8192, False),
            (1, 8193, False),
            (1, 130560, False),
            (8, 130560, False),
            (64, 130560, False),
            (1, 131072, False),
            (1, 130560, True),
        ):
            tensor = torch.randn(
                (m, n), dtype=torch.float32, device="cuda", generator=generator
            )
            tensor[:, :128] = 0.5
            tensor[0, :5] = torch.tensor(
                [float("-inf"), -0.0, 0.0, float("inf"), float("nan")],
                device="cuda",
            )
            original = candidate_sort(
                sort_module, tensor, False, descending=descending
            )
            candidate = candidate_sort(
                sort_module, tensor, True, descending=descending
            )
            synchronize(torch)
            assert_exact(
                torch, original, candidate, f"shape={(m, n)}, descending={descending}"
            )
            correctness.append(
                {
                    "shape": [m, n],
                    "descending": descending,
                    "special_values": True,
                    "value_bits_exact": True,
                    "indices_exact": True,
                }
            )

        noncontiguous = torch.randn(
            (130560, 2), dtype=torch.float32, device="cuda", generator=generator
        ).transpose(0, 1)
        if noncontiguous.is_contiguous():
            fail("non-contiguous sort fixture unexpectedly became contiguous")
        noncontiguous[:, :128] = 0.5
        original = candidate_public_sort(
            sort_module, noncontiguous, False, descending=False
        )
        candidate = candidate_public_sort(
            sort_module, noncontiguous, True, descending=False
        )
        synchronize(torch)
        assert_exact(torch, original, candidate, "non-contiguous public sort")
        correctness.append(
            {
                "shape": [2, 130560],
                "descending": False,
                "noncontiguous_input": True,
                "value_bits_exact": True,
                "indices_exact": True,
            }
        )

    set_stage("compare top-p filtering and sampling")
    contiguous_logits = torch.randn(
        (64, 130560), dtype=torch.float32, device="cuda", generator=generator
    )
    noncontiguous_logits = torch.randn(
        (130560, 8), dtype=torch.float32, device="cuda", generator=generator
    ).transpose(0, 1)
    if noncontiguous_logits.is_contiguous():
        fail("non-contiguous top-p fixture unexpectedly became contiguous")
    top_p_results = []
    for case_name, logits in (
        ("contiguous", contiguous_logits),
        ("noncontiguous", noncontiguous_logits),
    ):
        logits[:, :128] = 0.5
        top_p = torch.full(
            (logits.shape[0],), 0.95, dtype=torch.float32, device="cuda"
        )
        torch.cuda.manual_seed_all(20261002)
        initial_rng_state = torch.cuda.get_rng_state()
        original_filtered, original_token = top_p_and_sample(
            flag_gems, sampler_module, sort_module, logits, top_p, enabled=False
        )
        original_rng_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(initial_rng_state)
        candidate_filtered, candidate_token = top_p_and_sample(
            flag_gems, sampler_module, sort_module, logits, top_p, enabled=True
        )
        candidate_rng_state = torch.cuda.get_rng_state()
        synchronize(torch)
        if not torch.equal(
            original_filtered.view(torch.int32), candidate_filtered.view(torch.int32)
        ):
            fail(f"{case_name} top-p filtered logits differ at the bit level")
        if not torch.equal(original_token, candidate_token):
            fail(f"{case_name} top-p sampled token differs for the same seed")
        if not torch.equal(original_rng_state, candidate_rng_state):
            fail(f"{case_name} top-p sampling RNG state differs")
        top_p_results.append(
            {
                "layout": case_name,
                "shape": list(logits.shape),
                "filtered_logits_bits_exact": True,
                "sampled_tokens_exact": True,
                "rng_state_exact": True,
            }
        )

    set_stage("time histogram original-candidate-original")
    timing_tensor = torch.randn(
        (64, 130560), dtype=torch.float32, device="cuda", generator=generator
    )
    histogram_timings = [
        {
            "arm": arm,
            "mean_ms": timed_histogram(
                torch,
                triton,
                sort_module,
                timing_tensor,
                kernel,
                args.warmup,
                args.repeats,
            ),
        }
        for arm, kernel in (
            ("original-before", sort_module.compute_global_hist_kernel),
            ("candidate", sort_module.compute_global_hist_kernel_metax_fp32_k4),
            ("original-after", sort_module.compute_global_hist_kernel),
        )
    ]
    set_stage("time complete sort original-candidate-original")
    with flag_gems.use_gems():
        timings = [
            {
                "arm": arm,
                "mean_ms": timed_sort(
                    torch,
                    sort_module,
                    timing_tensor,
                    enabled,
                    args.warmup,
                    args.repeats,
                ),
            }
            for arm, enabled in (
                ("original-before", False),
                ("candidate", True),
                ("original-after", False),
            )
        ]

    set_stage("write report")
    report = {
        "schema_version": 1,
        "platform": "MetaX C500",
        "flaggems_root": str(root),
        "flaggems_import": str(package_path),
        "sort_module": str(module_path),
        "base_commit": args.expected_base_commit,
        "sort_sha256": sha256(sort_path),
        "histogram": histogram_results,
        "histogram_timings": histogram_timings,
        "sort_correctness": correctness,
        "top_p_sampling": {
            "top_p": 0.95,
            "cases": top_p_results,
        },
        "fallbacks": fallback_results,
        "timing_shape": [64, 130560],
        "timing_descending": False,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timings": timings,
    }
    old_umask = os.umask(0o077)
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    finally:
        os.umask(old_umask)
    print(f"WROTE: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR stage={CURRENT_STAGE}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(2)
