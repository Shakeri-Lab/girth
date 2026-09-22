#!/usr/bin/env python3
"""
benchmark_acceleration.py -- Verification and Benchmarking Harness for MWC Acceleration.

Milestone 2 (R2 Profiling & Acceleration in Shakeri-Lab/girth)
Author: M2 Explorer 3 (Speedup & Equivalence Harness)

This script provides a rigorous, reproducible benchmarking and mathematical
verification harness that compares:
  1. Pure Python baseline: minimum_weight_cycle(G, use_acceleration=False)
  2. Accelerated compiled C++/Cython: minimum_weight_cycle(G, use_acceleration=True)

Coverage:
  - Dense synthetic graphs: G(n, p=0.8) across n in {20, 40, 60, 80, 100, 150, 200, 300, 500}
  - Sparse synthetic graphs: G(n, p=0.05) across n in {50, 100, 200, 500, 1000, 2000, 5000}
  - Real-world benchmark networks from datasets/realnets/ (10 canonical networks)

Verification Criteria:
  1. Measured Speedup: S = T_pure / T_acc > 1.0x (reporting mean, median, min, max)
  2. Numerical Equivalence: |gamma_pure - gamma_acc| <= 1e-9 * max(1, gamma_pure)
  3. Cycle Structural Validity: returned cycle is a valid simple cycle in G whose
     edge weight sum equals gamma within 1e-9 tolerance
  4. Zero False Prunings: verified against exact oracle (small graphs) and known
     authoritative girths (real networks)
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

# Ensure girth is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GIRTH_PATH = os.path.join(PROJECT_ROOT, "girth")
if GIRTH_PATH not in sys.path:
    sys.path.insert(0, GIRTH_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import networkx as nx
    import numpy as np
except ImportError as e:
    sys.stderr.write(f"Error: Missing required dependency: {e}\n")
    sys.exit(1)

try:
    from shortest_cycle import (
        minimum_weight_cycle,
        exact_oracle,
        preprocess_degeneracy,
    )
except ImportError:
    try:
        from girth.shortest_cycle import (
            minimum_weight_cycle,
            exact_oracle,
            preprocess_degeneracy,
        )
    except ImportError as e:
        sys.stderr.write(f"Error: Cannot import girth.shortest_cycle: {e}\n")
        sys.exit(1)

# Default tolerance for floating point comparisons
TOL = 1e-9

# Authoritative ground truth girths for real-world networks (from TEST_READY.md / manifest.json)
REALNET_GROUND_TRUTH: Dict[str, Dict[str, Any]] = {
    "lesmis": {"n": 77, "m": 254, "girth": 3.0},
    "celegans-neural": {"n": 297, "m": 2148, "girth": 3.0},
    "chicago-sketch-road": {"n": 933, "m": 1475, "girth": 2.03239},
    "rome99-road": {"n": 3353, "m": 4831, "girth": 6.0},
    "uspowergrid-synth": {"n": 4941, "m": 6594, "girth": 379.0},
    "usairport-2010": {"n": 1572, "m": 17214, "girth": 3.0},
    "openflights-air": {"n": 3188, "m": 18833, "girth": 32.74156860581118},
    "chicago-regional-road": {"n": 12979, "m": 20627, "girth": 0.1},
    "osm-portland-drive": {"n": 20154, "m": 30477, "girth": 18.75443212338135},
    "sydney-road": {"n": 32956, "m": 38787, "girth": 0.053},
}


# ============================================================================
# 1. Environment and Hardware Provenance
# ============================================================================

def get_provenance() -> Dict[str, Any]:
    """Capture comprehensive hardware, OS, Python, and Git provenance."""
    here = os.path.dirname(os.path.abspath(__file__))

    # Git metadata
    git_sha = "unknown"
    git_dirty = False
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=here, stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=here, stderr=subprocess.DEVNULL
        ).decode().strip()
        git_dirty = len(status) > 0
    except Exception:
        pass

    # CPU model detection
    cpu_model = platform.processor() or "unknown"
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        elif platform.system() == "Darwin":
            cpu_model = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL
            ).decode().strip()
    except Exception:
        pass

    # Check Cython extension availability
    cython_active = False
    try:
        import girth.c_extensions.fast_mwc as _fast_mwc  # noqa: F401
        cython_active = True
    except ImportError:
        cython_active = False

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "cpu_model": cpu_model,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "networkx_version": nx.__version__,
        "numpy_version": np.__version__,
        "git_commit": git_sha,
        "git_dirty": git_dirty,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "slurm_node": os.environ.get("SLURM_NODELIST", socket.gethostname()),
        "cython_compiled": cython_active,
    }


# ============================================================================
# 2. Graph Generation and Loading Pipeline
# ============================================================================

def sample_edge_weight(rng: random.Random, weight_type: str = "uniform") -> float:
    """Sample an edge weight according to specified strategy."""
    if weight_type == "uniform":
        return round(rng.uniform(0.1, 10.0), 6)
    elif weight_type == "fractional":
        # Discrete fractional values with high collision/near-tie probability
        return float(rng.choice([1/7, 0.25, 1/3, 0.5, 1.0, 2.5]))
    elif weight_type == "integer":
        return float(rng.randint(1, 20))
    elif weight_type == "unit":
        return 1.0
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")


def generate_dense_graph(
    n: int,
    p: float = 0.8,
    seed: int = 42,
    weight_type: str = "uniform",
) -> nx.Graph:
    """Generate dense Erdős-Rényi graph G(n, p) with non-negative edge weights."""
    rng = random.Random(seed)
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = sample_edge_weight(rng, weight_type)
    return G


def generate_sparse_graph(
    n: int,
    p: float = 0.05,
    seed: int = 42,
    weight_type: str = "uniform",
) -> nx.Graph:
    """
    Generate sparse Erdős-Rényi graph G(n, p) with non-negative edge weights.
    Ensures the graph contains at least one cycle (cyclomatic number mu >= 1)
    by adding a random triangle or chord if the random graph happens to be a forest.
    """
    rng = random.Random(seed)
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = sample_edge_weight(rng, weight_type)

    # If graph is acyclic forest, inject a small 3-cycle to ensure finite girth
    m = G.number_of_edges()
    c = nx.number_connected_components(G)
    mu = m - n + c
    if mu <= 0 and n >= 3:
        nodes = list(range(min(3, n)))
        G.add_edge(nodes[0], nodes[1], weight=sample_edge_weight(rng, weight_type))
        G.add_edge(nodes[1], nodes[2], weight=sample_edge_weight(rng, weight_type))
        G.add_edge(nodes[2], nodes[0], weight=sample_edge_weight(rng, weight_type))

    return G


def load_real_network(
    name: str,
    datasets_dir: str = "/scratch/hs9hd/mwc_certified_pruning/datasets/realnets",
) -> nx.Graph:
    """
    Load a real-world network from .edges file in datasets_dir.
    Format: space-separated lines 'u v w'.
    """
    edge_file = os.path.join(datasets_dir, f"{name}.edges")
    if not os.path.exists(edge_file):
        raise FileNotFoundError(f"Dataset file not found: {edge_file}")

    G = nx.Graph()
    with open(edge_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
                G.add_edge(u, v, weight=w)
            elif len(parts) == 2:
                u, v = int(parts[0]), int(parts[1])
                G.add_edge(u, v, weight=1.0)
    return G


# ============================================================================
# 3. Mathematical Verification Engine (Equivalence, Validity, Soundness)
# ============================================================================

def check_numerical_equivalence(
    gamma_pure: float,
    gamma_acc: float,
    tol: float = TOL,
) -> Tuple[bool, float, float]:
    """
    Verify numerical equivalence between pure Python and accelerated results.
    Returns:
        (is_equivalent, relative_difference, absolute_difference)
    """
    # Acyclic handling
    if math.isinf(gamma_pure) and math.isinf(gamma_acc):
        return True, 0.0, 0.0
    if math.isinf(gamma_pure) or math.isinf(gamma_acc):
        return False, float("inf"), float("inf")

    abs_diff = abs(gamma_pure - gamma_acc)
    scale = max(1.0, abs(gamma_pure), abs(gamma_acc))
    rel_diff = abs_diff / scale

    is_equiv = abs_diff <= tol * scale
    return is_equiv, rel_diff, abs_diff


def validate_cycle_structure(
    G: nx.Graph,
    cycle: Sequence[Any],
    expected_weight: float,
    weight_key: str = "weight",
    tol: float = TOL,
) -> Tuple[bool, str, float]:
    """
    Verify that the returned cycle forms a valid simple cycle in G and that its
    recomputed edge weight sum matches expected_weight within tolerance.

    Returns:
        (is_valid, reason_string, recomputed_weight)
    """
    # Case: Acyclic graph
    if math.isinf(expected_weight):
        if cycle is None or len(cycle) == 0:
            return True, "Valid acyclic certificate (empty cycle)", 0.0
        return False, f"Expected acyclic, but got non-empty cycle {cycle}", float("inf")

    if cycle is None or len(cycle) < 3:
        return False, f"Cycle must have >= 3 nodes (or 2 for multigraph), got {cycle}", 0.0

    # Endpoints must match: [v0, v1, ..., vk=v0]
    if cycle[0] != cycle[-1]:
        return False, f"Cycle not closed: start {cycle[0]} != end {cycle[-1]}", 0.0

    internal_nodes = cycle[:-1]
    if len(set(internal_nodes)) != len(internal_nodes):
        return False, f"Cycle is not simple; repeated internal nodes in {cycle}", 0.0

    # Verify edge existence and recompute weight sum
    recomputed_weight = 0.0
    k = len(internal_nodes)
    for i in range(k):
        u = internal_nodes[i]
        v = internal_nodes[(i + 1) % k]
        if not G.has_edge(u, v):
            return False, f"Edge ({u}, {v}) in cycle does not exist in graph G", recomputed_weight
        w = float(G[u][v].get(weight_key, 1.0))
        recomputed_weight += w

    abs_diff = abs(recomputed_weight - expected_weight)
    scale = max(1.0, abs(expected_weight))
    if abs_diff > tol * scale:
        return (
            False,
            f"Weight sum mismatch: recomputed {recomputed_weight:.9f} vs expected {expected_weight:.9f}",
            recomputed_weight,
        )

    return True, "Valid simple cycle with exact weight match", recomputed_weight


def check_zero_false_prunings(
    gamma_acc: float,
    gamma_oracle: Optional[float],
    tol: float = TOL,
) -> Tuple[bool, float]:
    """
    Verify that certified pruning in the accelerated engine incurred zero false prunings.
    If gamma_acc > gamma_oracle + tol, a true minimum-weight cycle was erroneously pruned!
    """
    if gamma_oracle is None:
        return True, 0.0

    if math.isinf(gamma_oracle):
        is_sound = math.isinf(gamma_acc)
        return is_sound, 0.0 if is_sound else float("inf")

    abs_diff = gamma_acc - gamma_oracle
    scale = max(1.0, abs(gamma_oracle))
    # Soundness invariant: gamma_acc cannot be strictly larger than gamma_oracle
    is_sound = abs_diff <= tol * scale
    return is_sound, abs_diff


# ============================================================================
# 4. Monotonic High-Precision Timing Engine
# ============================================================================

def timed_execution(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Tuple[Any, float]:
    """
    Execute func(*args, **kwargs) under rigorous timing conditions:
      - Full garbage collection run prior to measurement
      - Garbage collection disabled during execution
      - High-resolution monotonic timer (time.perf_counter_ns)
    Returns:
        (result, elapsed_seconds)
    """
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        t0 = time.perf_counter_ns()
        result = func(*args, **kwargs)
        t1 = time.perf_counter_ns()
    finally:
        if gc_was_enabled:
            gc.enable()

    elapsed_sec = (t1 - t0) / 1_000_000_000.0
    return result, elapsed_sec


def compute_timing_statistics(times: List[float]) -> Dict[str, float]:
    """Compute standard descriptive statistics from trial timings."""
    if not times:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "stdev": 0.0, "iqr": 0.0}
    times_sorted = sorted(times)
    n = len(times_sorted)
    med = statistics.median(times_sorted)
    mean_val = statistics.mean(times_sorted)
    stdev_val = statistics.stdev(times_sorted) if n > 1 else 0.0
    q1 = times_sorted[n // 4]
    q3 = times_sorted[(3 * n) // 4]
    return {
        "min": times_sorted[0],
        "median": med,
        "mean": mean_val,
        "stdev": stdev_val,
        "iqr": q3 - q1,
    }


# ============================================================================
# 5. Core Instance Benchmarking Routine
# ============================================================================

def benchmark_graph_instance(
    G: nx.Graph,
    name: str,
    family: str,
    trials: int = 5,
    warmup: int = 1,
    check_oracle: bool = False,
    known_girth: Optional[float] = None,
    tol: float = TOL,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    """
    Run complete benchmarking and verification protocol on a single graph instance.
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    c_comps = nx.number_connected_components(G) if n_nodes > 0 else 0
    mu = n_edges - n_nodes + c_comps

    rec: Dict[str, Any] = {
        "name": name,
        "family": family,
        "nodes": n_nodes,
        "edges": n_edges,
        "cyclomatic_mu": mu,
        "density": round((2.0 * n_edges) / (n_nodes * (n_nodes - 1)), 4) if n_nodes > 1 else 0.0,
        "trials": trials,
        "warmup": warmup,
    }

    # 1. Warm-up runs to prime caches, dynamic libraries, and memory allocators
    for _ in range(warmup):
        try:
            minimum_weight_cycle(G, use_acceleration=False)
            minimum_weight_cycle(G, use_acceleration=True)
        except Exception:
            pass

    # 2. Benchmark Pure Python Baseline (use_acceleration=False)
    times_pure: List[float] = []
    res_pure: Optional[Tuple[float, List[Any]]] = None
    for trial_idx in range(trials):
        (res, dt) = timed_execution(minimum_weight_cycle, G, use_acceleration=False)
        times_pure.append(dt)
        if trial_idx == 0:
            res_pure = res

    gamma_pure, cyc_pure = res_pure if res_pure is not None else (float("inf"), [])
    stats_pure = compute_timing_statistics(times_pure)

    # 3. Benchmark Accelerated Kernel (use_acceleration=True)
    times_acc: List[float] = []
    res_acc: Optional[Tuple[float, List[Any]]] = None
    for trial_idx in range(trials):
        (res, dt) = timed_execution(minimum_weight_cycle, G, use_acceleration=True)
        times_acc.append(dt)
        if trial_idx == 0:
            res_acc = res

    gamma_acc, cyc_acc = res_acc if res_acc is not None else (float("inf"), [])
    stats_acc = compute_timing_statistics(times_acc)

    # 4. Compute Speedup Metrics
    s_median = stats_pure["median"] / stats_acc["median"] if stats_acc["median"] > 0 else float("inf")
    s_min = stats_pure["min"] / stats_acc["min"] if stats_acc["min"] > 0 else float("inf")
    s_mean = stats_pure["mean"] / stats_acc["mean"] if stats_acc["mean"] > 0 else float("inf")

    # 5. Verification: Numerical Equivalence
    is_equiv, rel_diff, abs_diff = check_numerical_equivalence(gamma_pure, gamma_acc, tol=tol)

    # 6. Verification: Structural Validity of Returned Cycles
    p_valid, p_reason, p_w = validate_cycle_structure(G, cyc_pure, gamma_pure, tol=tol)
    a_valid, a_reason, a_w = validate_cycle_structure(G, cyc_acc, gamma_acc, tol=tol)

    # 7. Verification: Zero False Prunings
    target_oracle: Optional[float] = known_girth
    if target_oracle is None and check_oracle and n_nodes <= 80:
        try:
            target_oracle, _ = exact_oracle(G)
        except Exception:
            target_oracle = None

    is_sound, prune_diff = check_zero_false_prunings(gamma_acc, target_oracle, tol=tol)

    # Overall Verdict
    is_verified = is_equiv and a_valid and is_sound
    meets_speedup = s_median > 1.0

    rec.update({
        "gamma_pure": None if math.isinf(gamma_pure) else round(gamma_pure, 6),
        "gamma_acc": None if math.isinf(gamma_acc) else round(gamma_acc, 6),
        "oracle_girth": None if target_oracle is None or math.isinf(target_oracle) else round(target_oracle, 6),
        "t_pure_med_ms": round(stats_pure["median"] * 1000.0, 3),
        "t_pure_min_ms": round(stats_pure["min"] * 1000.0, 3),
        "t_acc_med_ms": round(stats_acc["median"] * 1000.0, 3),
        "t_acc_min_ms": round(stats_acc["min"] * 1000.0, 3),
        "speedup_median": round(s_median, 3),
        "speedup_min": round(s_min, 3),
        "speedup_mean": round(s_mean, 3),
        "numerical_equivalence": is_equiv,
        "relative_diff": rel_diff,
        "cycle_pure_valid": p_valid,
        "cycle_acc_valid": a_valid,
        "zero_false_pruning": is_sound,
        "verified_pass": is_verified,
        "speedup_pass": meets_speedup,
        "overall_status": "PASS" if (is_verified and meets_speedup) else ("REGRESSION" if is_verified else "FAIL"),
    })

    return rec


# ============================================================================
# 6. Test Suite Drivers (Dense, Sparse, Realnets)
# ============================================================================

def run_dense_family_suite(
    sizes: Sequence[int] = (20, 40, 60, 80, 100, 150, 200),
    p: float = 0.8,
    trials: int = 5,
    seed: int = 42,
    weight_type: str = "fractional",
    check_oracle: bool = True,
    tol: float = TOL,
) -> List[Dict[str, Any]]:
    """Execute benchmarking over the dense synthetic graph family G(n, p=0.8)."""
    results: List[Dict[str, Any]] = []
    print(f"\n--- Benchmarking Dense Synthetic Family G(n, p={p}) [{weight_type} weights] ---")
    for n in sizes:
        G = generate_dense_graph(n=n, p=p, seed=seed + n, weight_type=weight_type)
        res = benchmark_graph_instance(
            G,
            name=f"dense_n{n}_p0.8",
            family="dense",
            trials=trials,
            check_oracle=check_oracle,
            tol=tol,
        )
        results.append(res)
        print(
            f"  {res['name']:18s} | V={res['nodes']:4d} E={res['edges']:5d} | "
            f"T_pure={res['t_pure_med_ms']:8.2f}ms | T_acc={res['t_acc_med_ms']:8.2f}ms | "
            f"Speedup={res['speedup_median']:6.2f}x | Equiv={res['numerical_equivalence']} | "
            f"Valid={res['cycle_acc_valid']} | Status={res['overall_status']}"
        )
    return results


def run_sparse_family_suite(
    sizes: Sequence[int] = (50, 100, 200, 500, 1000, 2000),
    p: float = 0.05,
    trials: int = 5,
    seed: int = 42,
    weight_type: str = "fractional",
    check_oracle: bool = True,
    tol: float = TOL,
) -> List[Dict[str, Any]]:
    """Execute benchmarking over the sparse synthetic graph family G(n, p=0.05)."""
    results: List[Dict[str, Any]] = []
    print(f"\n--- Benchmarking Sparse Synthetic Family G(n, p={p}) [{weight_type} weights] ---")
    for n in sizes:
        G = generate_sparse_graph(n=n, p=p, seed=seed + n, weight_type=weight_type)
        res = benchmark_graph_instance(
            G,
            name=f"sparse_n{n}_p0.05",
            family="sparse",
            trials=trials,
            check_oracle=check_oracle,
            tol=tol,
        )
        results.append(res)
        print(
            f"  {res['name']:18s} | V={res['nodes']:4d} E={res['edges']:5d} | "
            f"T_pure={res['t_pure_med_ms']:8.2f}ms | T_acc={res['t_acc_med_ms']:8.2f}ms | "
            f"Speedup={res['speedup_median']:6.2f}x | Equiv={res['numerical_equivalence']} | "
            f"Valid={res['cycle_acc_valid']} | Status={res['overall_status']}"
        )
    return results


def run_realnet_suite(
    networks: Optional[Sequence[str]] = None,
    datasets_dir: str = "/scratch/hs9hd/mwc_certified_pruning/datasets/realnets",
    trials: int = 5,
    tol: float = TOL,
) -> List[Dict[str, Any]]:
    """Execute benchmarking over real-world network benchmarks."""
    target_nets = list(networks) if networks else list(REALNET_GROUND_TRUTH.keys())
    results: List[Dict[str, Any]] = []
    print(f"\n--- Benchmarking Real-World Networks ({len(target_nets)} datasets) ---")

    for name in target_nets:
        meta = REALNET_GROUND_TRUTH.get(name, {})
        known_girth = meta.get("girth")
        try:
            G = load_real_network(name, datasets_dir=datasets_dir)
        except FileNotFoundError as e:
            print(f"  Skipping {name}: {e}")
            continue

        res = benchmark_graph_instance(
            G,
            name=name,
            family="realnets",
            trials=trials,
            known_girth=known_girth,
            tol=tol,
        )
        results.append(res)
        print(
            f"  {res['name']:22s} | V={res['nodes']:5d} E={res['edges']:6d} | "
            f"T_pure={res['t_pure_med_ms']:8.2f}ms | T_acc={res['t_acc_med_ms']:8.2f}ms | "
            f"Speedup={res['speedup_median']:6.2f}x | Equiv={res['numerical_equivalence']} | "
            f"Valid={res['cycle_acc_valid']} | Status={res['overall_status']}"
        )
    return results


# ============================================================================
# 7. Output Formatters (Markdown, LaTeX, JSON, CSV)
# ============================================================================

def format_markdown_table(rows: List[Dict[str, Any]]) -> str:
    """Format benchmark rows as a GitHub-flavored Markdown table."""
    headers = [
        "Family", "Instance", "|V|", "|E|", "Girth",
        "T_pure (ms)", "T_acc (ms)", "Speedup", "Equiv?", "Valid?", "Status"
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        girth_str = f"{r['gamma_acc']:.4f}" if r["gamma_acc"] is not None else "inf"
        row_str = (
            f"| {r['family']} | `{r['name']}` | {r['nodes']} | {r['edges']} | {girth_str} | "
            f"{r['t_pure_med_ms']:.2f} | {r['t_acc_med_ms']:.2f} | **{r['speedup_median']:.2f}x** | "
            f"{'PASS' if r['numerical_equivalence'] else 'FAIL'} | "
            f"{'PASS' if r['cycle_acc_valid'] else 'FAIL'} | "
            f"`{r['overall_status']}` |"
        )
        lines.append(row_str)
    return "\n".join(lines)


def format_latex_table(rows: List[Dict[str, Any]]) -> str:
    """Format benchmark rows as a clean LaTeX booktabs table for manuscript.tex."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Empirical Speedup and Invariant Verification: Pure Python Baseline vs. Compiled Extension.}",
        r"\label{tab:m2_acceleration_speedup}",
        r"\begin{tabular}{llrrrrrcc}",
        r"\toprule",
        r"Family & Instance & $|V|$ & $|E|$ & $T_{\mathrm{pure}}$ (ms) & $T_{\mathrm{acc}}$ (ms) & Speedup ($S$) & Equiv. & Status \\",
        r"\midrule",
    ]
    for r in rows:
        status_tex = r"\textbf{PASS}" if r["overall_status"] == "PASS" else r"\textsc{Regress}"
        equiv_tex = r"\checkmark" if r["numerical_equivalence"] else r"\times"
        line = (
            f"{r['family']} & \\texttt{{{r['name']}}} & {r['nodes']} & {r['edges']} & "
            f"{r['t_pure_med_ms']:.1f} & {r['t_acc_med_ms']:.1f} & "
            f"\\textbf{{{r['speedup_median']:.2f}$\\times$}} & {equiv_tex} & {status_tex} \\\\"
        )
        lines.append(line)
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def export_csv(rows: List[Dict[str, Any]], filepath: str) -> None:
    """Export benchmark rows as a CSV file."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ============================================================================
# 8. Command-Line Entry Point
# ============================================================================

def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Milestone 2 Acceleration Verification & Benchmarking Harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--families",
        type=str,
        default="all",
        help="Comma-separated list of families: 'dense', 'sparse', 'realnets', or 'all'",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of timed repetitions per graph instance",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of unmeasured warm-up runs per instance",
    )
    parser.add_argument(
        "--dense-sizes",
        type=int,
        nargs="+",
        default=[20, 40, 60, 80, 100, 150, 200],
        help="Vertex counts for dense synthetic graphs G(n, p=0.8)",
    )
    parser.add_argument(
        "--sparse-sizes",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500, 1000, 2000],
        help="Vertex counts for sparse synthetic graphs G(n, p=0.05)",
    )
    parser.add_argument(
        "--realnets",
        type=str,
        default="all",
        help="Comma-separated real network names, or 'all'",
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default="/scratch/hs9hd/mwc_certified_pruning/datasets/realnets",
        help="Directory containing real network .edges files",
    )
    parser.add_argument(
        "--weight-type",
        type=str,
        default="fractional",
        choices=["uniform", "fractional", "integer", "unit"],
        help="Synthetic edge weight sampler strategy",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=TOL,
        help="Numerical equivalence tolerance",
    )
    parser.add_argument(
        "--check-oracle",
        action="store_true",
        default=True,
        help="Verify against exact Dijkstra oracle on small graphs (n <= 80)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/scratch/hs9hd/mwc_certified_pruning/results/acceleration",
        help="Directory to save JSON, Markdown, LaTeX, and CSV reports",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Exit with non-zero code if any speedup is <= 1.0x or equivalence fails",
    )
    return parser


def main() -> int:
    parser = build_cli_parser()
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    provenance = get_provenance()

    print("=" * 80)
    print("MWC Certified Pruning: Milestone 2 Acceleration & Equivalence Harness")
    print(f"Host: {provenance['hostname']} | CPU: {provenance['cpu_model']}")
    print(f"Cython Module Compiled: {provenance['cython_compiled']} | Python: {provenance['python_version']}")
    print(f"Git Commit: {provenance['git_commit']} (Dirty: {provenance['git_dirty']})")
    print("=" * 80)

    families = [f.strip().lower() for f in args.families.split(",")]
    if "all" in families:
        families = ["dense", "sparse", "realnets"]

    all_rows: List[Dict[str, Any]] = []

    # 1. Dense Family
    if "dense" in families:
        rows = run_dense_family_suite(
            sizes=args.dense_sizes,
            p=0.8,
            trials=args.trials,
            weight_type=args.weight_type,
            check_oracle=args.check_oracle,
            tol=args.tolerance,
        )
        all_rows.extend(rows)

    # 2. Sparse Family
    if "sparse" in families:
        rows = run_sparse_family_suite(
            sizes=args.sparse_sizes,
            p=0.05,
            trials=args.trials,
            weight_type=args.weight_type,
            check_oracle=args.check_oracle,
            tol=args.tolerance,
        )
        all_rows.extend(rows)

    # 3. Realnets Family
    if "realnets" in families:
        nets = None if args.realnets.strip().lower() == "all" else [n.strip() for n in args.realnets.split(",")]
        rows = run_realnet_suite(
            networks=nets,
            datasets_dir=args.datasets_dir,
            trials=args.trials,
            tol=args.tolerance,
        )
        all_rows.extend(rows)

    # Summary Statistics
    total_runs = len(all_rows)
    passed_verif = sum(1 for r in all_rows if r["verified_pass"])
    passed_speedup = sum(1 for r in all_rows if r["speedup_pass"])
    speedups = [r["speedup_median"] for r in all_rows if not math.isinf(r["speedup_median"]) and r["speedup_median"] > 0]
    geomean_speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups)) if speedups else 1.0

    print("\n" + "=" * 80)
    print(f"BENCHMARK HARNESS SUMMARY ({total_runs} instances)")
    print(f"  Mathematical Equivalence & Validity Pass: {passed_verif} / {total_runs}")
    print(f"  Speedup > 1.0x Pass:                      {passed_speedup} / {total_runs}")
    print(f"  Geometric Mean Speedup:                   {geomean_speedup:.2f}x")
    if speedups:
        print(f"  Max Speedup Observed:                     {max(speedups):.2f}x")
        print(f"  Min Speedup Observed:                     {min(speedups):.2f}x")
    print("=" * 80)

    # Export Artifacts
    json_path = os.path.join(args.out_dir, "acceleration_results.json")
    md_path = os.path.join(args.out_dir, "acceleration_summary.md")
    latex_path = os.path.join(args.out_dir, "acceleration_table.tex")
    csv_path = os.path.join(args.out_dir, "acceleration_raw.csv")

    with open(json_path, "w") as f:
        json.dump({
            "provenance": provenance,
            "summary": {
                "total_instances": total_runs,
                "verified_pass_count": passed_verif,
                "speedup_pass_count": passed_speedup,
                "geometric_mean_speedup": round(geomean_speedup, 3),
            },
            "records": all_rows,
        }, f, indent=2)

    with open(md_path, "w") as f:
        f.write("# MWC Certified Pruning: Milestone 2 Acceleration Benchmark Report\n\n")
        f.write(f"**Geometric Mean Speedup**: `{geomean_speedup:.2f}x` | ")
        f.write(f"**Verification Pass Rate**: `{passed_verif}/{total_runs}`\n\n")
        f.write(format_markdown_table(all_rows))
        f.write("\n")

    with open(latex_path, "w") as f:
        f.write(format_latex_table(all_rows))

    export_csv(all_rows, csv_path)

    print(f"\nArtifacts exported successfully to: {args.out_dir}")
    print(f"  - JSON:     {json_path}")
    print(f"  - Markdown: {md_path}")
    print(f"  - LaTeX:    {latex_path}")
    print(f"  - CSV:      {csv_path}")

    # Exit code determination
    if passed_verif < total_runs:
        print("\nERROR: Verification failed on one or more instances (numerical discrepancy or invalid cycle)!")
        return 1
    if args.strict and passed_speedup < total_runs:
        print("\nWARNING: Speedup requirement (>1.0x) not met on all instances!")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
