"""Tier 4: Real-World Benchmark Workloads (>= 5 real-world networks).

Evaluates certified pruning and exact transversal root reduction on validated
real-world network benchmarks from /scratch/hs9hd/mwc_certified_pruning/datasets/realnets/:
1. lesmis (Les Misérables co-appearance: n=77, m=254)
2. celegans-neural (C. elegans neural wiring: n=297, m=2148)
3. chicago-sketch-road (Chicago sketch road network: n=933, m=1475)
4. rome99-road (Rome road network: n=3353, m=4831)
5. uspowergrid-synth (US Western Power Grid: n=4941, m=6594)
6. usairport-2010 (US Airport 2010 route network: n=1572, m=17214)
7. openflights-air (OpenFlights global flight routes: n=3188, m=18833)
8. Multi-network transversal reduction and equivalence property test
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import pytest

from tests.e2e.conftest import (
    INF,
    TOL,
    approx_eq,
    assert_valid_cycle,
    cycle_weight,
    is_simple_cycle,
    load_realnet_adj,
    mwc,
    mwc_transversal,
    solve_mwc,
)


def test_t4_realnet_lesmis():
    """Les Misérables co-appearance network (n=77, m=254)."""
    adj = load_realnet_adj("lesmis")
    assert len(adj) == 77
    res = mwc_transversal(adj, collect_stats=True)
    assert not math.isinf(res.length)
    assert approx_eq(res.length, 3.0)
    assert is_simple_cycle(adj, res.cycle)
    assert_valid_cycle(adj, res.cycle, 3.0)
    # Verify root reduction
    assert res.stats["transversal_size"] <= len(adj)


def test_t4_realnet_celegans_neural():
    """C. elegans neural connectivity network (n=297, m=2148)."""
    adj = load_realnet_adj("celegans-neural")
    assert len(adj) == 297
    res = mwc_transversal(adj, collect_stats=True)
    assert not math.isinf(res.length)
    assert approx_eq(res.length, 3.0)
    assert is_simple_cycle(adj, res.cycle)
    assert_valid_cycle(adj, res.cycle, 3.0)
    assert res.stats["transversal_size"] <= len(adj)


def test_t4_realnet_chicago_sketch_road():
    """Chicago sketch road network (n=933, m=1475)."""
    adj = load_realnet_adj("chicago-sketch-road")
    assert len(adj) == 933
    res = mwc_transversal(adj, collect_stats=True)
    assert not math.isinf(res.length)
    assert approx_eq(res.length, 2.03239, tol=1e-4)
    assert is_simple_cycle(adj, res.cycle)
    assert_valid_cycle(adj, res.cycle, res.length)


def test_t4_realnet_rome99_road():
    """Rome road network (n=3353, m=4831)."""
    adj = load_realnet_adj("rome99-road")
    assert len(adj) == 3353
    res = mwc_transversal(adj, collect_stats=True)
    assert not math.isinf(res.length)
    assert approx_eq(res.length, 6.0)
    assert is_simple_cycle(adj, res.cycle)
    assert_valid_cycle(adj, res.cycle, 6.0)


def test_t4_realnet_uspowergrid_synth():
    """US Western Power Grid synthetic network (n=4941, m=6594)."""
    adj = load_realnet_adj("uspowergrid-synth")
    assert len(adj) == 4941
    res = mwc_transversal(adj, collect_stats=True)
    assert not math.isinf(res.length)
    assert approx_eq(res.length, 379.0)
    assert is_simple_cycle(adj, res.cycle)
    assert_valid_cycle(adj, res.cycle, 379.0)


def test_t4_realnet_usairport_2010():
    """US Airport 2010 route network (n=1572, m=17214)."""
    adj = load_realnet_adj("usairport-2010")
    assert len(adj) == 1572
    res = mwc_transversal(adj, collect_stats=True)
    assert not math.isinf(res.length)
    assert approx_eq(res.length, 3.0)
    assert is_simple_cycle(adj, res.cycle)
    assert_valid_cycle(adj, res.cycle, 3.0)


def test_t4_realnet_openflights_air():
    """OpenFlights airport routes network (n=3188, m=18833)."""
    adj = load_realnet_adj("openflights-air")
    assert len(adj) == 3188
    res = mwc_transversal(adj, collect_stats=True)
    assert not math.isinf(res.length)
    assert approx_eq(res.length, 32.7415686, tol=1e-4)
    assert is_simple_cycle(adj, res.cycle)
    assert_valid_cycle(adj, res.cycle, res.length)


def test_t4_realnet_transversal_speedup_and_equivalence():
    """Verifies transversal reduction matches full root MWC exactly on real networks."""
    # Test on lesmis and chicago-sketch-road
    for net in ["lesmis", "chicago-sketch-road"]:
        adj = load_realnet_adj(net)
        res_trans = mwc_transversal(adj, collect_stats=True)
        res_full = mwc(adj, collect_stats=True)
        assert approx_eq(res_trans.length, res_full.length)
        # Roots run in transversal must be <= roots in full
        assert res_trans.stats["roots_run"] <= res_full.stats["roots_run"]
