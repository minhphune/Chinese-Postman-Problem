from __future__ import annotations

import itertools
import math
import random

import pytest

from adaptive_cpp_robot import (
    Edge,
    example_edges,
    fleury_undirected,
    hierholzer_undirected,
    make_anchor,
    regret_certificate,
    solve_undirected_cpp,
)
from directed_cpp import (
    Arc,
    example_arcs,
    min_cost_transport,
    solve_directed_cpp,
)


def _route_instance_set(result: object) -> list[int]:
    return sorted(getattr(result, "instance_ids"))


def test_undirected_reference_example() -> None:
    result = solve_undirected_cpp(example_edges(), "A")
    assert math.isclose(result.base_cost, 36.0)
    assert math.isclose(result.matching_cost, 7.0)
    assert math.isclose(result.total_cost, 43.0)
    assert result.vertices[0] == result.vertices[-1] == "A"
    assert _route_instance_set(result) == list(range(len(result.instances)))
    assert result.multiplicity["AB"] == 2
    assert result.multiplicity["CD"] == 2


def test_fleury_and_hierholzer_cover_the_same_instances() -> None:
    result = solve_undirected_cpp(example_edges(), "A")
    hv, he = hierholzer_undirected(result.instances, "A")
    fv, fe = fleury_undirected(result.instances, "A")
    assert hv[0] == hv[-1] == "A"
    assert fv[0] == fv[-1] == "A"
    assert sorted(he) == sorted(fe) == list(range(len(result.instances)))


def test_already_eulerian_undirected_graph_has_zero_augmentation() -> None:
    edges = [
        Edge("AB", "A", "B", 2.0),
        Edge("BC", "B", "C", 3.0),
        Edge("CA", "C", "A", 4.0),
    ]
    result = solve_undirected_cpp(edges, "A")
    assert result.matching == []
    assert math.isclose(result.matching_cost, 0.0)
    assert math.isclose(result.total_cost, 9.0)
    assert all(value == 1 for value in result.multiplicity.values())


def test_parallel_edges_and_zero_weight_are_preserved_by_identity() -> None:
    edges = [
        Edge("AB0", "A", "B", 0.0),
        Edge("AB2", "A", "B", 2.0),
        Edge("BC", "B", "C", 1.0),
        Edge("CA", "C", "A", 1.0),
    ]
    result = solve_undirected_cpp(edges, "A")
    assert math.isclose(result.base_cost, 4.0)
    assert math.isclose(result.matching_cost, 0.0)
    assert result.multiplicity["AB0"] == 2
    assert result.multiplicity["AB2"] == 1
    assert _route_instance_set(result) == list(range(len(result.instances)))


def test_directed_reference_example() -> None:
    result = solve_directed_cpp(example_arcs(), "A")
    assert result.imbalance["C"] == 1
    assert result.imbalance["A"] == -1
    assert result.transportation == {("C", "A"): 1}
    assert math.isclose(result.base_cost, 7.0)
    assert math.isclose(result.augmentation_cost, 2.0)
    assert math.isclose(result.total_cost, 9.0)
    assert result.vertices[0] == result.vertices[-1] == "A"
    assert _route_instance_set(result) == list(range(len(result.instances)))


def test_already_balanced_directed_graph_has_zero_augmentation() -> None:
    arcs = [
        Arc("AB", "A", "B", 1.0),
        Arc("BC", "B", "C", 1.0),
        Arc("CA", "C", "A", 1.0),
    ]
    result = solve_directed_cpp(arcs, "A")
    assert result.transportation == {}
    assert math.isclose(result.augmentation_cost, 0.0)
    assert math.isclose(result.total_cost, 3.0)


def test_two_by_two_transport_matches_brute_force() -> None:
    positive = ["p1", "p2"]
    negative = ["n1", "n2"]
    b = {"p1": 2, "p2": 1, "n1": -1, "n2": -2}
    distances = {
        ("p1", "n1"): 1.0,
        ("p1", "n2"): 4.0,
        ("p2", "n1"): 3.0,
        ("p2", "n2"): 1.0,
    }
    flow, cost = min_cost_transport(positive, negative, b, distances)

    feasible_costs = []
    for f11 in range(3):
        f12 = 2 - f11
        f21 = 1 - f11
        f22 = 1 - f21
        if min(f12, f21, f22) >= 0 and f11 + f21 == 1 and f12 + f22 == 2:
            feasible_costs.append(f11 + 4 * f12 + 3 * f21 + f22)
    assert math.isclose(cost, min(feasible_costs))
    assert sum(flow.get((p, n), 0) for n in negative for p in ["p1"]) == 2
    assert sum(flow.get((p, n), 0) for n in negative for p in ["p2"]) == 1


def test_random_strongly_connected_dcpp_instances() -> None:
    rng = random.Random(20260722)
    for n in range(3, 9):
        arcs = [Arc(f"r{i}", i, (i + 1) % n, rng.uniform(0.0, 5.0)) for i in range(n)]
        next_id = n
        for _ in range(n // 2 + 1):
            u, v = rng.sample(range(n), 2)
            arcs.append(Arc(f"x{next_id}", u, v, rng.uniform(0.0, 5.0)))
            next_id += 1
        result = solve_directed_cpp(arcs, 0)
        assert result.vertices[0] == result.vertices[-1] == 0
        assert _route_instance_set(result) == list(range(len(result.instances)))
        assert result.total_cost + 1e-9 >= result.base_cost


def test_certificate_dominates_true_regret_on_random_updates() -> None:
    rng = random.Random(20260722)
    base = example_edges()
    anchor_result = solve_undirected_cpp(base, "A")
    anchor = make_anchor(base, anchor_result)
    for _ in range(80):
        current = {e.base_id: e.weight * rng.uniform(0.80, 1.25) for e in base}
        cert = regret_certificate(anchor, current)
        changed = [Edge(e.base_id, e.u, e.v, current[e.base_id]) for e in base]
        optimum = solve_undirected_cpp(changed, "A").total_cost
        true_regret = cert["keep_cost"] / optimum - 1.0
        assert true_regret <= cert["relative_regret_bound"] + 1e-9


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        solve_undirected_cpp([Edge("loop", "A", "A", 1.0)], "A")
    with pytest.raises(ValueError):
        solve_undirected_cpp([Edge("neg", "A", "B", -1.0)], "A")
    with pytest.raises(ValueError):
        solve_directed_cpp([Arc("loop", "A", "A", 1.0)], "A")
    with pytest.raises(ValueError):
        solve_directed_cpp([Arc("AB", "A", "B", 1.0)], "A")
