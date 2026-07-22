from __future__ import annotations

from dataclasses import dataclass
import csv
import random
import statistics
import time
from pathlib import Path
from typing import Dict, List

from adaptive_cpp_robot import (
    Edge,
    _make_instances,
    blossom_matching,
    fleury_undirected,
    hierholzer_undirected,
    make_anchor,
    metric_closure,
    odd_vertices,
    regret_certificate,
    restore_path_edges,
    solve_undirected_cpp,
)


def generate_graph(n: int, q: int, extra_pairs: int, seed: int) -> List[Edge]:
    """Generate a connected multigraph with exactly q odd vertices."""
    if q % 2 or q > n:
        raise ValueError("q must be even and no larger than n")
    rng = random.Random(seed)
    edges: List[Edge] = []
    eid = 0

    # A Hamiltonian cycle guarantees connectivity and starts with even degrees.
    for u in range(n):
        v = (u + 1) % n
        edges.append(Edge(f"e{eid}", u, v, rng.uniform(4.0, 18.0)))
        eid += 1

    # Each extra single edge toggles parity at exactly its two endpoints.
    odd_targets = rng.sample(range(n), q)
    rng.shuffle(odd_targets)
    for i in range(0, q, 2):
        u, v = odd_targets[i], odd_targets[i + 1]
        edges.append(Edge(f"e{eid}", u, v, rng.uniform(4.0, 18.0)))
        eid += 1

    # Add edges in pairs so the prescribed parity set is unchanged.
    for _ in range(extra_pairs):
        u, v = rng.sample(range(n), 2)
        for _copy in range(2):
            edges.append(Edge(f"e{eid}", u, v, rng.uniform(4.0, 18.0)))
            eid += 1

    assert len(odd_vertices(edges)) == q
    return edges


def _percentile(values: List[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = p * (len(ordered) - 1)
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    frac = index - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _summary(values: List[float]) -> tuple[float, float]:
    return statistics.median(values), _percentile(values, 0.75) - _percentile(values, 0.25)


def _one_static_run(edges: List[Edge]) -> Dict[str, float]:
    start = time.perf_counter_ns()
    odd = odd_vertices(edges)
    distances, predecessors = metric_closure(edges, odd)
    metric_ms = (time.perf_counter_ns() - start) / 1e6

    start = time.perf_counter_ns()
    matching, _ = blossom_matching(odd, distances)
    matching_ms = (time.perf_counter_ns() - start) / 1e6

    duplicated: List[int] = []
    for u, v in matching:
        duplicated.extend(restore_path_edges(predecessors[u], u, v))
    instances = _make_instances(edges, duplicated)

    start = time.perf_counter_ns()
    hierholzer_undirected(instances, 0)
    hierholzer_ms = (time.perf_counter_ns() - start) / 1e6

    start = time.perf_counter_ns()
    fleury_undirected(instances, 0)
    fleury_ms = (time.perf_counter_ns() - start) / 1e6

    start = time.perf_counter_ns()
    solve_undirected_cpp(edges, 0, verify_small_matching=False)
    total_ms = (time.perf_counter_ns() - start) / 1e6

    return {
        "m_prime": float(len(instances)),
        "metric_ms": metric_ms,
        "matching_ms": matching_ms,
        "hierholzer_ms": hierholzer_ms,
        "fleury_ms": fleury_ms,
        "total_ms": total_ms,
    }


def static_benchmark(output: Path, repeats: int = 15) -> List[Dict[str, float]]:
    """Report medians and IQRs after one untimed warm-up per configuration."""
    configs = [
        (50, 4, 25),
        (100, 8, 50),
        (200, 16, 100),
        (400, 24, 200),
    ]
    rows: List[Dict[str, float]] = []

    for n, q, extra_pairs in configs:
        warmup = generate_graph(n, q, extra_pairs, seed=9000 + n)
        _one_static_run(warmup)

        samples: Dict[str, List[float]] = {
            "m_prime": [],
            "metric_ms": [],
            "matching_ms": [],
            "hierholzer_ms": [],
            "fleury_ms": [],
            "total_ms": [],
        }
        m = 0
        for rep in range(repeats):
            edges = generate_graph(n, q, extra_pairs, seed=1000 + 31 * n + rep)
            m = len(edges)
            result = _one_static_run(edges)
            for key, value in result.items():
                samples[key].append(value)

        row: Dict[str, float] = {"n": n, "m": m, "q": q, "repeats": repeats}
        row["m_prime"] = statistics.mean(samples["m_prime"])
        for key in ("metric_ms", "matching_ms", "hierholzer_ms", "fleury_ms", "total_ms"):
            median, iqr = _summary(samples[key])
            row[key] = median
            row[f"{key}_iqr"] = iqr
        rows.append(row)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def mutate_weights(edges: List[Edge], rng: random.Random, step: int) -> List[Edge]:
    new: List[Edge] = []
    chosen = set(rng.sample(range(len(edges)), max(1, len(edges) // 7)))
    shock = rng.randrange(len(edges)) if step % 8 == 0 else None
    shortcut = rng.randrange(len(edges)) if step % 11 == 0 else None
    for i, edge in enumerate(edges):
        factor = rng.uniform(0.95, 1.08) if i in chosen else 1.0
        if i == shock:
            factor *= rng.uniform(1.25, 1.55)
        if i == shortcut:
            factor *= rng.uniform(0.65, 0.82)
        new.append(Edge(edge.base_id, edge.u, edge.v, max(0.05, edge.weight * factor)))
    return new


@dataclass
class PolicyState:
    name: str
    result: object
    anchor: object
    replans: int = 0
    regrets: List[float] | None = None
    bounds: List[float] | None = None

    def __post_init__(self) -> None:
        self.regrets = []
        self.bounds = []


def route_cost(multiplicity: Dict[str, int], weights: Dict[str, float]) -> float:
    return sum(multiplicity[e] * weights[e] for e in multiplicity)


def dynamic_benchmark(output: Path) -> List[Dict[str, float]]:
    rng = random.Random(20260722)
    current = generate_graph(48, 10, 20, seed=77)
    initial = solve_undirected_cpp(current, 0, verify_small_matching=False)
    policies = {
        "always": PolicyState("always", initial, make_anchor(current, initial)),
        "local_0.15": PolicyState("local_0.15", initial, make_anchor(current, initial)),
        "certificate_0.05": PolicyState("certificate_0.05", initial, make_anchor(current, initial)),
        "certificate_0.10": PolicyState("certificate_0.10", initial, make_anchor(current, initial)),
    }
    steps = 50
    for step in range(1, steps + 1):
        current = mutate_weights(current, rng, step)
        current_weights = {e.base_id: e.weight for e in current}
        optimum = solve_undirected_cpp(current, 0, verify_small_matching=False)
        for name, state in policies.items():
            assert state.regrets is not None and state.bounds is not None
            do_replan = False
            bound = 0.0
            if name == "always":
                do_replan = True
            elif name.startswith("local"):
                max_relative = max(
                    abs(current_weights[e] - state.anchor.weights[e])
                    / max(state.anchor.weights[e], 1e-12)
                    for e in current_weights
                )
                do_replan = max_relative >= 0.15
            else:
                cert = regret_certificate(state.anchor, current_weights)
                bound = float(cert["relative_regret_bound"])
                tolerance = 0.05 if name.endswith("0.05") else 0.10
                do_replan = bound > tolerance
            if do_replan:
                state.result = optimum
                state.anchor = make_anchor(current, optimum)
                state.replans += 1
                bound = 0.0
            keep_cost = route_cost(state.result.multiplicity, current_weights)
            regret = keep_cost / optimum.total_cost - 1.0
            state.regrets.append(regret)
            state.bounds.append(bound)

    rows: List[Dict[str, float]] = []
    for state in policies.values():
        assert state.regrets is not None and state.bounds is not None
        rows.append({
            "policy": state.name,
            "steps": steps,
            "replans": state.replans,
            "mean_regret_pct": 100 * statistics.mean(state.regrets),
            "max_regret_pct": 100 * max(state.regrets),
            "mean_certificate_pct": 100 * statistics.mean(state.bounds),
            "max_certificate_pct": 100 * max(state.bounds),
        })

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    out = Path("results")
    out.mkdir(exist_ok=True)
    static = static_benchmark(out / "static.csv")
    dynamic = dynamic_benchmark(out / "dynamic.csv")
    print("STATIC")
    for row in static:
        print(row)
    print("DYNAMIC")
    for row in dynamic:
        print(row)
