"""Exact undirected CPP and certificate-guided dynamic replanning.

The implementation keeps two identities:
- base_id: physical corridor/edge;
- instance_id: one concrete copy in the Eulerized multigraph.

Requires NetworkX only for Edmonds' weighted matching backend.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
import heapq
import math
from typing import Dict, Hashable, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import networkx as nx

Vertex = Hashable


@dataclass(frozen=True)
class Edge:
    base_id: str
    u: Vertex
    v: Vertex
    weight: float


@dataclass(frozen=True)
class EdgeInstance:
    instance_id: int
    base_id: str
    u: Vertex
    v: Vertex
    duplicated: bool


@dataclass
class UCPPResult:
    vertices: List[Vertex]
    instance_ids: List[int]
    instances: List[EdgeInstance]
    matching: List[Tuple[Vertex, Vertex]]
    matching_cost: float
    base_cost: float
    total_cost: float
    multiplicity: Dict[str, int]


def _validate_edges(edges: Sequence[Edge]) -> None:
    if not edges:
        raise ValueError("The graph must contain at least one edge.")
    seen: set[str] = set()
    for edge in edges:
        if edge.base_id in seen:
            raise ValueError(f"Duplicate base_id: {edge.base_id}")
        seen.add(edge.base_id)
        if edge.u == edge.v:
            raise ValueError("Loops are excluded in this implementation.")
        if not math.isfinite(edge.weight) or edge.weight < 0:
            raise ValueError(f"Invalid nonnegative weight on {edge.base_id}.")


def _base_adjacency(edges: Sequence[Edge]) -> Dict[Vertex, List[int]]:
    adj: Dict[Vertex, List[int]] = defaultdict(list)
    for i, edge in enumerate(edges):
        adj[edge.u].append(i)
        adj[edge.v].append(i)
    return dict(adj)


def _other(edge: Edge | EdgeInstance, v: Vertex) -> Vertex:
    if edge.u == v:
        return edge.v
    if edge.v == v:
        return edge.u
    raise ValueError("The edge is not incident with the given vertex.")


def connected_vertices(edges: Sequence[Edge], start: Vertex | None = None) -> set[Vertex]:
    adj = _base_adjacency(edges)
    if not adj:
        return set()
    root = start if start is not None else next(iter(adj))
    if root not in adj:
        return {root}
    seen = {root}
    queue = deque([root])
    while queue:
        v = queue.popleft()
        for edge_id in adj[v]:
            u = _other(edges[edge_id], v)
            if u not in seen:
                seen.add(u)
                queue.append(u)
    return seen


def odd_vertices(edges: Sequence[Edge]) -> List[Vertex]:
    degree: Counter[Vertex] = Counter()
    for edge in edges:
        degree[edge.u] += 1
        degree[edge.v] += 1
    return sorted((v for v, d in degree.items() if d % 2 == 1), key=str)


def dijkstra(
    edges: Sequence[Edge], source: Vertex
) -> Tuple[Dict[Vertex, float], Dict[Vertex, Tuple[Vertex, int]]]:
    """Single-source shortest paths with predecessor edge identities."""
    adj = _base_adjacency(edges)
    if source not in adj:
        raise ValueError(f"Unknown or isolated source vertex: {source}")
    dist: Dict[Vertex, float] = {v: math.inf for v in adj}
    pred: Dict[Vertex, Tuple[Vertex, int]] = {}
    dist[source] = 0.0
    pq: List[Tuple[float, int, Vertex]] = [(0.0, 0, source)]
    serial = 1
    while pq:
        dv, _, v = heapq.heappop(pq)
        if dv != dist[v]:
            continue
        for edge_id in adj[v]:
            edge = edges[edge_id]
            u = _other(edge, v)
            candidate = dv + edge.weight
            if candidate + 1e-12 < dist[u]:
                dist[u] = candidate
                pred[u] = (v, edge_id)
                heapq.heappush(pq, (candidate, serial, u))
                serial += 1
    return dist, pred


def restore_path_edges(
    predecessor: Mapping[Vertex, Tuple[Vertex, int]], source: Vertex, target: Vertex
) -> List[int]:
    if source == target:
        return []
    path: List[int] = []
    current = target
    while current != source:
        if current not in predecessor:
            raise ValueError(f"No path from {source} to {target}.")
        previous, edge_id = predecessor[current]
        path.append(edge_id)
        current = previous
    path.reverse()
    return path


def metric_closure(
    edges: Sequence[Edge], terminals: Sequence[Vertex]
) -> Tuple[Dict[Tuple[Vertex, Vertex], float], Dict[Vertex, Dict[Vertex, Tuple[Vertex, int]]]]:
    distances: Dict[Tuple[Vertex, Vertex], float] = {}
    predecessors: Dict[Vertex, Dict[Vertex, Tuple[Vertex, int]]] = {}
    for source in terminals:
        dist, pred = dijkstra(edges, source)
        predecessors[source] = pred
        for target in terminals:
            if source != target:
                if not math.isfinite(dist.get(target, math.inf)):
                    raise ValueError("The required graph is disconnected.")
                distances[(source, target)] = dist[target]
    return distances, predecessors


def blossom_matching(
    terminals: Sequence[Vertex], distances: Mapping[Tuple[Vertex, Vertex], float]
) -> Tuple[List[Tuple[Vertex, Vertex]], float]:
    if len(terminals) % 2:
        raise ValueError("A perfect matching requires an even number of terminals.")
    closure = nx.Graph()
    closure.add_nodes_from(terminals)
    for i, u in enumerate(terminals):
        for v in terminals[i + 1 :]:
            closure.add_edge(u, v, weight=float(distances[(u, v)]))
    raw = nx.algorithms.matching.min_weight_matching(closure, weight="weight")
    matching = [tuple(sorted((u, v), key=str)) for u, v in raw]
    matching.sort(key=lambda pair: (str(pair[0]), str(pair[1])))
    if len(matching) * 2 != len(terminals):
        raise RuntimeError("The matching backend did not return a perfect matching.")
    cost = sum(distances[(u, v)] for u, v in matching)
    return matching, cost


def bitmask_matching_cost(
    terminals: Sequence[Vertex], distances: Mapping[Tuple[Vertex, Vertex], float]
) -> float:
    q = len(terminals)
    if q % 2:
        raise ValueError("The number of terminals must be even.")

    @lru_cache(maxsize=None)
    def solve(mask: int) -> float:
        if mask == 0:
            return 0.0
        i = (mask & -mask).bit_length() - 1
        without_i = mask ^ (1 << i)
        best = math.inf
        remaining = without_i
        while remaining:
            j = (remaining & -remaining).bit_length() - 1
            u, v = terminals[i], terminals[j]
            best = min(best, distances[(u, v)] + solve(without_i ^ (1 << j)))
            remaining ^= 1 << j
        return best

    return solve((1 << q) - 1)


def _make_instances(
    edges: Sequence[Edge], duplicated_edge_ids: Iterable[int]
) -> List[EdgeInstance]:
    instances: List[EdgeInstance] = []
    for edge in edges:
        instances.append(
            EdgeInstance(len(instances), edge.base_id, edge.u, edge.v, duplicated=False)
        )
    for edge_id in duplicated_edge_ids:
        edge = edges[edge_id]
        instances.append(
            EdgeInstance(len(instances), edge.base_id, edge.u, edge.v, duplicated=True)
        )
    return instances


def hierholzer_undirected(
    instances: Sequence[EdgeInstance], start: Vertex
) -> Tuple[List[Vertex], List[int]]:
    adj: Dict[Vertex, List[int]] = defaultdict(list)
    for inst in instances:
        adj[inst.u].append(inst.instance_id)
        adj[inst.v].append(inst.instance_id)
    if start not in adj:
        raise ValueError("The start vertex is not incident with any edge instance.")
    cursor: Dict[Vertex, int] = defaultdict(int)
    used = [False] * len(instances)
    stack_v: List[Vertex] = [start]
    stack_e: List[int] = []
    reverse_vertices: List[Vertex] = []
    reverse_edges: List[int] = []

    while stack_v:
        v = stack_v[-1]
        incident = adj[v]
        while cursor[v] < len(incident) and used[incident[cursor[v]]]:
            cursor[v] += 1
        if cursor[v] == len(incident):
            reverse_vertices.append(stack_v.pop())
            if stack_e:
                reverse_edges.append(stack_e.pop())
            continue
        instance_id = incident[cursor[v]]
        cursor[v] += 1
        if used[instance_id]:
            continue
        used[instance_id] = True
        inst = instances[instance_id]
        stack_v.append(_other(inst, v))
        stack_e.append(instance_id)

    vertices = list(reversed(reverse_vertices))
    edge_ids = list(reversed(reverse_edges))
    if len(edge_ids) != len(instances) or len(vertices) != len(instances) + 1:
        raise RuntimeError("Hierholzer did not consume every edge instance.")
    if vertices[0] != start or vertices[-1] != start:
        raise RuntimeError("The resulting Euler walk is not closed at the depot.")
    return vertices, edge_ids


def _reachable_without_instance(
    instances: Sequence[EdgeInstance], used: Sequence[bool], source: Vertex, target: Vertex, skip: int
) -> bool:
    adj: Dict[Vertex, List[int]] = defaultdict(list)
    for inst in instances:
        if not used[inst.instance_id] and inst.instance_id != skip:
            adj[inst.u].append(inst.instance_id)
            adj[inst.v].append(inst.instance_id)
    queue = deque([source])
    seen = {source}
    while queue:
        v = queue.popleft()
        if v == target:
            return True
        for instance_id in adj[v]:
            u = _other(instances[instance_id], v)
            if u not in seen:
                seen.add(u)
                queue.append(u)
    return target in seen


def fleury_undirected(
    instances: Sequence[EdgeInstance], start: Vertex
) -> Tuple[List[Vertex], List[int]]:
    """Pedagogical O(m^2) implementation with exact edge-instance handling."""
    adj: Dict[Vertex, List[int]] = defaultdict(list)
    for inst in instances:
        adj[inst.u].append(inst.instance_id)
        adj[inst.v].append(inst.instance_id)
    used = [False] * len(instances)
    current = start
    vertices = [start]
    route_edges: List[int] = []
    for _ in range(len(instances)):
        candidates = [i for i in adj[current] if not used[i]]
        if not candidates:
            raise RuntimeError("Fleury became stuck before using all instances.")
        chosen = candidates[0]
        if len(candidates) > 1:
            for candidate in candidates:
                next_vertex = _other(instances[candidate], current)
                if _reachable_without_instance(instances, used, current, next_vertex, candidate):
                    chosen = candidate
                    break
        used[chosen] = True
        current = _other(instances[chosen], current)
        route_edges.append(chosen)
        vertices.append(current)
    if current != start:
        raise RuntimeError("Fleury returned an open trail.")
    return vertices, route_edges


def solve_undirected_cpp(
    edges: Sequence[Edge], depot: Vertex, *, verify_small_matching: bool = True
) -> UCPPResult:
    _validate_edges(edges)
    vertices = {e.u for e in edges} | {e.v for e in edges}
    if depot not in vertices:
        raise ValueError("The depot is not a vertex of the graph.")
    if connected_vertices(edges, depot) != vertices:
        raise ValueError("UCPP requires a connected graph after isolated vertices are removed.")

    odd = odd_vertices(edges)
    matching: List[Tuple[Vertex, Vertex]] = []
    matching_cost = 0.0
    duplicated: List[int] = []
    if odd:
        distances, predecessors = metric_closure(edges, odd)
        matching, matching_cost = blossom_matching(odd, distances)
        if verify_small_matching and len(odd) <= 18:
            oracle = bitmask_matching_cost(odd, distances)
            if not math.isclose(matching_cost, oracle, rel_tol=1e-9, abs_tol=1e-9):
                raise AssertionError("Blossom and bitmask matching costs disagree.")
        for u, v in matching:
            duplicated.extend(restore_path_edges(predecessors[u], u, v))

    instances = _make_instances(edges, duplicated)
    route_vertices, route_instances = hierholzer_undirected(instances, depot)
    multiplicity = Counter(inst.base_id for inst in instances)
    weight_by_id = {edge.base_id: edge.weight for edge in edges}
    total_cost = sum(weight_by_id[base_id] * count for base_id, count in multiplicity.items())
    base_cost = sum(edge.weight for edge in edges)
    if not math.isclose(total_cost, base_cost + matching_cost, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError("CPP cost identity failed.")
    return UCPPResult(
        vertices=route_vertices,
        instance_ids=route_instances,
        instances=instances,
        matching=matching,
        matching_cost=matching_cost,
        base_cost=base_cost,
        total_cost=total_cost,
        multiplicity=dict(multiplicity),
    )


@dataclass
class Anchor:
    weights: Dict[str, float]
    optimal_cost: float
    multiplicity: Dict[str, int]
    odd_pair_count: int


def make_anchor(edges: Sequence[Edge], result: UCPPResult) -> Anchor:
    return Anchor(
        weights={edge.base_id: edge.weight for edge in edges},
        optimal_cost=result.total_cost,
        multiplicity=dict(result.multiplicity),
        odd_pair_count=len(odd_vertices(edges)) // 2,
    )


def regret_certificate(anchor: Anchor, current_weights: Mapping[str, float]) -> Dict[str, float | bool]:
    if set(current_weights) != set(anchor.weights):
        raise ValueError("A structural edge-set change invalidates the weight-only certificate.")
    if any((not math.isfinite(w) or w < 0) for w in current_weights.values()):
        raise ValueError("Current weights must be finite and nonnegative.")
    drift_l1 = sum(abs(current_weights[e] - anchor.weights[e]) for e in anchor.weights)
    base_lower = sum(current_weights.values())
    lipschitz_lower = anchor.optimal_cost - (anchor.odd_pair_count + 1) * drift_l1
    lower_bound = max(base_lower, lipschitz_lower)
    keep_cost = sum(anchor.multiplicity[e] * current_weights[e] for e in anchor.weights)
    if lower_bound <= 0:
        relative_bound = math.inf
    else:
        relative_bound = max(0.0, keep_cost / lower_bound - 1.0)
    return {
        "drift_l1": drift_l1,
        "base_lower": base_lower,
        "lipschitz_lower": lipschitz_lower,
        "lower_bound": lower_bound,
        "keep_cost": keep_cost,
        "relative_regret_bound": relative_bound,
    }


def example_edges() -> List[Edge]:
    return [
        Edge("AB", "A", "B", 4),
        Edge("BC", "B", "C", 2),
        Edge("CD", "C", "D", 3),
        Edge("AD", "A", "D", 8),
        Edge("BD", "B", "D", 10),
        Edge("CE", "C", "E", 4),
        Edge("EF", "E", "F", 2),
        Edge("FA", "F", "A", 3),
    ]


if __name__ == "__main__":
    result = solve_undirected_cpp(example_edges(), "A")
    print("matching:", result.matching)
    print("base cost:", result.base_cost)
    print("augmentation:", result.matching_cost)
    print("optimal cost:", result.total_cost)
    print("route:", " -> ".join(map(str, result.vertices)))
