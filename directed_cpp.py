"""Exact directed Chinese Postman Problem with float-safe min-cost transshipment."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import heapq
import math
from typing import Dict, Hashable, List, Mapping, Sequence, Tuple

Vertex = Hashable


@dataclass(frozen=True)
class Arc:
    base_id: str
    tail: Vertex
    head: Vertex
    weight: float


@dataclass(frozen=True)
class ArcInstance:
    instance_id: int
    base_id: str
    tail: Vertex
    head: Vertex
    duplicated: bool


@dataclass
class DCPPResult:
    vertices: List[Vertex]
    instance_ids: List[int]
    instances: List[ArcInstance]
    transportation: Dict[Tuple[Vertex, Vertex], int]
    base_cost: float
    augmentation_cost: float
    total_cost: float
    imbalance: Dict[Vertex, int]


def _validate(arcs: Sequence[Arc]) -> None:
    if not arcs:
        raise ValueError("The digraph must contain at least one arc.")
    ids = set()
    for arc in arcs:
        if arc.base_id in ids:
            raise ValueError(f"Duplicate base_id: {arc.base_id}")
        ids.add(arc.base_id)
        if arc.tail == arc.head:
            raise ValueError("Loops are excluded in this implementation.")
        if not math.isfinite(arc.weight) or arc.weight < 0:
            raise ValueError("Dijkstra-based DCPP requires finite nonnegative weights.")


def adjacency(arcs: Sequence[Arc]) -> Dict[Vertex, List[int]]:
    adj: Dict[Vertex, List[int]] = defaultdict(list)
    for i, arc in enumerate(arcs):
        adj[arc.tail].append(i)
        adj.setdefault(arc.head, [])
    return dict(adj)


def reverse_adjacency(arcs: Sequence[Arc]) -> Dict[Vertex, List[Vertex]]:
    radj: Dict[Vertex, List[Vertex]] = defaultdict(list)
    for arc in arcs:
        radj[arc.head].append(arc.tail)
        radj.setdefault(arc.tail, [])
    return dict(radj)


def _reachable(adj_vertices: Mapping[Vertex, Sequence[Vertex]], source: Vertex) -> set[Vertex]:
    seen = {source}
    queue = deque([source])
    while queue:
        v = queue.popleft()
        for u in adj_vertices.get(v, []):
            if u not in seen:
                seen.add(u)
                queue.append(u)
    return seen


def strongly_connected(arcs: Sequence[Arc]) -> bool:
    vertices = {a.tail for a in arcs} | {a.head for a in arcs}
    root = next(iter(vertices))
    forward: Dict[Vertex, List[Vertex]] = defaultdict(list)
    backward: Dict[Vertex, List[Vertex]] = defaultdict(list)
    for arc in arcs:
        forward[arc.tail].append(arc.head)
        backward[arc.head].append(arc.tail)
        forward.setdefault(arc.head, [])
        backward.setdefault(arc.tail, [])
    return _reachable(forward, root) == vertices and _reachable(backward, root) == vertices


def imbalance(arcs: Sequence[Arc]) -> Dict[Vertex, int]:
    indeg: Counter[Vertex] = Counter()
    outdeg: Counter[Vertex] = Counter()
    vertices = set()
    for arc in arcs:
        vertices.update((arc.tail, arc.head))
        outdeg[arc.tail] += 1
        indeg[arc.head] += 1
    return {v: indeg[v] - outdeg[v] for v in vertices}


def dijkstra_directed(
    arcs: Sequence[Arc], source: Vertex
) -> Tuple[Dict[Vertex, float], Dict[Vertex, Tuple[Vertex, int]]]:
    adj = adjacency(arcs)
    if source not in adj:
        raise ValueError("Unknown source vertex.")
    dist = {v: math.inf for v in adj}
    pred: Dict[Vertex, Tuple[Vertex, int]] = {}
    dist[source] = 0.0
    pq: List[Tuple[float, int, Vertex]] = [(0.0, 0, source)]
    serial = 1
    while pq:
        dv, _, v = heapq.heappop(pq)
        if dv != dist[v]:
            continue
        for arc_id in adj[v]:
            arc = arcs[arc_id]
            candidate = dv + arc.weight
            if candidate + 1e-12 < dist[arc.head]:
                dist[arc.head] = candidate
                pred[arc.head] = (v, arc_id)
                heapq.heappush(pq, (candidate, serial, arc.head))
                serial += 1
    return dist, pred


def restore_directed_path(
    predecessor: Mapping[Vertex, Tuple[Vertex, int]], source: Vertex, target: Vertex
) -> List[int]:
    path: List[int] = []
    current = target
    while current != source:
        if current not in predecessor:
            raise ValueError(f"No directed path from {source} to {target}.")
        previous, arc_id = predecessor[current]
        path.append(arc_id)
        current = previous
    path.reverse()
    return path


@dataclass
class _ResidualEdge:
    to: int
    rev: int
    cap: int
    cost: float
    original_cap: int
    pair: Tuple[Vertex, Vertex] | None = None


def _add_residual_edge(
    graph: List[List[_ResidualEdge]], u: int, v: int, cap: int, cost: float,
    pair: Tuple[Vertex, Vertex] | None = None,
) -> None:
    forward = _ResidualEdge(v, len(graph[v]), cap, cost, cap, pair)
    backward = _ResidualEdge(u, len(graph[u]), 0, -cost, 0, None)
    graph[u].append(forward)
    graph[v].append(backward)


def min_cost_transport(
    positive: Sequence[Vertex], negative: Sequence[Vertex], b: Mapping[Vertex, int],
    distances: Mapping[Tuple[Vertex, Vertex], float],
) -> Tuple[Dict[Tuple[Vertex, Vertex], int], float]:
    """Successive shortest augmenting paths on a bipartite transport network."""
    total = sum(b[p] for p in positive)
    if total != sum(-b[n] for n in negative):
        raise ValueError("Supply and demand do not balance.")
    nodes: List[Vertex | str] = ["__source__", *positive, *negative, "__sink__"]
    idx = {node: i for i, node in enumerate(nodes)}
    source, sink = 0, len(nodes) - 1
    residual: List[List[_ResidualEdge]] = [[] for _ in nodes]
    for p in positive:
        _add_residual_edge(residual, source, idx[p], b[p], 0.0)
    for p in positive:
        for n in negative:
            _add_residual_edge(residual, idx[p], idx[n], total, distances[(p, n)], (p, n))
    for n in negative:
        _add_residual_edge(residual, idx[n], sink, -b[n], 0.0)

    potential = [0.0] * len(nodes)
    sent = 0
    cost = 0.0
    while sent < total:
        dist = [math.inf] * len(nodes)
        parent: List[Tuple[int, int] | None] = [None] * len(nodes)
        dist[source] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, source)]
        while pq:
            dv, v = heapq.heappop(pq)
            if dv != dist[v]:
                continue
            for edge_index, edge in enumerate(residual[v]):
                if edge.cap <= 0:
                    continue
                reduced = edge.cost + potential[v] - potential[edge.to]
                candidate = dv + reduced
                if candidate + 1e-12 < dist[edge.to]:
                    dist[edge.to] = candidate
                    parent[edge.to] = (v, edge_index)
                    heapq.heappush(pq, (candidate, edge.to))
        if not math.isfinite(dist[sink]):
            raise RuntimeError("The transportation problem is infeasible.")
        for v in range(len(nodes)):
            if math.isfinite(dist[v]):
                potential[v] += dist[v]
        augmentation = total - sent
        v = sink
        while v != source:
            parent_info = parent[v]
            if parent_info is None:
                raise RuntimeError("Broken residual predecessor chain.")
            u, edge_index = parent_info
            augmentation = min(augmentation, residual[u][edge_index].cap)
            v = u
        v = sink
        path_cost = 0.0
        while v != source:
            u, edge_index = parent[v]  # type: ignore[misc]
            edge = residual[u][edge_index]
            path_cost += edge.cost
            edge.cap -= augmentation
            residual[v][edge.rev].cap += augmentation
            v = u
        sent += augmentation
        cost += augmentation * path_cost

    flow: Dict[Tuple[Vertex, Vertex], int] = {}
    for p in positive:
        for edge in residual[idx[p]]:
            if edge.pair is not None:
                used = edge.original_cap - edge.cap
                if used:
                    flow[edge.pair] = used
    return flow, cost


def hierholzer_directed(
    instances: Sequence[ArcInstance], depot: Vertex
) -> Tuple[List[Vertex], List[int]]:
    adj: Dict[Vertex, List[int]] = defaultdict(list)
    for inst in instances:
        adj[inst.tail].append(inst.instance_id)
        adj.setdefault(inst.head, [])
    cursor: Dict[Vertex, int] = defaultdict(int)
    used = [False] * len(instances)
    stack_v = [depot]
    stack_e: List[int] = []
    reverse_v: List[Vertex] = []
    reverse_e: List[int] = []
    while stack_v:
        v = stack_v[-1]
        incident = adj[v]
        while cursor[v] < len(incident) and used[incident[cursor[v]]]:
            cursor[v] += 1
        if cursor[v] == len(incident):
            reverse_v.append(stack_v.pop())
            if stack_e:
                reverse_e.append(stack_e.pop())
        else:
            instance_id = incident[cursor[v]]
            cursor[v] += 1
            if used[instance_id]:
                continue
            used[instance_id] = True
            inst = instances[instance_id]
            stack_v.append(inst.head)
            stack_e.append(instance_id)
    vertices = list(reversed(reverse_v))
    edge_ids = list(reversed(reverse_e))
    if len(edge_ids) != len(instances) or vertices[0] != depot or vertices[-1] != depot:
        raise RuntimeError("Directed Hierholzer failed to produce a closed Euler circuit.")
    return vertices, edge_ids


def solve_directed_cpp(arcs: Sequence[Arc], depot: Vertex) -> DCPPResult:
    _validate(arcs)
    vertices = {a.tail for a in arcs} | {a.head for a in arcs}
    if depot not in vertices:
        raise ValueError("Unknown depot.")
    if not strongly_connected(arcs):
        raise ValueError("The directed CPP requires strong connectivity on the arc support.")
    b = imbalance(arcs)
    positive = sorted((v for v in vertices if b[v] > 0), key=str)
    negative = sorted((v for v in vertices if b[v] < 0), key=str)
    predecessors: Dict[Vertex, Dict[Vertex, Tuple[Vertex, int]]] = {}
    distances: Dict[Tuple[Vertex, Vertex], float] = {}
    for p in positive:
        dist, pred = dijkstra_directed(arcs, p)
        predecessors[p] = pred
        for n in negative:
            distances[(p, n)] = dist[n]
    transport, augmentation_cost = min_cost_transport(positive, negative, b, distances)

    duplicated: List[int] = []
    for (p, n), amount in transport.items():
        path = restore_directed_path(predecessors[p], p, n)
        for _ in range(amount):
            duplicated.extend(path)

    instances: List[ArcInstance] = []
    for arc in arcs:
        instances.append(ArcInstance(len(instances), arc.base_id, arc.tail, arc.head, False))
    for arc_id in duplicated:
        arc = arcs[arc_id]
        instances.append(ArcInstance(len(instances), arc.base_id, arc.tail, arc.head, True))
    route_vertices, route_instances = hierholzer_directed(instances, depot)
    base_cost = sum(a.weight for a in arcs)
    total_cost = base_cost + augmentation_cost
    physical_cost = {a.base_id: a.weight for a in arcs}
    check = sum(physical_cost[i.base_id] for i in instances)
    if not math.isclose(check, total_cost, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError("DCPP cost identity failed.")
    indeg = Counter(i.head for i in instances)
    outdeg = Counter(i.tail for i in instances)
    if any(indeg[v] != outdeg[v] for v in vertices):
        raise AssertionError("Eulerization did not balance every directed vertex.")
    return DCPPResult(
        vertices=route_vertices,
        instance_ids=route_instances,
        instances=instances,
        transportation=transport,
        base_cost=base_cost,
        augmentation_cost=augmentation_cost,
        total_cost=total_cost,
        imbalance=b,
    )


def example_arcs() -> List[Arc]:
    return [
        Arc("AB", "A", "B", 2),
        Arc("BC", "B", "C", 1),
        Arc("CA", "C", "A", 2),
        Arc("AD", "A", "D", 1),
        Arc("DC", "D", "C", 1),
    ]


if __name__ == "__main__":
    result = solve_directed_cpp(example_arcs(), "A")
    print("imbalance:", result.imbalance)
    print("transport:", result.transportation)
    print("optimal cost:", result.total_cost)
    print("route:", " -> ".join(map(str, result.vertices)))
