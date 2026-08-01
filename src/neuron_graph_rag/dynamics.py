from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .models import ActivationPath, DocumentNode, PathStep, TypedEdge
from .retrieval import tokenize


@dataclass(frozen=True, slots=True)
class DynamicsSettings:
    strategy: str
    max_hops: int
    hop_decay: float
    max_expansions: int
    activation_budget: float
    inhibition_ratio: float
    inhibition_top_k: int
    query_transmission_floor: float
    query_transmission_power: float
    recurrent_steps: int
    recurrent_decay: float
    convergence_tolerance: float


@dataclass(frozen=True, slots=True)
class PropagationDiagnostics:
    strategy: str
    steps: int
    expansions: int
    activation_total: float
    converged: bool
    stop_reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "steps": self.steps,
            "expansions": self.expansions,
            "activation_total": self.activation_total,
            "converged": self.converged,
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True, slots=True)
class PropagationResult:
    activation: dict[str, float]
    paths: dict[str, list[ActivationPath]]
    diagnostics: PropagationDiagnostics


OutgoingEdges = Callable[[str], Sequence[TypedEdge]]


def propagate(
    *,
    query: str,
    seed_ids: list[str],
    entry: dict[str, float],
    nodes: dict[str, DocumentNode],
    outgoing_edges: OutgoingEdges,
    settings: DynamicsSettings,
) -> PropagationResult:
    if settings.strategy == "recurrent_competition":
        return _recurrent(
            query=query,
            seed_ids=seed_ids,
            entry=entry,
            nodes=nodes,
            outgoing_edges=outgoing_edges,
            settings=settings,
        )
    if settings.strategy not in {
        "current_positive_additive",
        "finite_activation_budget",
        "lateral_inhibition",
        "query_conditioned_transmission",
    }:
        raise ValueError(f"Unknown activation strategy: {settings.strategy}")

    activation: dict[str, float] = defaultdict(float)
    paths: dict[str, list[ActivationPath]] = defaultdict(list)
    frontier: list[tuple[str, str, float, tuple[PathStep, ...], frozenset[str]]] = []
    for seed_id in seed_ids:
        contribution = entry[seed_id]
        activation[seed_id] += contribution
        paths[seed_id].append(ActivationPath(seed_id, contribution))
        frontier.append((seed_id, seed_id, contribution, (), frozenset({seed_id})))

    expansions = 0
    steps = 0
    stop_reason = "max_hops"
    query_terms = set(tokenize(query))
    for depth in range(settings.max_hops):
        proposals: list[
            tuple[str, str, float, tuple[PathStep, ...], frozenset[str]]
        ] = []
        for seed_id, current_id, contribution, path_steps, visited in frontier:
            for edge in outgoing_edges(current_id):
                if edge.target_id in visited:
                    continue
                next_contribution = (
                    contribution
                    * edge.weight
                    * edge.factuality
                    * settings.hop_decay
                )
                if settings.strategy == "query_conditioned_transmission":
                    next_contribution *= _query_gate(
                        query_terms,
                        nodes[edge.target_id],
                        edge,
                        settings,
                    )
                if next_contribution <= 0.0:
                    continue
                step = PathStep(
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type,
                    edge.weight,
                    edge.factuality,
                )
                proposals.append(
                    (
                        seed_id,
                        edge.target_id,
                        next_contribution,
                        path_steps + (step,),
                        visited | {edge.target_id},
                    )
                )
                expansions += 1
                if expansions >= settings.max_expansions:
                    stop_reason = "max_expansions"
                    break
            if expansions >= settings.max_expansions:
                break
        if not proposals:
            stop_reason = "frontier_exhausted"
            break
        if settings.strategy == "finite_activation_budget":
            total = sum(item[2] for item in proposals)
            scale = min(1.0, settings.activation_budget / total) if total else 1.0
            proposals = [
                (seed_id, node_id, contribution * scale, path_steps, visited)
                for seed_id, node_id, contribution, path_steps, visited in proposals
            ]
        for seed_id, node_id, contribution, path_steps, _ in proposals:
            activation[node_id] += contribution
            paths[node_id].append(
                ActivationPath(seed_id, contribution, path_steps)
            )
        frontier = proposals
        steps = depth + 1
        if expansions >= settings.max_expansions:
            break

    if settings.strategy == "lateral_inhibition":
        activation = defaultdict(float, _inhibit(dict(activation), settings))
        paths = defaultdict(
            list,
            {
                node_id: node_paths
                for node_id, node_paths in paths.items()
                if activation.get(node_id, 0.0) > 0.0
            },
        )

    output = dict(activation)
    return PropagationResult(
        output,
        dict(paths),
        PropagationDiagnostics(
            strategy=settings.strategy,
            steps=steps,
            expansions=expansions,
            activation_total=sum(output.values()),
            converged=stop_reason == "frontier_exhausted",
            stop_reason=stop_reason,
        ),
    )


def _recurrent(
    *,
    query: str,
    seed_ids: list[str],
    entry: dict[str, float],
    nodes: dict[str, DocumentNode],
    outgoing_edges: OutgoingEdges,
    settings: DynamicsSettings,
) -> PropagationResult:
    query_terms = set(tokenize(query))
    state = {node_id: entry[node_id] for node_id in seed_ids}
    paths: dict[str, list[ActivationPath]] = defaultdict(list)
    best_paths: dict[str, ActivationPath] = {}
    for seed_id in seed_ids:
        path = ActivationPath(seed_id, entry[seed_id])
        paths[seed_id].append(path)
        best_paths[seed_id] = path

    expansions = 0
    stop_reason = "recurrent_step_limit"
    converged = False
    completed_steps = 0
    for step_number in range(settings.recurrent_steps):
        messages: dict[str, float] = defaultdict(float)
        message_paths: dict[str, list[ActivationPath]] = defaultdict(list)
        for source_id, source_activation in sorted(state.items()):
            if source_activation <= 0.0:
                continue
            source_path = best_paths.get(
                source_id, ActivationPath(source_id, source_activation)
            )
            visited = {source_path.seed_id}
            visited.update(path_step.target_id for path_step in source_path.steps)
            for edge in outgoing_edges(source_id):
                if edge.target_id in visited:
                    continue
                contribution = (
                    source_activation
                    * edge.weight
                    * edge.factuality
                    * settings.hop_decay
                    * _query_gate(
                        query_terms,
                        nodes[edge.target_id],
                        edge,
                        settings,
                    )
                )
                if contribution <= 0.0:
                    continue
                path_step = PathStep(
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type,
                    edge.weight,
                    edge.factuality,
                )
                path = ActivationPath(
                    source_path.seed_id,
                    contribution,
                    source_path.steps + (path_step,),
                )
                messages[edge.target_id] += contribution
                message_paths[edge.target_id].append(path)
                expansions += 1
                if expansions >= settings.max_expansions:
                    stop_reason = "max_expansions"
                    break
            if expansions >= settings.max_expansions:
                break

        total_messages = sum(messages.values())
        if total_messages > settings.activation_budget:
            scale = settings.activation_budget / total_messages
            messages = {node_id: value * scale for node_id, value in messages.items()}
            message_paths = {
                node_id: [
                    ActivationPath(path.seed_id, path.contribution * scale, path.steps)
                    for path in node_paths
                ]
                for node_id, node_paths in message_paths.items()
            }

        candidate: dict[str, float] = defaultdict(float)
        for node_id, value in state.items():
            candidate[node_id] += settings.recurrent_decay * value
        for node_id, value in messages.items():
            candidate[node_id] += value
        candidate = _inhibit(dict(candidate), settings)
        delta = max(
            (
                abs(candidate.get(node_id, 0.0) - state.get(node_id, 0.0))
                for node_id in set(candidate) | set(state)
            ),
            default=0.0,
        )
        state = candidate
        for node_id, node_paths in message_paths.items():
            paths[node_id].extend(node_paths)
            best_paths[node_id] = max(
                node_paths,
                key=lambda path: (path.contribution, path.seed_id),
            )
        completed_steps = step_number + 1
        if delta <= settings.convergence_tolerance:
            stop_reason = "converged"
            converged = True
            break
        if expansions >= settings.max_expansions:
            break

    return PropagationResult(
        dict(state),
        dict(paths),
        PropagationDiagnostics(
            strategy=settings.strategy,
            steps=completed_steps,
            expansions=expansions,
            activation_total=sum(state.values()),
            converged=converged,
            stop_reason=stop_reason,
        ),
    )


def _query_gate(
    query_terms: set[str],
    target: DocumentNode,
    edge: TypedEdge,
    settings: DynamicsSettings,
) -> float:
    evidence_terms = set(tokenize(target.text)) | set(tokenize(edge.edge_type))
    if not query_terms or not evidence_terms:
        overlap = 0.0
    else:
        overlap = len(query_terms & evidence_terms) / math.sqrt(
            len(query_terms) * len(evidence_terms)
        )
    conditioned = math.pow(max(0.0, min(1.0, overlap)), settings.query_transmission_power)
    return settings.query_transmission_floor + (
        1.0 - settings.query_transmission_floor
    ) * conditioned


def _inhibit(
    activation: dict[str, float], settings: DynamicsSettings
) -> dict[str, float]:
    if not activation:
        return {}
    maximum = max(activation.values())
    threshold = settings.inhibition_ratio * maximum
    inhibited = {
        node_id: max(0.0, value - threshold)
        for node_id, value in activation.items()
    }
    inhibited = {node_id: value for node_id, value in inhibited.items() if value > 0.0}
    if settings.inhibition_top_k > 0:
        keep = {
            node_id
            for node_id, _ in sorted(
                inhibited.items(), key=lambda item: (-item[1], item[0])
            )[: settings.inhibition_top_k]
        }
        inhibited = {
            node_id: value for node_id, value in inhibited.items() if node_id in keep
        }
    return inhibited
