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
    max_active_paths_per_node: int


@dataclass(frozen=True, slots=True)
class CompetitionSetDiagnostic:
    step: int
    source_id: str
    path_identity: str
    neighbor_count: int
    mean_query_relevance: float
    message_total_before: float
    message_total_after: float

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "source_id": self.source_id,
            "path_identity": self.path_identity,
            "neighbor_count": self.neighbor_count,
            "mean_query_relevance": self.mean_query_relevance,
            "message_total_before": self.message_total_before,
            "message_total_after": self.message_total_after,
        }


@dataclass(frozen=True, slots=True)
class PropagationDiagnostics:
    strategy: str
    steps: int
    expansions: int
    activation_total: float
    converged: bool
    stop_reason: str
    active_path_count: int = 0
    competition_sets: tuple[CompetitionSetDiagnostic, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "steps": self.steps,
            "expansions": self.expansions,
            "activation_total": self.activation_total,
            "converged": self.converged,
            "stop_reason": self.stop_reason,
            "active_path_count": self.active_path_count,
            "competition_sets": [item.as_dict() for item in self.competition_sets],
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
    if settings.strategy in {
        "local_neighbor_competition",
        "local_neighbor_query_competition",
        "local_neighbor_path_competition",
        "local_neighbor_query_path_competition",
        "anchored_local_competition",
        "anchored_local_query_competition",
    }:
        result = _local_recurrent(
            query=query,
            seed_ids=seed_ids,
            entry=entry,
            nodes=nodes,
            outgoing_edges=outgoing_edges,
            settings=settings,
        )
        if settings.strategy.startswith("anchored_"):
            return _without_zero_hop_anchor(result, seed_ids, entry, settings)
        return result
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
            active_path_count=sum(len(node_paths) for node_paths in paths.values()),
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
            active_path_count=len(best_paths),
        ),
    )


@dataclass(frozen=True, slots=True)
class _PathState:
    node_id: str
    activation: float
    path: ActivationPath


def _local_recurrent(
    *,
    query: str,
    seed_ids: list[str],
    entry: dict[str, float],
    nodes: dict[str, DocumentNode],
    outgoing_edges: OutgoingEdges,
    settings: DynamicsSettings,
) -> PropagationResult:
    query_conditioned = settings.strategy in {
        "local_neighbor_query_competition",
        "local_neighbor_query_path_competition",
        "anchored_local_query_competition",
    }
    path_conditioned = settings.strategy in {
        "local_neighbor_path_competition",
        "local_neighbor_query_path_competition",
    }
    if path_conditioned:
        return _local_path_recurrent(
            query=query,
            seed_ids=seed_ids,
            entry=entry,
            nodes=nodes,
            outgoing_edges=outgoing_edges,
            settings=settings,
            query_conditioned=query_conditioned,
        )

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
    competition_sets: list[CompetitionSetDiagnostic] = []
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
            proposals: list[tuple[TypedEdge, float, float, ActivationPath]] = []
            for edge in outgoing_edges(source_id):
                if edge.target_id in visited:
                    continue
                relevance = (
                    _query_gate(query_terms, nodes[edge.target_id], edge, settings)
                    if query_conditioned
                    else 1.0
                )
                contribution = (
                    source_activation
                    * edge.weight
                    * edge.factuality
                    * settings.hop_decay
                    * relevance
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
                proposals.append(
                    (
                        edge,
                        contribution,
                        relevance,
                        ActivationPath(
                            source_path.seed_id,
                            contribution,
                            source_path.steps + (path_step,),
                        ),
                    )
                )
                expansions += 1
                if expansions >= settings.max_expansions:
                    stop_reason = "max_expansions"
                    break
            competed, diagnostic = _compete_neighbors(
                proposals,
                source_activation=source_activation,
                settings=settings,
                step_number=step_number + 1,
                source_id=source_id,
                path_identity=_path_identity(source_path),
            )
            if diagnostic is not None:
                competition_sets.append(diagnostic)
            for edge, contribution, _, path in competed:
                adjusted = ActivationPath(path.seed_id, contribution, path.steps)
                messages[edge.target_id] += contribution
                message_paths[edge.target_id].append(adjusted)
            if expansions >= settings.max_expansions:
                break

        candidate: dict[str, float] = defaultdict(float)
        for node_id, value in state.items():
            candidate[node_id] += settings.recurrent_decay * value
        for node_id, value in messages.items():
            candidate[node_id] += value
        candidate = {node_id: value for node_id, value in candidate.items() if value > 0.0}
        delta = _state_delta(candidate, state)
        state = candidate
        for node_id, node_paths in message_paths.items():
            paths[node_id].extend(node_paths)
            best_paths[node_id] = max(
                node_paths,
                key=lambda path: (path.contribution, _path_identity(path)),
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
            active_path_count=len(best_paths),
            competition_sets=tuple(competition_sets),
        ),
    )


def _without_zero_hop_anchor(
    result: PropagationResult,
    seed_ids: list[str],
    entry: dict[str, float],
    settings: DynamicsSettings,
) -> PropagationResult:
    """Remove the decayed zero-hop seed residual from an anchored graph signal."""
    residual_factor = settings.recurrent_decay ** result.diagnostics.steps
    activation = dict(result.activation)
    for seed_id in seed_ids:
        remaining = activation.get(seed_id, 0.0) - entry[seed_id] * residual_factor
        if remaining > 1e-15:
            activation[seed_id] = remaining
        else:
            activation.pop(seed_id, None)
    paths = {
        node_id: [path for path in node_paths if path.steps]
        for node_id, node_paths in result.paths.items()
    }
    paths = {node_id: node_paths for node_id, node_paths in paths.items() if node_paths}
    diagnostics = PropagationDiagnostics(
        strategy=result.diagnostics.strategy,
        steps=result.diagnostics.steps,
        expansions=result.diagnostics.expansions,
        activation_total=sum(activation.values()),
        converged=result.diagnostics.converged,
        stop_reason=result.diagnostics.stop_reason,
        active_path_count=sum(len(node_paths) for node_paths in paths.values()),
        competition_sets=result.diagnostics.competition_sets,
    )
    return PropagationResult(activation, paths, diagnostics)


def _local_path_recurrent(
    *,
    query: str,
    seed_ids: list[str],
    entry: dict[str, float],
    nodes: dict[str, DocumentNode],
    outgoing_edges: OutgoingEdges,
    settings: DynamicsSettings,
    query_conditioned: bool,
) -> PropagationResult:
    query_terms = set(tokenize(query))
    path_states = [
        _PathState(seed_id, entry[seed_id], ActivationPath(seed_id, entry[seed_id]))
        for seed_id in seed_ids
    ]
    paths: dict[str, list[ActivationPath]] = defaultdict(list)
    for item in path_states:
        paths[item.node_id].append(item.path)

    expansions = 0
    stop_reason = "recurrent_step_limit"
    converged = False
    completed_steps = 0
    competition_sets: list[CompetitionSetDiagnostic] = []
    for step_number in range(settings.recurrent_steps):
        candidate_states = [
            _PathState(item.node_id, settings.recurrent_decay * item.activation, item.path)
            for item in path_states
            if settings.recurrent_decay * item.activation > 0.0
        ]
        for item in sorted(path_states, key=_path_state_sort_key):
            visited = {item.path.seed_id}
            visited.update(path_step.target_id for path_step in item.path.steps)
            proposals: list[tuple[TypedEdge, float, float, ActivationPath]] = []
            for edge in outgoing_edges(item.node_id):
                if edge.target_id in visited:
                    continue
                relevance = (
                    _query_gate(query_terms, nodes[edge.target_id], edge, settings)
                    if query_conditioned
                    else 1.0
                )
                contribution = (
                    item.activation
                    * edge.weight
                    * edge.factuality
                    * settings.hop_decay
                    * relevance
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
                proposals.append(
                    (
                        edge,
                        contribution,
                        relevance,
                        ActivationPath(
                            item.path.seed_id,
                            contribution,
                            item.path.steps + (path_step,),
                        ),
                    )
                )
                expansions += 1
                if expansions >= settings.max_expansions:
                    stop_reason = "max_expansions"
                    break
            competed, diagnostic = _compete_neighbors(
                proposals,
                source_activation=item.activation,
                settings=settings,
                step_number=step_number + 1,
                source_id=item.node_id,
                path_identity=_path_identity(item.path),
            )
            if diagnostic is not None:
                competition_sets.append(diagnostic)
            for edge, contribution, _, path in competed:
                adjusted = ActivationPath(path.seed_id, contribution, path.steps)
                candidate_states.append(_PathState(edge.target_id, contribution, adjusted))
                paths[edge.target_id].append(adjusted)
            if expansions >= settings.max_expansions:
                break

        previous = _aggregate_path_state(path_states)
        path_states = _prune_path_states(candidate_states, settings.max_active_paths_per_node)
        current = _aggregate_path_state(path_states)
        delta = _state_delta(current, previous)
        completed_steps = step_number + 1
        if delta <= settings.convergence_tolerance:
            stop_reason = "converged"
            converged = True
            break
        if expansions >= settings.max_expansions:
            break

    state = _aggregate_path_state(path_states)
    return PropagationResult(
        state,
        dict(paths),
        PropagationDiagnostics(
            strategy=settings.strategy,
            steps=completed_steps,
            expansions=expansions,
            activation_total=sum(state.values()),
            converged=converged,
            stop_reason=stop_reason,
            active_path_count=len(path_states),
            competition_sets=tuple(competition_sets),
        ),
    )


def _compete_neighbors(
    proposals: list[tuple[TypedEdge, float, float, ActivationPath]],
    *,
    source_activation: float,
    settings: DynamicsSettings,
    step_number: int,
    source_id: str,
    path_identity: str,
) -> tuple[
    list[tuple[TypedEdge, float, float, ActivationPath]],
    CompetitionSetDiagnostic | None,
]:
    if not proposals:
        return [], None
    before = sum(item[1] for item in proposals)
    maximum = max(item[1] for item in proposals)
    threshold = settings.inhibition_ratio * maximum if len(proposals) > 1 else 0.0
    inhibited = [
        (edge, max(0.0, value - threshold), relevance, path)
        for edge, value, relevance, path in proposals
    ]
    inhibited = [item for item in inhibited if item[1] > 0.0]
    total = sum(item[1] for item in inhibited)
    local_budget = settings.activation_budget * source_activation
    scale = min(1.0, local_budget / total) if total else 1.0
    competed = [
        (edge, value * scale, relevance, path)
        for edge, value, relevance, path in inhibited
    ]
    after = sum(item[1] for item in competed)
    return competed, CompetitionSetDiagnostic(
        step=step_number,
        source_id=source_id,
        path_identity=path_identity,
        neighbor_count=len(proposals),
        mean_query_relevance=sum(item[2] for item in proposals) / len(proposals),
        message_total_before=before,
        message_total_after=after,
    )


def _prune_path_states(
    states: list[_PathState], max_per_node: int
) -> list[_PathState]:
    grouped: dict[str, list[_PathState]] = defaultdict(list)
    for item in states:
        grouped[item.node_id].append(item)
    output: list[_PathState] = []
    for node_id in sorted(grouped):
        output.extend(sorted(grouped[node_id], key=_path_state_sort_key)[:max_per_node])
    return output


def _aggregate_path_state(states: list[_PathState]) -> dict[str, float]:
    output: dict[str, float] = defaultdict(float)
    for item in states:
        output[item.node_id] += item.activation
    return dict(output)


def _path_state_sort_key(item: _PathState) -> tuple[float, str, str]:
    return (-item.activation, item.node_id, _path_identity(item.path))


def _path_identity(path: ActivationPath) -> str:
    targets = [step.target_id for step in path.steps]
    return ">".join((path.seed_id, *targets))


def _state_delta(left: dict[str, float], right: dict[str, float]) -> float:
    return max(
        (
            abs(left.get(node_id, 0.0) - right.get(node_id, 0.0))
            for node_id in set(left) | set(right)
        ),
        default=0.0,
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
