from __future__ import annotations

from dataclasses import dataclass

from .engine import EngineConfig, NeuronGraphRAG
from .sample import load_sample_corpus


@dataclass(frozen=True, slots=True)
class EvalCase:
    query: str
    expected_node_id: str


EVAL_CASES = (
    EvalCase(
        "The payment retry policy changed from two attempts to five attempts.",
        "decision",
    ),
    EvalCase(
        "Decision D17 accepted five payment attempts to reduce transient failures.",
        "implementation",
    ),
    EvalCase(
        "Pull request 42 implemented decision D17 in the payment worker.",
        "incident",
    ),
)


def evaluate(*, limit: int = 5) -> dict[str, object]:
    baseline = _run_cases(
        EngineConfig(
            entry_weight=1.0,
            graph_weight=0.0,
            seed_count=1,
            max_hops=2,
        ),
        limit,
    )
    graph = _run_cases(
        EngineConfig(
            entry_weight=0.25,
            graph_weight=0.75,
            seed_count=1,
            max_hops=2,
        ),
        limit,
    )
    improved_queries = sum(
        graph_rank < baseline_rank
        for graph_rank, baseline_rank in zip(
            graph["ranks"], baseline["ranks"], strict=True
        )
    )
    return {
        "corpus_size": len(loadable_document_ids()),
        "cases": len(EVAL_CASES),
        "baseline_hybrid": baseline,
        "graph_rag": graph,
        "improved_queries": improved_queries,
    }


def loadable_document_ids() -> tuple[str, ...]:
    from .sample import SAMPLE_DOCUMENTS

    return tuple(document[0] for document in SAMPLE_DOCUMENTS)


def _run_cases(config: EngineConfig, limit: int) -> dict[str, object]:
    ranks: list[int] = []
    with NeuronGraphRAG(config=config) as engine:
        load_sample_corpus(engine)
        for case in EVAL_CASES:
            trace = engine.search(case.query, limit=limit, now=1_000.0)
            rank = next(
                (
                    index
                    for index, hit in enumerate(trace.hits, start=1)
                    if hit.node.node_id == case.expected_node_id
                ),
                limit + 1,
            )
            ranks.append(rank)
    reciprocal_rank = sum(1.0 / rank for rank in ranks) / len(ranks)
    hit_at_three = sum(rank <= 3 for rank in ranks) / len(ranks)
    return {
        "mean_reciprocal_rank": reciprocal_rank,
        "hit_at_3": hit_at_three,
        "ranks": ranks,
    }
