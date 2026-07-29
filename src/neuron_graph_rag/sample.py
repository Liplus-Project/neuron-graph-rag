from __future__ import annotations

from .engine import NeuronGraphRAG

SAMPLE_DOCUMENTS = (
    (
        "policy",
        "The payment retry policy changed from two attempts to five attempts.",
        {"kind": "policy"},
        0.96,
    ),
    (
        "decision",
        "Decision D17 accepted five payment attempts to reduce transient failures.",
        {"kind": "decision"},
        0.93,
    ),
    (
        "implementation",
        "Pull request 42 implemented decision D17 in the payment worker.",
        {"kind": "pull_request"},
        0.99,
    ),
    (
        "incident",
        "Incident I8 reported worker queue saturation after deployment.",
        {"kind": "incident"},
        0.91,
    ),
    (
        "dashboard",
        "The operations dashboard tracks payment latency and queue depth.",
        {"kind": "documentation"},
        0.89,
    ),
)

SAMPLE_EDGES = (
    ("policy", "decision", "justified_by", 0.70, 0.95),
    ("decision", "implementation", "implemented_by", 0.65, 1.00),
    ("implementation", "incident", "followed_by", 0.55, 0.85),
    ("incident", "dashboard", "observed_in", 0.45, 0.90),
)


def load_sample_corpus(engine: NeuronGraphRAG) -> None:
    for node_id, text, metadata, confidence in SAMPLE_DOCUMENTS:
        engine.add_document(
            node_id,
            text,
            metadata=metadata,
            confidence=confidence,
        )
    for source_id, target_id, edge_type, weight, factuality in SAMPLE_EDGES:
        try:
            engine.store.edge(source_id, target_id, edge_type)
        except KeyError:
            engine.add_edge(
                source_id,
                target_id,
                edge_type,
                weight=weight,
                factuality=factuality,
            )
