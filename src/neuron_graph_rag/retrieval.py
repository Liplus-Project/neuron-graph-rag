from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence

from .models import DocumentNode

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
DenseEncoder = Callable[[str], Sequence[float]]


def tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text)]


class BM25Retriever:
    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(self, query: str, nodes: Sequence[DocumentNode]) -> dict[str, float]:
        query_terms = tokenize(query)
        documents = [tokenize(node.text) for node in nodes]
        if not query_terms or not documents:
            return {node.node_id: 0.0 for node in nodes}

        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(document))
        average_length = sum(map(len, documents)) / len(documents) or 1.0
        scores: dict[str, float] = {}
        document_count = len(documents)

        for node, document in zip(nodes, documents, strict=True):
            frequencies = Counter(document)
            document_length = len(document)
            score = 0.0
            for term in query_terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                frequency_in_documents = document_frequency[term]
                inverse_document_frequency = math.log(
                    1.0
                    + (
                        document_count
                        - frequency_in_documents
                        + 0.5
                    )
                    / (frequency_in_documents + 0.5)
                )
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * document_length / average_length
                )
                score += inverse_document_frequency * (
                    frequency * (self.k1 + 1.0) / denominator
                )
            scores[node.node_id] = score
        return scores


class FeatureHashingEncoder:
    """Dependency-free dense baseline.

    The encoder hashes word features and character n-grams into a fixed-width
    float vector. It is deterministic and replaceable; it is not a learned
    semantic embedding model.
    """

    def __init__(self, dimensions: int = 192) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions

    def __call__(self, text: str) -> list[float]:
        normalized = " ".join(tokenize(text))
        features = [f"w:{token}" for token in normalized.split()]
        compact = normalized.replace(" ", "_")
        for size in (3, 4):
            features.extend(
                f"c{size}:{compact[index:index + size]}"
                for index in range(max(0, len(compact) - size + 1))
            )
        vector = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.blake2b(
                feature.encode("utf-8"), digest_size=8, person=b"ngrag-v1"
            ).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        magnitude = math.sqrt(sum(component * component for component in vector))
        if magnitude:
            return [component / magnitude for component in vector]
        return vector


class DenseRetriever:
    def __init__(self, encoder: DenseEncoder | None = None) -> None:
        self.encoder = encoder or FeatureHashingEncoder()

    def score(self, query: str, nodes: Sequence[DocumentNode]) -> dict[str, float]:
        query_vector = tuple(float(value) for value in self.encoder(query))
        return {
            node.node_id: max(
                0.0,
                self._cosine(
                    query_vector,
                    tuple(float(value) for value in self.encoder(node.text)),
                ),
            )
            for node in nodes
        }

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Dense encoder returned inconsistent dimensions")
        left_magnitude = math.sqrt(sum(value * value for value in left))
        right_magnitude = math.sqrt(sum(value * value for value in right))
        if not left_magnitude or not right_magnitude:
            return 0.0
        return (
            sum(a * b for a, b in zip(left, right, strict=True))
            / left_magnitude
            / right_magnitude
        )


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    maximum = max(scores.values(), default=0.0)
    if maximum <= 0.0:
        return {key: 0.0 for key in scores}
    return {key: value / maximum for key, value in scores.items()}
