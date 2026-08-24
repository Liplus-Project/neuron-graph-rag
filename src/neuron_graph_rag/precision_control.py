from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from .models import SearchHit

RULES = (
    "absolute_final_score_floor",
    "relative_top_score_ratio",
    "relative_top_score_margin",
    "entry_graph_signal_agreement",
)


@dataclass(frozen=True, slots=True)
class PrecisionControl:
    """Explicit post-ranking precision filter; disabled unless configured."""

    candidate_id: str
    minimum_final_score: float | None = None
    minimum_top_score_ratio: float | None = None
    maximum_top_score_margin: float | None = None
    require_entry_graph_signal_agreement: bool = False

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        thresholds = (
            self.minimum_final_score,
            self.minimum_top_score_ratio,
            self.maximum_top_score_margin,
        )
        if any(value is not None and not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("precision thresholds must be between 0 and 1")
        if all(value is None for value in thresholds) and not (
            self.require_entry_graph_signal_agreement
        ):
            raise ValueError("at least one precision rule must be enabled")

    @classmethod
    def from_mapping(cls, value: object) -> PrecisionControl:
        if not isinstance(value, dict):
            raise ValueError("precision control candidate must be an object")
        allowed = {
            "candidate_id",
            "minimum_final_score",
            "minimum_top_score_ratio",
            "maximum_top_score_margin",
            "require_entry_graph_signal_agreement",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown precision control fields: {sorted(unknown)}")
        return cls(**value)

    def rules(self) -> tuple[str, ...]:
        enabled = []
        if self.minimum_final_score is not None:
            enabled.append(RULES[0])
        if self.minimum_top_score_ratio is not None:
            enabled.append(RULES[1])
        if self.maximum_top_score_margin is not None:
            enabled.append(RULES[2])
        if self.require_entry_graph_signal_agreement:
            enabled.append(RULES[3])
        return tuple(enabled)

    def thresholds(self) -> dict[str, float | bool]:
        values: dict[str, float | bool] = {}
        if self.minimum_final_score is not None:
            values["minimum_final_score"] = self.minimum_final_score
        if self.minimum_top_score_ratio is not None:
            values["minimum_top_score_ratio"] = self.minimum_top_score_ratio
        if self.maximum_top_score_margin is not None:
            values["maximum_top_score_margin"] = self.maximum_top_score_margin
        if self.require_entry_graph_signal_agreement:
            values["require_entry_graph_signal_agreement"] = True
        return values


def apply_precision_control(
    ranked_hits: Sequence[SearchHit], control: PrecisionControl
) -> tuple[tuple[SearchHit, ...], tuple[SearchHit, ...]]:
    """Annotate all ranked hits and return the accepted subsequence."""

    if not ranked_hits:
        return (), ()
    top_score = ranked_hits[0].final_score
    annotated: list[SearchHit] = []
    accepted: list[SearchHit] = []
    for pre_filter_rank, hit in enumerate(ranked_hits, start=1):
        ratio = hit.final_score / top_score if top_score > 0.0 else 0.0
        margin = top_score - hit.final_score
        has_entry_signal = hit.entry_score > 0.0
        has_graph_signal = hit.normalized_graph_activation > 0.0 and any(
            path.steps for path in hit.paths
        )
        rule_results: dict[str, bool] = {}
        if control.minimum_final_score is not None:
            rule_results[RULES[0]] = hit.final_score >= control.minimum_final_score
        if control.minimum_top_score_ratio is not None:
            rule_results[RULES[1]] = ratio >= control.minimum_top_score_ratio
        if control.maximum_top_score_margin is not None:
            rule_results[RULES[2]] = margin <= control.maximum_top_score_margin
        if control.require_entry_graph_signal_agreement:
            rule_results[RULES[3]] = has_entry_signal and has_graph_signal
        decision = {
            "candidate_id": control.candidate_id,
            "accepted": all(rule_results.values()),
            "applied_rules": list(control.rules()),
            "thresholds": control.thresholds(),
            "pre_filter_rank": pre_filter_rank,
            "pre_filter_score": hit.final_score,
            "top_score": top_score,
            "top_score_ratio": ratio,
            "top_score_margin": margin,
            "entry_signal_present": has_entry_signal,
            "graph_signal_present": has_graph_signal,
            "rule_results": rule_results,
            "source_provenance": {
                key: hit.node.metadata[key]
                for key in (
                    "repository",
                    "commit",
                    "path",
                    "source_url",
                    "content_sha256",
                )
                if key in hit.node.metadata
            },
        }
        annotated_hit = replace(hit, precision_control=decision)
        annotated.append(annotated_hit)
        if decision["accepted"]:
            accepted.append(annotated_hit)
    return tuple(annotated), tuple(accepted)
