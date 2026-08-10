# Sibling relation feedback normalization

## Purpose

This candidate makes a successful relation-trace edge compete only with its uncredited siblings from the same source. It is intended to make local feedback contrastive without introducing global or source-unrelated inhibition.

## Candidate behavior

`EngineConfig.sibling_feedback_normalization` is disabled by default (`0.0`). A positive value enables the candidate for a `search_channels` relation trace only.

- Each credited edge is reinforced through the existing learning-rate and maximum-weight rule.
- For each credited source, the candidate divides the actual credited weight increase among the source's uncredited sibling edges and reduces each share by the configured normalization ratio, never below zero.
- Credited siblings, lexical traces, zero-hop successes, and edges from other sources are unchanged by the normalization step.
- The feedback transaction remains atomic: a failed credited or sibling update records neither feedback nor partial edge updates.

## Validation boundary

The implementation is covered by synthetic fixtures that fix all of the following: the default `0.0` behavior, an opt-in target and uncredited sibling, multiple credited edges from one source, an unrelated source, lexical and zero-hop traces, and a direct source match that remains the top direct result. It does not modify D1 fixtures or make any result claim. Before changing the default, a result-free development / holdout plan must freeze relation improvement and direct, lexical, and directional-negative non-regression gates.
