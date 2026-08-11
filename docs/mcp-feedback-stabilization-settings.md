# MCP Feedback Stabilization Settings

## Purpose

The local MCP stdio server can opt into the controlled feedback stabilization candidate without changing library or command defaults. This is a process-local configuration surface, not a project-wide adoption decision.

## Normative requirements

1. The CLI exposes `--relation-feedback-evidence-quorum`, accepts positive integers, and defaults to `1`.
2. The CLI exposes `--sibling-feedback-normalization`, accepts finite values from `0.0` through `1.0`, and defaults to `0.0`.
3. The CLI validates both settings and constructs `neuron_graph_rag.evidence_feedback.EngineConfig` before opening the configured SQLite database.
4. Invalid feedback settings exit with a CLI usage error before creating or mutating the database.
5. At normalization `0.0`, the MCP `search` tool retains the existing hybrid `search()` behavior.
6. At positive normalization, the MCP `search` tool returns `search_channels(...).relation` so later source-use feedback carries relation provenance and can normalize same-source siblings.
7. Quorum `3` with normalization `1.0` leaves serving weights unchanged after the first and second independent evidence items. The third item applies one bounded credited update and the corresponding same-source sibling normalization.
8. Idempotency replay and duplicate source-use stages do not increase evidence or repeat mutations.
9. Library defaults, MCP CLI defaults, legacy engine/storage contracts, and the frozen Issue #76 / #77 protocol and observed artifacts remain unchanged.

## Adoption boundary

The `3` / `1.0` combination passed the registered controlled development and holdout evaluation. This supports a reversible local opt-in only. It does not establish external-corpus generalization, production quality, remote deployment readiness, or project-wide default adoption.

## Verification

The MCP protocol tests cover default q1/s0 behavior, q3/s1 activation across three independent stdio traces, retry idempotency, credited and sibling weight locality, and invalid-before-database-mutation behavior. Frozen evaluation verification reads the registered bytes from the Git commit that introduced each manifest, keeping historical integrity separate from later README and test-infrastructure evolution.

## Related

- [Optional MCP Feedback Interface](optional-mcp-interface.md)
- [Evidence-gated local feedback reinforcement](evidence-gated-local-feedback-reinforcement.md)
- [Canonical evidence gate evaluation](canonical-evidence-gate-evaluation.md)
