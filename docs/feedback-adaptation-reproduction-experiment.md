# Trace-credited feedback adaptation reproduction experiment

## Purpose

This experiment independently reproduces the local trace-credit claim on a new frozen split. It does not reuse, rerun, modify, aggregate, or use the observed result from the prior feedback-adaptation experiment as a selection input.

The development split contains three D1 wiki documents from `Liplus-Project/liplus-language`; the holdout contains three D1 wiki documents from `Liplus-Project/dipper_ai`. Both acquisitions use only read-only D1 queries, and their provenance records report zero writes.

## Frozen inputs and separation

- Manifest: `tests/fixtures/d1_liplus_feedback_adaptation_reproduction_experiment.manifest.json`
- Development and holdout fixture, gold, and provenance files use the `d1_liplus_feedback_adaptation_reproduction_*` prefix.
- The contamination audit compares document paths, node IDs, source URLs, normalized queries, endpoints, and complete `{source_id, target_id, edge_type}` edge identities. It reads only identifiers from the prior fixtures; it never loads prior gold or result artifacts.
- Development and holdout are mutually disjoint and disjoint from the prior feedback-adaptation fixtures.

## Path identity

Observed relation-trace steps may contain runtime fields such as `edge_weight`, `factuality`, activation, or trace identifiers. Before comparison, each raw step is projected to exactly:

```json
{"source_id":"...","target_id":"...","edge_type":"..."}
```

Gold paths use the same endpoint/type-only shape. The synthetic unit test proves that changing runtime fields cannot change a projected path identity.

## Registered runs and gates

For each permitted stage, the fixed schedule contains one no-mutation control, one treatment, and one treatment replay. The control records the same feedback event but does not mutate an edge; treatment can reinforce only the credited relation path.

The gates require strict relation-MRR improvement, no relation Recall or Hit@k regression, projected-path identity, non-regressing direct lexical and directional-negative controls, credited-only mutation, deterministic replay, hash verification, contamination separation, exclusive result outputs, and stage ordering.

After this result-free freeze is pushed, development may run exactly once:

```powershell
python tools/run_feedback_adaptation_reproduction.py development `
  --manifest tests/fixtures/d1_liplus_feedback_adaptation_reproduction_experiment.manifest.json `
  --output tests/fixtures/d1_liplus_feedback_adaptation_reproduction_experiment.development.result.json
```

Only a development pass permits one holdout run with that development result. A failed development run leaves holdout unopened. After an observed result exists, the frozen protocol, code, documentation, tests, fixtures, and output must not be changed or rerun.

## Interpretation boundary

A passing result supports only this minimal frozen corpus and the trace-credit mechanism. It does not validate a learned router, alter defaults, establish production quality, or support general-corpus claims.
