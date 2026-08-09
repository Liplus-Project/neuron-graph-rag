# Longitudinal multi-corpus feedback adaptation experiment

## Purpose

This frozen experiment tests whether trace-credited feedback remains directionally useful after multiple success events across independently constructed corpus clusters. It compares a no-mutation control with trace-credited treatment at fixed horizons `h=0,1,2,3`.

It does not reuse, rerun, modify, aggregate, or use observed results from the earlier feedback-adaptation experiments as selection inputs. Their fixture identifiers are read only by the contamination audit; their gold and observed result artifacts are not loaded.

## Frozen cohorts and provenance

- Manifest: `tests/fixtures/d1_liplus_longitudinal_feedback_experiment.manifest.json`
- Each development and holdout split contains three disjoint clusters: two headroom clusters and one ceiling cluster.
- Every cluster has a separately hashed fixture, gold schedule, and provenance record. The records identify these as deterministic in-repository constructed corpora, with no network or D1 access and zero writes.
- The audit compares all six clusters and both earlier feedback-adaptation fixture families by document path, node ID, source URL, normalized query, endpoint, and complete edge identity. It reads prior fixture identifiers only.

## Schedule and path identity

Every arm shares the same corpus, configuration, limit, timestamped feedback schedule, and scoring schedule. At horizons one through three, both arms record the same successful relation feedback event. The control records it without an edge mutation; treatment uses only the credited relation trace path.

Relation path checks retain raw observed steps for audit but compare only the projected identity:

```json
{"source_id":"...","target_id":"...","edge_type":"..."}
```

Runtime fields cannot affect identity. Each horizon also checks a direct lexical case and a directional-negative case.

## Ceiling-aware gate and one-shot execution

For headroom clusters, the control relation MRR at `h=0` must be below `1.0`; treatment must strictly improve final-horizon aggregate MRR and must not regress in any headroom cluster. For ceiling clusters, the `h=0` control relation MRR must equal `1.0`, and treatment requires non-regression only. A ceiling pass records that no additional rank improvement is evidenced; it never authorizes default changes, generalization, or production adoption.

After the result-free freeze is pushed, development is allowed once at its registered output path:

```powershell
python tools/run_longitudinal_feedback_experiment.py development `
  --manifest tests/fixtures/d1_liplus_longitudinal_feedback_experiment.manifest.json `
  --output tests/fixtures/d1_liplus_longitudinal_feedback_experiment.development.result.json
```

Only a passing development result allows one holdout execution at its registered output path. The runner rejects any other output path and refuses to overwrite an observed result. Once observation begins, fixture, gold, provenance, audit, manifest, implementation, tests, documentation, result aggregation, and reruns are forbidden.

## Interpretation boundary

Even a passing frozen result supports only longitudinal relation retrieval adaptation on these six small constructed clusters. It does not evaluate agent task success, tool calls, tokens, latency, autonomous-feedback correctness, a production default, or general corpora.
