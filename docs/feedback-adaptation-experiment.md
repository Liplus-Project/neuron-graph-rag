# Trace-credited feedback adaptation experiment

## Purpose

This experiment tests a causal, local claim: a recorded success on a relation trace can improve a later relation retrieval under the same frozen corpus and schedule. It does not validate a learned router, change the default API, or estimate production quality.

## Frozen protocol

The manifest fixes a development split and an unseen holdout split. Each has one source node, a credited target edge, and a competing edge. A feedback query retrieves the source and its relation trace; the later scoring queries are distinct from that feedback query.

- Control records the same relation feedback event and selected node, but applies no edge update.
- Treatment records the same event and permits only `record_success` credited-path reinforcement.
- Both groups share the corpus, config, limit, timestamps, feedback schedule, and scoring schedule.
- The relation case measures target MRR, Recall, and Hit@k. Direct lexical and directional-negative cases are controls.
- The expected target endpoint and `mention` edge type must remain visible in the treatment path.

## Freeze and stop rule

`d1_liplus_feedback_adaptation_experiment.manifest.json` hashes the fixtures, gold schedules, provenance, and contamination audit before any observed result exists. Its registered runs are one control and two treatment executions per permitted split; the second treatment execution is the deterministic replay.

The runner rejects an existing output path. Development is the only stage initially permitted. Holdout opens exactly once only when every development gate passes; a development failure leaves holdout absent. After observation, the frozen protocol, implementation, tests, documentation, and result artifacts are not changed or rerun.

## Interpretation boundary

A passing result supports only this fixed minimal corpus and trace-credit mechanism. It does not establish general-corpus adaptation, autonomous feedback correctness, a learned policy, or a production default change.
