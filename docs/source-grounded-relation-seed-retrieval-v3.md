# Source-grounded relation-seed retrieval v3

## 目的

Issue #206 は、frozen test module 自身を repository lifecycle の hard gate にする。v2 は lifecycle API と個別 placeholder test では観測後を受理する一方、同じ test module 内の repository-root test が `audit_result_free` を無条件に呼ぶため、正当な observation artifact が登録された後の full suite では失敗する。この v2 limitation は frozen v2 artifact を変更せず、観測前の successor v3 で解消する。

本 protocol は result-free freeze 専用であり、development / holdout を実行しない。v1 の excluded evidence を再利用せず、未観測の v2 claim / raw packet / output も v3 へ移さない。

## v2 identity の byte 継承

v3 は別 protocol ID と別 claim / raw / output registry を持つ。performance が一度も観測されていない v2 の corpus、development / holdout query、gold、gate は tuning や変換を行わず、v2 の登録 path と SHA-256 を v3 manifest から直接参照する。v3 用の複製を生成しないため、継承 identity は v2 artifact と byte 同一である。

v3 loader は v1 / v2 manifest と frozen artifact hash を先に検証し、その評価 identity に v3 lifecycle registry を重ねた effective manifest を作る。freeze verifier は disk 上の v3 manifest を freeze commit blob と照合し、effective manifest の v3 artifact / registry に加えて、継承した v2 manifest、corpus、query、gold、gate の commit blob hash も検証する。

## Phase-aware frozen module

repository-root test は常に `audit_repository_lifecycle` を実行し、current phase が登録 lifecycle のいずれかであることを検証する。`audit_result_free` は current phase が result-free の場合だけ追加確認する。したがって観測後 repository root へ result-free 専用 assertion を適用しない。

frozen v3 test module 全体は subprocess で次の isolated temporary repository state に対して実行する。

- claim / raw / output が存在しない result-free state
- canonical development claim、4 raw packet、recomputed output を持つ synthetic post-observation state

post-observation probe は candidate 不成立の `development-closed` と candidate 成立の `holdout-eligible` を別 temporary root で覆う。module の一部 test method だけを呼ばず、毎回 `tests.test_source_grounded_relation_observation_v3` 全体を実行する。

## Fail-closed lifecycle

claim、raw packet、output は canonical JSON として manifest 登録 path へ exclusive create する。protocol commit、stage、arm、run、attempt 1、retry 0 を一致させ、raw packet は固定順序の append-only prefix だけを受理する。gap、tamper、overwrite、output と disk packet の不一致を拒否する。

development output が全 hard gate 通過かつ candidate selected の場合だけ holdout eligible とする。development 未完了または eligible でない状態の holdout artifact は拒否する。worker は gold を受け取らず、各 worker は fresh SQLite を使い、finalizer だけが disk 上の4 packetを再読込して gold を開く。shared database は SHA-256 前後比較だけに使い、SQLite として開かない。

## Result-free hard gate

次の検証は temporary directory に placeholder / synthetic artifact だけを生成する。登録 development / holdout query を実行せず、shared database を開かず、repository の evidence pathへ artifact を残さない。

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_source_grounded_relation_observation_v3 -v
python tools/probe_source_grounded_relation_observation_v3.py --root .
python tools/acquire_source_grounded_relation_corpus_v3.py --output tests/fixtures/github_source_grounded_relation_v2.corpus.json --verify
python -c "from neuron_graph_rag.source_grounded_relation_observation_v3 import audit_result_free; print(audit_result_free())"
```

freeze commit の Core CI / Optional MCP、Ruff、corpus verify、canonical UTF-8 / artifact hash が green になるまで観測を開始しない。merge 後も本 Issue では development / holdout を実行せず、別 successor observation issue が v3 development を exactly once 開始する。
