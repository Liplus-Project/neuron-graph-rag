# Real-task feedback shadow protocol v2

## 1. 目的と境界

`real-task-feedback-shadow-v2` は、live MCP search trace を frozen SQLite snapshot 上で exact replay できる result-free capture / replay 契約である。v1 は packet に effective runtime config と検索時刻を保存せず、固定 shadow arm config と synthetic time を retrieval verification に使うため、serving retrieval と feedback policy の差を分離できない。v2 は v1 artifact を書き換えず、この欠落だけを superseding protocol として修正する。

この protocol commit は placeholder identity だけを含み、実 task packet、実 source identity、observed outcome、aggregate result を含めない。serving retrieval/default、local registration、remote deployment、Issue #85 の frozen/observed artifact、Issue #89 の事前登録 gate を変更しない。Issue #90 は slot を作成していないため再利用しない。

## 2. Capture contract

MCP `search` success output は trace の `created_at` と `effective_config_provenance` を返す。provenance は `EngineConfig` の全 field を次の二つへ分離し、省略せず、実際に trace を生成した `search_surface`（`combined` / `relation`）を併記する。

- `retrieval`: `sparse_weight`、`dense_weight`、`entry_weight`、`graph_weight`、`seed_count`、`max_hops`、`hop_decay`、`activation_half_life_seconds`、`maximum_activation`、path / propagation bound、activation strategy とその係数、dense / graph switch、normalization、fusion、`rrf_k`
- `feedback`: `feedback_learning_rate`、`sibling_feedback_normalization`、`maximum_edge_weight`、`relation_feedback_evidence_quorum`、`confirmed_outcome_reinforcement`、`confirmation_decay_ratio`

各 config 区分と全体は canonical JSON bytes（UTF-8、key sort、2-space indent、末尾 LF）の SHA-256 を持つ。packet の `capture` は `searched_at`、`search_surface`、分離済み effective config、三 fingerprint を保存する。さらに query、limit、candidate identity / content hash、credited path、検索時刻、search surface、config と fingerprint をまとめた `capture_fingerprint` を保存する。通常の field 書換え、欠落、未知 field、非 canonical config は fail closed とする。

packet schema は `tests/fixtures/real_task_shadow_v2.packet-schema.json`、result schema は `tests/fixtures/real_task_shadow_v2.result-schema.json` を正本とする。v1 の task、snapshot、candidate、source-use、objective evidence、efficiency、append-only correction semantics は維持する。

## 3. Snapshot verification

packet verifier は次を順に満たす。

1. canonical packet validation と全 config field の `EngineConfig` round-trip を行う。
2. fingerprint と capture binding を再計算する。
3. live WAL / journal sidecar を拒否し、snapshot byte hash を照合する。
4. snapshot の fresh clone を開き、packet の capture effective config、`search_surface`、`searched_at` で同じ query / limit を同じ検索surfaceへ渡す。
5. candidate 順、source URL、content hash、used node、credited path を exact 比較する。

config、timestamp、candidate、source identity、content、path、fingerprint のいずれかが不整合なら packet は登録・replay できない。

## 4. Two-arm shadow replay

ordered batch ごとに arm あたり一つの fresh snapshot clone を作り、slot 1 から累積 replay する。全 packet は同じ snapshot hash と同じ capture effective config を共有し、capture timestamp は slot 順に厳密増加する。

両 arm は capture config 全体を基底にし、次の feedback-only field だけを上書きする。

| arm | feedback override |
| --- | --- |
| `used_q3_s1` | `relation_feedback_evidence_quorum=3`、`confirmed_outcome_reinforcement=false`、`confirmation_decay_ratio=null`、`sibling_feedback_normalization=1.0` |
| `confirmed_r05_s1` | `relation_feedback_evidence_quorum=1`、`confirmed_outcome_reinforcement=true`、`confirmation_decay_ratio=0.5`、`sibling_feedback_normalization=1.0` |

`feedback_learning_rate` と `maximum_edge_weight` は capture effective value を共通利用する。retrieval field と capture `search_surface` は一つも上書きしない。result は各 arm の full effective config、search surface、retrieval / feedback / full fingerprint、override field list を保存し、二 arm の retrieval config / search surface と capture provenance が byte-semantic に一致しなければ拒否する。

各 packet は capture timestamp で pre-feedback relation trace を再計算し、candidate 順と credited path を packet と照合してから source-use / outcome を適用する。同じ idempotency key の再送が同じ semantic receipt を返すこと、同じ batch の再実行が同じ semantic result を返すこと、source snapshot hash が不変であることを必須とする。stored result verifier は packet / registry と snapshot から batch 全体を再計算し、result と exact 比較する。

## 5. Registry と one-time observation boundary

registry writer は exclusive lock の内側で scan、slot / correction validation、exclusive create を行う。新規 root packet は slot 1 から連番、correction は immutable task / snapshot / retrieval / source-use / capture を保持する。異なる packet ID が同一 slot を競合した場合、一つだけが作成される。

この result-free commit の placeholder probe は registry へ永続登録しない。実 task stream の capture と一回だけの final aggregate は別の事前登録 observation issue だけが開始できる。development / holdout、#85 result、#90 task の遡及利用、観測後の再実行・再集計は行わない。

## 6. Commands

```text
python tools/run_real_task_shadow_v2.py probe --fixture tests/fixtures/real_task_shadow_v2.placeholder.json
python tools/run_real_task_shadow_v2.py capture --input PACKET.json --registry-dir LOCAL_REGISTRY
python tools/run_real_task_shadow_v2.py verify-packet --packet PACKET.json --snapshot SNAPSHOT.db
python tools/run_real_task_shadow_v2.py replay --registry-dir LOCAL_REGISTRY --snapshot SNAPSHOT.db --output RESULT.json
python tools/run_real_task_shadow_v2.py verify-result --result RESULT.json --registry-dir LOCAL_REGISTRY --snapshot SNAPSHOT.db
```

`probe` は modern timestamp、serving-default retrieval mix `0.55 / 0.45`、relation path、placeholder-only source を使い、実 MCP adapter search から packet capture、snapshot verification、二 arm replay、exact result verification までを一巡する。registered output は生成しない。
