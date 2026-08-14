# Real-task feedback shadow protocol

## 目的と境界

この protocol は、将来の Codex 実タスクで取得する同一の frozen packet と SQLite snapshot を、`used_q3_s1` と `confirmed_r05_s1` の二つの非 serving shadow arm へ再生するための result-free 契約である。この変更では protocol、schema、capture/replay mechanics、登録外 placeholder round-trip だけを固定し、実タスク、実 source identity、観測 outcome、observed result は保存しない。

既定の `NeuronGraphRAG`、local MCP registration、remote deployment、validated 状態は変更しない。既存の frozen / observed evaluation artifact は再実行、再集計、変更しない。本 protocol の結果だけで candidate を既定採用してはならない。

## Sequential inclusion

対象は、この protocol の merge 後に初めて現れた eligible task から順番に選ぶ。結果を見て task を選択せず、slot は `1, 2, ...` の連番とする。eligible task は task URL、repository、base commit、close condition、eligible 時刻、使用した SQLite snapshot の raw SHA-256 を持つ。同じ packet を二つの arm へ再生する。

source が `selected`、`validated`、`used` の三段階を同じ node で完了しなかった task は confirmed の対象にしない。source が unsupported または inadequate で停止した場合、その task を再利用して別結果を作らない。観測 batch、同じ edge の最低反復数、stop rule は後続の observation issue で result を見る前に固定し、本 issue では決めない。

## Packet と evidence

packet schema の正本は `tests/fixtures/real_task_shadow_v1.packet-schema.json`、runtime の fail-closed validation は `neuron_graph_rag.real_task_shadow.validate_packet()` である。packet は次を含む。

- packet ID、slot、supersedes packet ID
- task URL、repository、base commit、close condition、eligible 時刻
- database snapshot SHA-256
- query、limit、順序付き候補 node、source URL、content SHA-256、used node、credited relation path
- `selected` → `validated` → `used` の source-use event
- outcome status、summary、external reference、objective evidence
- nullable な tool call 数、research 数、elapsed seconds、token 数
- capture 時刻

confirmed outcome は used node に結び付く positive objective evidence を最低一つ必要とする。許可する evidence kind は `test_passed`、`citation_verified`、`review_accepted`、`rollback_or_correction` の四つだけである。前の三つを positive、最後を negative とする。self-evaluation は gold にしない。`pending` は evidence を持たず、`corrected` と `rolled_back` は `rollback_or_correction` を必要とする。

packet registry は append-only で canonical UTF-8 JSON を exclusive create する。同じ packet ID、slot skip、上書きを拒否する。訂正は既存 packet を編集せず `supersedes_packet_id` で直前の packet を指す新規 packet とし、slot、task、snapshot、retrieval、source-use は変えない。一つの packet から複数の直接 successor を作らない。

## Replay

CLI は次の command を持つ。

```text
python tools/run_real_task_shadow.py probe --fixture tests/fixtures/real_task_shadow_v1.placeholder.json
python tools/run_real_task_shadow.py capture --input PACKET.json --registry-dir LOCAL_REGISTRY
python tools/run_real_task_shadow.py verify-packet --packet PACKET.json --snapshot SNAPSHOT.db
python tools/run_real_task_shadow.py replay --packet PACKET.json --snapshot SNAPSHOT.db --output RESULT.json
python tools/run_real_task_shadow.py verify-result --result RESULT.json
```

`capture` と `replay` の出力は exclusive create であり、既存 file を置換しない。registry と observed result の登録先は manifest に記載するが、この result-free commit には directory も file も作らない。

replay は raw snapshot hash、captured node の存在、source URL、node text の UTF-8 SHA-256、fresh relation retrieval の候補順、used node の credited path を検証する。各 arm は source snapshot から独立した temporary clone を作り、同じ replay を二回行って semantic equality を確認する。source snapshot 自体は変更しない。

`used_q3_s1` は quorum `3`、sibling normalization `1.0`、used-time policy を使う。`confirmed_r05_s1` は quorum `1`、confirmed outcome reinforcement、decay `0.5`、sibling normalization `1.0` を使う。両 arm は Issue #85 で固定した明示 configuration であり、serving default ではない。

result schema は、使用 node の前後 rank / graph score、その delta、path edge の weight / reinforced / evidence / confirmation count と delta、non-target churn、source-use と outcome の semantic receipt、idempotency replay、二回の deterministic replay、source snapshot 不変を保存する。efficiency field は未計測なら `null` のまま保持する。

hash mismatch、node absence、source identity mismatch、content hash mismatch、candidate order mismatch、credited path mismatch、unsupported evidence、objective evidence 不足、duplicate packet、idempotency conflict、非 deterministic replay は fail closed する。negative outcome は監査記録だけを作り、negative reinforcement や rollback を実行しない。

## Placeholder freeze

`tests/fixtures/real_task_shadow_v1.placeholder.json` は `example.invalid` と `placeholder-*` だけを使う登録外 fixture である。test と `probe` は temporary SQLite を構築し、実 writer → canonical reader → packet verifier → 二 arm replay → exclusive result writer → result verifier の round-trip を実行する。temporary file は終了時に破棄され、repository に packet や result を残さない。

manifest は protocol/schema/CLI/test/docs の byte hash と、空であるべき local registry / observed result path を固定する。placeholder probe が通っても、実 task の outcome や arm の優位性を観測したことにはならない。
