# Real-task feedback shadow protocol v3

## 1. 目的と境界

`real-task-feedback-shadow-v3` は、protocol freeze と将来の observation registry lifecycle を分離する result-free capture / replay 契約である。v2 の capture provenance、snapshot exact verification、two-arm config、append-only correction、exclusive registry、slot 順 cumulative replay、determinism、idempotencyを維持し、freeze test が登録先を永久に空と要求する lifecycle contradiction だけを解消する。

v1 / v2 の source、schema、fixture、manifest、test、docs は変更しない。v3 protocol commit は placeholder identity だけを含み、実 task packet、実 source identity、observed outcome、aggregate result を含めない。Issue #95 / PR #96 を含む過去の query、trace、snapshot、source-use、outcome、packet は v3 へ移送・再利用しない。

## 2. Freeze と lifecycle の分離

result-free manifest は protocol artifact と legacy artifact の hash、登録先、固定 lifecycle rule を保存する。freeze 時点の事実は `freeze_observation_status=not_started_at_protocol_commit` として固定し、現在の registry が空であることは hash 対象にも永続 assertion にもしない。

同じ frozen manifest / test は次の状態を順に許可する。

1. packet registry と final aggregate が存在しない。
2. slot 1 から連番の root packet と、immutable field を保つ append-only correction が追加される。
3. effective registry 全体を参照する final aggregate が一回だけ exclusive create される。

packet の追加後に manifest、schema、gate、test の書換えを要求しない。final aggregate 後の packet 追加、別 aggregate、field reorder、non-canonical JSON は repository lifecycle audit で fail closed になる。

## 3. Capture と registry contract

packet schema は `tests/fixtures/real_task_shadow_v3.packet-schema.json`、result schema は `tests/fixtures/real_task_shadow_v3.result-schema.json` を正本とする。v3 packet / result は別 protocol ID と schema version を持つが、semantic validation と exact replay は frozen v2 の実装へ protocol identity を変換して実行する。これにより次を維持する。

- query、limit、candidate identity / content hash、credited path、検索時刻、search surface、effective config、各 fingerprint を結ぶ capture provenance
- packet と SQLite snapshot byte hash、candidate 順、source URL、content hash、used node、credited path の exact verification
- `used_q3_s1` / `confirmed_r05_s1` の feedback-only override と共通 retrieval config
- slot 順 cumulative replay、fresh clone per arm、同一 idempotency key の semantic receipt、repeat replay の determinism

registry writer は exclusive lock 内で全 packet を scan / canonical validate し、新規 root を slot 1 から連番で追加する。correction は既存 packet を一度だけ supersede でき、slot、task、snapshot、retrieval、source-use、capture を変更できない。packet ID と出力 file は exclusive create であり、上書きしない。

## 4. 二つの verifier

local exact verifier は明示された SQLite snapshot を必要とする。live WAL / journal sidecar を拒否し、snapshot の fresh clone から packet と aggregate を再計算して exact 比較する。CI から取得できない live snapshot を推測・再構築しない。

repository lifecycle audit は repository 内 artifact だけを使い、snapshot replay を担当しない。次を検証する。

- v3 manifest path の初回追加 commit にある protocol / legacy frozen artifact の exact blob hash
- 初回追加 commit の存在と current `HEAD` の ancestor 関係、および current manifest bytes と初回登録 manifest bytes の一致
- packet と aggregate の canonical JSON bytes
- root slot の連番、correction chain の到達可能性、immutable correction field
- effective packet 全体で共通の snapshot hash、capture config、search surface と、slot 順に厳密増加する capture timestamp
- final aggregate が一つだけで、effective packet ID / slot / snapshot hash / capture config / search surface と一致すること

この責務分離により、CI は repository lifecycle を再現可能に検証し、snapshot を持つ観測環境だけが retrieval semantics を exact 検証する。

## 5. Result-free gate

`tests/test_real_task_shadow_v3.py` は placeholder identity のみで空 registry、sequential root、superseding correction、one-time final aggregate を同じ manifest 上に再現する。さらに temporary lifecycle probe とは別に、manifest の実 registered path を repository root 上で毎回 audit する。protocol hash、canonical JSON、slot、immutable field、batch 共通入力、capture 時刻順、exclusive write、one-time result の tamper を fail closed にする。実 task URL、実 source identity、Issue #95 packet / outcome、PR #96 artifact は fixture と registered output に含めない。

## 6. Commands

```text
python tools/run_real_task_shadow_v3.py probe --fixture tests/fixtures/real_task_shadow_v3.placeholder.json
python tools/run_real_task_shadow_v3.py capture --input PACKET.json --registry-dir artifacts/real-task-feedback-shadow-v3/packets
python tools/run_real_task_shadow_v3.py verify-packet --packet PACKET.json --snapshot SNAPSHOT.db
python tools/run_real_task_shadow_v3.py replay --registry-dir artifacts/real-task-feedback-shadow-v3/packets --snapshot SNAPSHOT.db --output artifacts/real-task-feedback-shadow-v3/observed/final.json
python tools/run_real_task_shadow_v3.py verify-result --result artifacts/real-task-feedback-shadow-v3/observed/final.json --registry-dir artifacts/real-task-feedback-shadow-v3/packets --snapshot SNAPSHOT.db
python tools/run_real_task_shadow_v3.py audit-lifecycle --manifest tests/fixtures/real_task_shadow_v3.manifest.json --repository-root .
```

`probe` は実 MCP adapter search から placeholder packet、snapshot exact verification、two-arm replay、one-time result までを一時 directory 内で一巡し、registered output を repository に生成しない。`audit-lifecycle` は snapshot を使わず、freeze 後の repository lifecycle state を検証する。

同名 path の current working tree は repository evolution の現在状態であり、v1 / v2 / v3 の historical evidence ではない。lifecycle audit は manifest、packet、aggregate、gate、観測結果を変更せず、hash 取得元だけを manifest path の初回追加 commit の blob に固定する。
