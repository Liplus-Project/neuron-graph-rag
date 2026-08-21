# Confirmed-outcome feedback reinforcement

## 目的と既定境界

この文書は、永続 relation reinforcement を source の `used` ではなく後続の `confirmed` outcome へ接続する default-off candidate の正本である。candidate は `neuron_graph_rag.evidence_feedback.EngineConfig` で次の二値を同時に明示した時だけ有効になる。

```python
EngineConfig(
    confirmed_outcome_reinforcement=True,
    confirmation_decay_ratio=0.5,
)
```

`confirmation_decay_ratio` は `0 < r < 1` を満たす有限値でなければならない。flag なしの ratio、ratio なしの flag は拒否する。package root と `neuron_graph_rag.engine` の class identity、`SQLiteStore.apply_success_feedback` の既存契約、MCP と library の既定値、ローカル q3/s1 deployment は変更しない。

## Trigger と credit assignment

candidate 有効時、`retrieved`、`selected`、`validated`、`used` は retrieval と利用状態だけを保存する。`success_feedback`、relation evidence、edge weight、`reinforced_count` は変更しない。

正の update は次をすべて満たす `confirmed` だけが発火できる。

- node が candidate 有効時に同じ trace で `used` となり、その policy marker が保存されている。
- trace が `search_channels(...).relation` で保存した relation trace である。
- node の保存済み path に一つ以上の relation step がある。
- 同じ credited edge と trace の confirmation がまだ保存されていない。

credited path は retrieval 時に保存した path のうち contribution 最大、同値なら `seed_id` の決定論的順序で一つ選ぶ。回答全体の成功、lexical trace、hybrid zero-hop、trace 外 node から edge を推測しない。`corrected`、`rolled_back`、`superseded` は outcome 履歴へ保存するだけで、負の update や既存 weight rollback を行わない。

candidate 有効化前に `used` となって即時 reinforcement 済みの trace は policy marker を持たない。その trace を後から `confirmed` に再送しても audit outcome だけを保存し、同じ利用を二重強化しない。

## Diminishing schedule

credited edge の最初の独立 confirmation を count `1`、multiplier `1.0` とする。初回の bounded increment は既存 feedback planning と同じ `feedback_learning_rate * clamped_contribution` で固定し、同じ edge の後続 confirmation `n` はこの初回 increment に `r^(n-1)` を掛ける。したがって cap 到達前の delta は厳密に減少し、trace ごとの contribution 差で後続 delta が増えることはない。

edge schedule は初回 weight、初回 increment、ratio、confirmation count を SQLite に保存する。累積 weight は既存 `maximum_edge_weight` と `initial_weight + first_increment / (1-r)` の小さい側で飽和する。同じ edge を別 ratio で継続しようとした場合は `confirmation_policy_conflict` として拒否し、途中から schedule を再解釈しない。

maximum 到達後も新しい独立 confirmation は count と `reinforced_count` を一回進められるが、actual delta は `0` となる。この event は sibling weight を変更しない。

## Same-source locality と transaction

`sibling_feedback_normalization > 0` の場合、今回の confirmation で実際に増加した credited delta だけを同じ source の uncredited outgoing sibling へ均等配分する。別 source、credited sibling、lexical、zero-hop は変更しない。

outcome row、outcome node、edge schedule、confirmation row、credited edge update、sibling update、idempotency receipt は一つの nested SQLite transaction に含める。edge、sibling、receipt 保存のいずれかが失敗した場合、count、weight、`reinforced_count`、outcome を call 前へ rollback する。`CREATE TABLE IF NOT EXISTS` migration は既存 database の node、edge、retrieval、feedback、outcome row を書き換えず、restart 後も count と idempotency receipt を維持する。

## Receipt

core の `OutcomeReceipt` と MCP `record_outcome` output は次を同じ順序で返す。

- `confirmations`: edge identity、`confirmation_count`、`multiplier`、`actual_delta`、old/new weight。
- `credited_paths`: used node と保存済み relation steps。
- `normalized_sibling_edges`: actual delta から局所減算した sibling。
- `reinforcement_applied`: 今回新しい独立 edge confirmation を一件以上保存したか。

idempotency replay は保存済み receipt をそのまま返す。同じ trace が別 key で再送された場合は outcome audit row を保存できるが、`confirmations` は空、`reinforcement_applied=false` となり、count と weight を変更しない。

## Optional MCP config

local stdio server では flag と ratio を組にして明示する。

```bash
neuron-graph-rag-mcp \
  --database /absolute/path/to/knowledge.db \
  --confirmed-outcome-reinforcement \
  --confirmation-decay-ratio 0.5
```

candidate 有効時、MCP `search` は relation trace を返し、`tools/list` の source-use / outcome description も confirmed-triggered policy を明示する。無効時は既存 hybrid search、used-evidence reinforcement、audit-only delayed outcome description を保つ。

## Soft-start successor candidate

soft-start は confirmed-only と排他的な別の default-off candidate である。次の三値を同時に明示した場合だけ有効になる。

```python
EngineConfig(
    soft_start_feedback_reinforcement=True,
    soft_start_feedback_ratio=0.25,
    confirmation_decay_ratio=0.5,
)
```

`soft_start_feedback_ratio` と `confirmation_decay_ratio` はどちらも有限の `0 < value < 1` とする。soft-start は `confirmed_outcome_reinforcement=True` または `relation_feedback_evidence_quorum != 1` と組み合わせられず、矛盾する設定は database を開く前に拒否する。

credited relation edge で最初に発生した新規 `used` は、通常の bounded increment の `soft_start_feedback_ratio` 倍だけを provisional に適用する。edge ごとの schedule は initial weight、base increment、soft-start ratio、confirmation decay、geometric maximum を SQLite に保存する。同じ edge の後続 `used` は独立 trace の policy marker と audit row を保存するが、provisional update を重複適用しない。`used` 時点では sibling normalization を一切行わない。

最初の独立 `confirmed` は count `1`、表示 multiplier `1 - soft_start_feedback_ratio` とし、edge を `min(maximum_edge_weight, geometric maximum, initial weight + base increment)` まで増やす。この target との差だけを actual delta とするため、provisional と最初の confirmation の合計は通常 bounded update 一回を超えず、途中で cap または別の増加があっても weight を減らさない。後続 confirmation `n` は multiplier `confirmation_decay_ratio^(n-1)` を使う。same-source sibling normalization は各 confirmation の actual delta だけに適用する。

source-use state、policy marker、provisional edge update、soft-start audit、idempotency receipt は一つの transaction で保存する。confirmed outcome 側も outcome、count、edge/sibling update、receipt を一つの transaction で保存する。receipt 保存を含む途中失敗は全変更を rollback する。core `SourceUseReceipt.feedback` と MCP output は provisional edge を同じ形で返し、`OutcomeReceipt` と MCP output は remainder または後続 decay の count、multiplier、actual delta を同じ形で返す。

local stdio server では次のように起動する。

```bash
neuron-graph-rag-mcp \
  --database /absolute/path/to/knowledge.db \
  --soft-start-feedback-reinforcement \
  --soft-start-feedback-ratio 0.25 \
  --confirmation-decay-ratio 0.5
```

inactive な soft-start field は effective-config provenance へ追加しない。これにより既存 default、q3/s1、confirmed-only capture の canonical bytes と fingerprint を維持する。active な soft-start process だけが二つの soft-start field を provenance に含める。

## Adoption boundary

この実装は mechanics、atomicity、receipt parity、default compatibility を固定する。decay ratio の採用値、q3/s1 に対する優位性、production default 変更は主張しない。比較には、既存 #76 / #77 artifact を変更、再実行、再集計しない fresh result-free evaluation を別 Issue で固定する必要がある。

soft-start も同じ adoption boundary に従う。既存 #89 / PR #99 とその凍結 artifact は変更せず、soft-start、q3/s1、confirmed-only の比較は fresh successor evaluation として別に固定する。mechanics の実装だけで現在の local serving database または project default を切り替えない。

## Related

- [Evidence-gated local feedback reinforcement](evidence-gated-local-feedback-reinforcement.md)
- [Optional MCP feedback interface](optional-mcp-interface.md)
- [Decision Structure](Decision-Structure.md)
- [Issue #81](https://github.com/Liplus-Project/neuron-graph-rag/issues/81)
- [Issue #100](https://github.com/Liplus-Project/neuron-graph-rag/issues/100)
- [Soft-start feedback reinforcement decision](https://github.com/Liplus-Project/neuron-graph-rag/wiki/soft-start-feedback-reinforcement)
