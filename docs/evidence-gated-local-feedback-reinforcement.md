# Evidence-gated local feedback reinforcement

## 目的と既定値

この文書は、relation path の success feedback を独立 evidence と serving edge update に分離する opt-in candidate の正本である。`EngineConfig.relation_feedback_evidence_quorum` は正の整数で、既定値は `1` とする。既定値では最初の独立 success trace が従来どおり一回の bounded reinforcement を発火し、learning rate、maximum edge weight、fusion、normalization の既定値を変更しない。`2` 以上を明示した場合だけ quorum 到達前の serving weight update を遅延する。

## Evidence identity と activation

evidence identity は `(source_id, target_id, edge_type, trace_id)` とする。同じ edge を credit する異なる trace だけを独立 evidence として数える。同一 trace の再送、source-use stage の duplicate、MCP idempotency replay は evidence count を増やさない。

各 credited edge は event 後の evidence count、設定 quorum、今回の activation 有無を feedback receipt の `evidence` に保存する。

- count が quorum 未満なら `activated=false` とし、weight と `reinforced_count` を変更しない。
- count が初めて quorum に達した event を一回目の通常 reinforcement とする。quorum 前の未適用分を一括加算しない。
- quorum 到達後は、新しい独立 trace 一件につき一回の通常 reinforcement を適用する。
- 同じ feedback が複数 edge を credit する場合、evidence と activation は edge identity ごとに判定する。
- lexical trace、zero-hop success、credited path を持たない node は evidence を作らず、edge を変更しない。

`rank_elasticity` や既存の result-free evaluation はこの candidate の採用値を決めない。quorum の default 変更は新しい固定 protocol と別判断を必要とする。

## Local sibling normalization

`sibling_feedback_normalization` は、今回実際に増加した credited weight の合計だけを入力とする。quorum 前、duplicate trace、maximum weight 到達など actual delta が `0` の event では sibling を変更しない。activation event では従来どおり、同じ source の uncredited sibling にだけ actual increase と normalization ratio に基づく局所減算を配分する。credited sibling、別 source、lexical、zero-hop は変更しない。

## Storage と transaction

SQLite は edge identity と trace ID の複合一意制約を持つ evidence table を `CREATE TABLE IF NOT EXISTS` で追加する。既存 node、edge、retrieval、feedback、source-use、outcome row は書き換えず、既存 database を開いた時に非破壊で migration する。evidence count は process restart 後も同じ database から継続する。

feedback row、success node、evidence insert、credited edge update、sibling update、source-use state、idempotency receipt は既存の nested transaction 境界に含める。credited edge、multi-edge path、sibling、ledger 保存のいずれかが失敗した場合は、evidence count、weight、`reinforced_count`、feedback row をすべて call 前へ rollback する。

## Receipt schema

core の `FeedbackReceipt.evidence` と optional MCP の `feedback.evidence` は同じ順序の edge state を返す。

```json
{
  "source_id": "decision-17",
  "target_id": "pr-42",
  "edge_type": "implemented_by",
  "count": 3,
  "quorum": 3,
  "activated": true
}
```

`reinforced_edges` は actual serving update だけを保持する。したがって quorum 前の receipt は evidence を含む一方で `reinforced_edges` が空となり、既存 field の意味を変更しない。idempotency replay は保存済み receipt をそのまま返す。

## 検証境界

synthetic regression は quorum `3` の 1 / 2 / 3 / 4 evidence 境界、同一 trace、source-use idempotency、restart、旧 database migration、lexical / zero-hop / uncredited / 別 source、sibling parity、multi-edge / sibling rollback、core / MCP receipt parity を固定する。これは candidate mechanics の検証であり、production quality、external corpus generalization、default 採用を主張しない。

## Related

- [Feedback rank elasticity](feedback-rank-elasticity.md)
- [Sibling relation feedback normalization](sibling-relation-feedback-normalization.md)
- [Optional MCP feedback interface](optional-mcp-interface.md)
- [Decision Structure](Decision-Structure.md)
