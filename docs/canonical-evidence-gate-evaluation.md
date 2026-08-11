# Canonical evidence gate evaluation

## 目的

この文書は、evidence quorum `3` と same-source sibling normalization `1.0` の組合せを、既定値を変更せず独立に評価する Issue #77 の result-free protocol の正本である。`neuron_graph_rag.evidence_feedback` の明示 opt-in path だけを使い、legacy engine、MCP default、採用値、一般化は変更しない。

## Identity と contamination 境界

protocol ID は `canon77-canonical-gates-v1`、node と edge prefix は `canon77-`、corpus は `corpora/canonical-evidence-gates-v1`、source URL は `https://example.invalid/ngr/canon77/` 以下とする。development と holdout は query token、node、URL、cluster、credited edge を共有しない。prior artifact は公開 path / filename 由来の namespace だけを identity registry に固定し、prior result、metric、gate、gold を読まない。

## 固定設計

checkpoint は `0, 1, 2, 3, 4, 10`、variant は current `(q1,s0)`、evidence-only `(q3,s0)`、local-only `(q1,s1)`、combined `(q3,s1)` とする。各 checkpoint は fresh in-memory engine から event を再生し、各 event は同じ relation trace を一度 duplicate replay して evidence、weight、rank が重複更新されないことを確認する。

各 split は headroom と ceiling の二 strata を持つ。headroom は current / local-only が checkpoint 1、combined が checkpoint 3 で target rank 1 になる境界を固定する。ceiling は全 checkpoint / variant で target rank 1 の non-regression を要求する。combined checkpoint 1 / 2 は evidence-only と weight、reinforced count、rank が一致し、baseline から変わらない。

全 feedback は実際の `search_channels().relation` trace から固定 endpoint / type の path を credit する。direct lexical rank、reverse-direction control、unrelated edge、non-target ordering は不変とする。credited failure と sibling failure は transaction 内へ missing edge を注入し、evidence、weight、reinforced count、feedback row の全 rollback を要求する。

## Canonical gate schema

gate の唯一の正本は gate artifact の非アルファベット順 array である。observed result も同じ順序の array を保持する。

```json
{
  "gates": [
    {"gate_id": "trace-credit", "passed": true},
    {"gate_id": "canonical-roundtrip", "passed": true}
  ]
}
```

verifier は mapping iteration を比較しない。array であること、完全な ID membership、登録順、重複なし、boolean `passed`、`all_pass` 再計算を検証する。freeze 前には登録 query / corpus / gold / output を使わない placeholder gate ID と temporary output で、実 writer の exclusive creation と実 verifier の非アルファベット順 round-trip を一度実証する。

## Result lifecycle

fixture、gold、schedule、gate、identity registry、audit、manifest、evaluator、runner、tests、corpus、docs は observed result 不在の単一 freeze commit として push する。freeze 後に development を一度だけ実行する。全 gate pass の場合だけ holdout を一度だけ実行する。output は `O_EXCL` で生成し、競合時は既存 byte を変更しない。

観測後は protocol、artifact、code、tests、docs を変更せず、再実行、再集約、再選択をしない。不合格または protocol failure も immutable result として保存し、その時点で停止する。

## Gate

1. source/hash/artifact/UTF-8/identity/output absence と canonical round-trip。
2. relation trace、固定 credited path、duplicate replay の provenance。
3. combined checkpoint 1 / 2 の pre-quorum stability。
4. headroom threshold と combined pre-quorum churn の優位。
5. checkpoint 10 の combined rank / MRR non-regression と ceiling safety。
6. direct lexical、reverse-direction、unrelated edge、non-target control 不変。
7. ablation ごとの mutation locality。
8. credited / sibling failure の atomic rollback。
9. score、rank、MRR、mutation、gate の deterministic recompute。

## 実行

freeze commit の push 後だけ次を実行する。

```powershell
python tools/run_canonical_gate_evaluation.py --stage development
python tools/run_canonical_gate_evaluation.py --stage holdout
```

二行目は development の `all_pass=true` を確認した場合だけ許可する。

## Related

- [Evidence-gated local feedback reinforcement](evidence-gated-local-feedback-reinforcement.md)
- [Decision Structure](Decision-Structure.md)
