# Outcome-driven feedback deactivation

## 目的と有効化境界

この文書は、soft-start が加算した relation feedback を、保存済み trace と credited path へ因果帰属できる negative delayed outcome に限って可逆に不活性化する candidate の正本である。candidate は `neuron_graph_rag.evidence_feedback.EngineConfig` の `outcome_driven_feedback_deactivation=True` で明示的に有効化し、`soft_start_feedback_reinforcement=True`、`soft_start_feedback_ratio`、`confirmation_decay_ratio` を同時に必要とする。

既定値は `False` である。無効時は `corrected`、`rolled_back`、`superseded` を従来どおり audit-only で保存し、既存 config fingerprint、library default、MCP default、local serving config を変更しない。candidate は一般的な負の reinforcement、時間経過だけの decay、未利用だけを理由にした減衰を導入しない。

## Contribution journal と exact reversal

SQLite は soft-start の provisional contribution と各 confirmation contribution を独立した永続単位として保存する。一つの contribution は次を一体として持つ。

- 保存済み trace ID、source record ID、credited edge identity、初期 baseline weight
- credited edge へ実際に加算した正の actual delta
- その加算が同じ transaction で発生させた same-source sibling ごとの負の actual delta
- active / reversed 状態と reversal を行った outcome ID

`corrected` と `rolled_back` は、outcome の node と保存済み credited path が一致する active contribution だけを対象にする。credited delta と、その同じ contribution に属する全 sibling normalization delta を一つの可逆単位として逆適用する。片側だけを戻さず、別 trace、別 contribution、別 source、uncredited edge、lexical path、zero-hop path は変更しない。

credited edge の逆適用は contribution の登録 baseline より下へ下げない。これは誤りに帰属された加算を取り消す境界であり、基礎 weight 以下の punitive update ではない。後から別 contribution が同じ edge へ加えた delta は保持する。candidate 有効時、active contribution を持つ edge は他 contribution の sibling normalization 対象から除外し、逆写像を曖昧にしない。

## Dormancy と reactivation

`superseded` は帰属可能な credited relation edge を削除せず dormant として保存する。dormant edge は通常の graph activation に使う outgoing edge 集合から除外するが、edge、証拠、trace、contribution journal は保持する。

同じ保存済み credited path に後から `confirmed` が記録された場合、confirmation transaction の先頭で dormant 状態を解除する。receipt は `dormancy_changes` または `reactivated_edges` として old/new state を返す。再活性化だけで過去 contribution を再適用せず、新しい confirmation delta は既存 soft-start schedule に従う。

## Transaction、idempotency、receipt

outcome row、node association、credited / sibling inverse、contribution state、dormancy state、idempotency receipt は一つの SQLite transaction に含める。途中で一つでも失敗した場合は outcome と graph mutation の両方を call 前へ戻す。同じ idempotency key と payload の replay は最初の receipt を返し、二重減算、二重 dormancy、二重 reactivation を行わない。restart 後も同じ journal と状態を使う。

core の `OutcomeReceipt` と optional MCP output は次を同じ意味と順序で公開する。

- `deactivation_applied`
- `reversed_contributions` と各 `mutations`
- `dormancy_changes`
- `reactivated_edges`

既存 `reinforcement_applied` は正の confirmation update だけを表し、negative outcome では `False` のままとする。

## Result-free snapshot evaluation

`outcome-driven-feedback-deactivation-v1` は fresh transaction-consistent private snapshot の clone 上で、`control` と `deactivation_candidate` を比較する。development / holdout は互いに異なる固定 case identity を使い、`corrected`、`rolled_back`、`superseded`、unattributed control を含む。protocol、fixture、schedule、gate、result schema、manifest、exclusive output path は registered result 不在の freeze commit で固定する。

result-free preflight は snapshot hash、integrity、schema、case capacity、registered baseline、relation path identity、same-source sibling capacity、privacy、writer / verifier round-trip、development / holdout output 不在を検証する。この Issue では development と holdout を実行しない。freeze が main へ squash mergeされた後の successor Issue が development を一度だけ実行し、全 hard gate 通過時だけ holdout を一度だけ実行する。観測後に query、case、ratio、schedule、metric、gate を変更しない。

public artifact は locator、capture時刻、logical source / snapshot hash、snapshot container hash、schema hash、row count、public node / edge identifier、query、baseline数値だけを持つ。private本文、credential、absolute private path、snapshot本体をcommitしない。source database、live config、private snapshotは実行を通して変更しない。

## 解釈境界

mechanics test と result-free preflight の通過は、candidate の有用性や local adoption を支持しない。successor development が不支持または判定不能なら holdout を開かず、現行 policy を維持する。holdoutまで全 gate を通過しても、支持範囲は固定 local snapshot と登録 schedule に限定し、library default や一般化は別判断とする。

## 関連

- [Requirements](requirements.md)
- [Confirmed-outcome feedback reinforcement](confirmed-outcome-feedback-reinforcement.md)
- [Baseline-aware soft-start snapshot evaluation](baseline-aware-soft-start-snapshot-evaluation.md)
- [Optional MCP Feedback Interface](optional-mcp-interface.md)
- [Decision Structure](Decision-Structure.md)
- [Issue #111](https://github.com/Liplus-Project/neuron-graph-rag/issues/111)
