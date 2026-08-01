# Neural dynamics experiment

## 目的

現行の正方向加算だけでなく、有限活性 budget、側方抑制、query-conditioned transmission、反復競合を同一 benchmark で比較する。relation 改善を保ちながら negative control の悪化を抑える候補だけを選び、独立 holdout で一度だけ確認する。

この文書、manifest、development 入力、holdout 入力は variant result を観測する前に固定する。観測後に gold、doc path、閾値、parameter、選択規則、停止規則を変更しない。

## 固定入力

- manifest: `tests/fixtures/d1_liplus_dynamics_experiment.manifest.json`
- development fixture / gold: PR #10 の12件を byte 内容ではなく canonical JSON hash で固定
- holdout fixture: `tests/fixtures/d1_liplus_dynamics_holdout.json`
- holdout gold: `tests/fixtures/d1_liplus_dynamics_holdout.gold.json`
- holdout provenance: `tests/fixtures/d1_liplus_dynamics_holdout.provenance.json`
- holdout は production D1 `search_docs` / `doc_edges` から既存 read-only acquisition tool で取得した9 node / 11 `mention` edge の弱連結 subgraph
- holdout の9 doc path は development の12 doc path と重複しない
- holdout gold は direct lookup / relation / negative control 各3件で、期待 node、許容 rank、public source URL、relation の期待 endpoint / edge type を持つ
- provenance は schema fingerprint、coverage、取得時刻、zero-write evidence、`known_gaps=[]` を保持する

## 固定探索空間

variant は13件で、上限24件を超えない。同じ family の微細な総当たりは行わない。

| family | variant |
|---|---|
| current positive additive | `current` |
| finite activation budget | `budget-025`, `budget-050`, `budget-100` |
| lateral inhibition | `inhibition-010`, `inhibition-025`, `inhibition-top4` |
| query-conditioned transmission | `query-floor-020`, `query-floor-040`, `query-floor-060` |
| recurrent competition | `recurrent-balanced`, `recurrent-selective`, `recurrent-conservative` |

全 variant は同じ sparse / dense / entry / graph weight、seed count、hop limit、hop decay を使う。family 固有 parameter の literal は manifest を正本とする。特定 query 文字列や node ID の例外は持たない。

## development 選択規則

全13 variant を development set で一度評価し、全体・cohort 別 MRR / Hit@3 / rank、説明 path、feedback isolation、step 数、展開数、活性総量、収束、停止理由を保存する。

候補 gate は current に対して次をすべて満たすこととする。

1. relation MRR が弱く改善する。
2. negative-control MRR が弱く改善する。
3. 1 または 2 の少なくとも一方が厳密に改善する。

gate 通過候補から relation MRR と negative-control MRR の Pareto frontier を作る。複数候補の tie-break は、worst-cohort MRR の高い順、平均展開数の少ない順、構造複雑度の低い順、variant ID の辞書順とする。gate 不合格と Pareto 支配された variant も result から削除しない。

候補がなければ `current` を維持し、holdout を開かず終了する。

## holdout 停止規則

development で候補が一意に選ばれた場合だけ、`current` と選択候補を holdout で一度評価する。result file が既に存在する場合、runner は上書きを拒否する。

次をすべて満たす場合だけ候補を採用する。

1. direct lookup、relation、negative control の各 MRR が current から退行しない。
2. 全 relation case で固定 endpoint / edge type path が一致する。
3. credited feedback edge が1本以上変化する。
4. uncredited edge と非対象 case rank が変化しない。

一つでも満たさなければ既定は `current` のままとする。holdout result を見て parameter、gold、doc path、閾値を調整しない。

## 実行境界

freeze commit より前に development / holdout runner を実行しない。freeze 後は次の順序だけを許可する。

1. development stage を実行し、versioned result を新規作成する。
2. frozen selection rule で候補を決める。
3. 候補がなければ停止する。
4. 候補があれば holdout stage を一度だけ実行し、versioned result を新規作成する。
5. 採用または不採用を frozen stop rule から機械的に記録する。

品質数値を CI 合格閾値にはしない。CI は manifest hash、入力分離、determinism、再生成一致、停止規則の適用を検証する。

## 観測結果

freeze commit `4f240dd` の push 後、development stage を一度実行した。全結果は `tests/fixtures/d1_liplus_dynamics_experiment.development.result.json` に保存する。

| variant | direct MRR | relation MRR | negative MRR | candidate gate | Pareto |
|---|---:|---:|---:|---|---|
| `current` | 1.0000 | 0.3833 | 0.7083 | reference | - |
| `budget-025` | 1.0000 | 0.3833 | 1.0000 | pass | frontier |
| `budget-050` | 1.0000 | 0.3833 | 1.0000 | pass | frontier |
| `budget-100` | 1.0000 | 0.3833 | 1.0000 | pass | frontier |
| `inhibition-010` | 1.0000 | 0.3833 | 0.7083 | fail | - |
| `inhibition-025` | 1.0000 | 0.3833 | 0.6667 | fail | - |
| `inhibition-top4` | 1.0000 | 0.3750 | 0.7083 | fail | - |
| `query-floor-020` | 1.0000 | 0.3333 | 1.0000 | fail | - |
| `query-floor-040` | 1.0000 | 0.3833 | 1.0000 | pass | frontier |
| `query-floor-060` | 1.0000 | 0.3833 | 1.0000 | pass | frontier |
| `recurrent-balanced` | 0.7083 | 0.6333 | 0.5417 | fail | - |
| `recurrent-selective` | 0.8750 | 0.4524 | 0.6667 | fail | - |
| `recurrent-conservative` | 0.6250 | 0.6333 | 0.5417 | fail | - |

Pareto 支配された gate 通過候補はなかった。5候補は relation / negative-control 軸で同値の frontier になり、平均展開数と構造複雑度も同値だったため、最後の辞書順 tie-break で `budget-025` を選択した。recurrent family は relation を改善した一方で direct / negative-control を退行させたため、候補 gate を通らなかった。この tradeoff を失敗結果として保持する。

## Holdout 判定

development 選択後、holdout stage を一度だけ実行した。全結果は `tests/fixtures/d1_liplus_dynamics_experiment.holdout.result.json` に保存する。

| variant | direct MRR | relation MRR | negative MRR | path | feedback isolation |
|---|---:|---:|---:|---:|---:|
| `current` | 1.0000 | 0.3611 | 1.0000 | 3/3 | pass |
| `budget-025` | 1.0000 | 0.3254 | 1.0000 | 3/3 | pass |

`budget-025` は `holdout-relation-history-architecture` を rank 4 から rank 7 へ退行させた。他2 relation case は rank 2 と rank 3 のままだった。全 relation path は一致し、credited edge 1本だけが変化し、uncredited edge と非対象 rank の変更はなかった。

しかし relation cohort MRR が current 0.3611 から 0.3254 へ退行したため、固定 stop rule の `no_cohort_regression` を満たさない。`budget-025` は不採用とし、既定は `current_positive_additive` のまま変更しない。holdout を見た後の parameter、gold、doc path、閾値の調整は行わない。

## 適用限界

development と holdout はいずれも固定 Li+ wiki subset と feature-hashing encoder に限定される。D1 は損失あり search snapshot であり、一般 corpus、learned embedding、別 graph topology へ外挿しない。今回の結果は有限 budget が development negative control を回復できても、独立 subgraph の relation 品質を維持する一般則にはならなかったことを示す。
