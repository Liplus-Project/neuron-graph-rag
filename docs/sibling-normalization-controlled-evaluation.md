# Sibling normalization controlled evaluation

## 目的と主張範囲

この protocol は、`sibling_feedback_normalization` の opt-in candidate を repository-native controlled corpus 上で評価する。baseline は `0.0`、treatment は `1.0` とし、両条件で別々の `NeuronGraphRAG` を同じ corpus、query、時刻、feedback schedule から構築する。

結果の主張範囲はこの固定 corpus に限る。production D1 への一般化、既定値の変更、production 採用はこの評価だけでは認めない。

## Result-free freeze

`sibling_normalization_controlled_v1` の fixture、gold、schedule、manifest、gate、evaluator、runner、test と本書を、observed output が存在しない状態で commit / push する。manifest は評価対象の engine source commit と SHA-256、登録 artifact の SHA-256、split / cluster identity、exclusive output path を固定する。

development と holdout は node ID、cluster ID、query 語彙を共有しない。過去の feedback trajectory evaluator、fixture、gold、schedule、manifest、gate、observed result は選択入力にも実行入力にも使わない。

登録 runner は clean worktree かつ `HEAD == upstream` の freeze 後だけ実行できる。既存 output の上書きを拒否し、development が全 hard gate を通過した場合だけ holdout を一度開く。失敗結果も保存し、protocol 調整や再実行を行わない。

## 実行契約

各 case は実 engine の `search_channels()` から lexical trace と relation trace を得る。lexical trace と direct `search()` の zero-hop trace に対する `record_success()` が edge を変えないことを確認した後、保存済み relation trace ID と固定 used node を `record_success()` に渡す。

headroom case は、baseline の一回の credited reinforcement だけでは rank 2 に残る target が、treatment の同一 reinforcement と same-source uncredited sibling の局所減算によって rank 1 になる境界を固定する。ceiling case は rank 1 target と、同時に credit される sibling edge を持ち、credited edge の treatment / baseline delta 一致と未 credit sibling だけの減算を監査する。

各 split の hard gate は次の九つである。

- headroom relation の treatment 対 baseline strict improvement
- ceiling relation の non-regression
- direct rank の全 case / 全条件 non-regression
- lexical rank の全 case / 全条件 non-regressionと lexical / zero-hop feedback の edge 不変
- directional-negative query で reverse source が現れず、relation ordering が不変
- relation trace の credited endpoint / edge type が gold と一致
- baseline は credited edge だけ、treatment はそれに加えて same-source uncredited sibling だけを変更
- 後段の credited edge を欠落させた失敗 transaction が先行 update と feedback row を rollback
- 同一登録 run 内の fresh engine replay が stable observation で一致

## Artifact

- `tests/fixtures/sibling_normalization_controlled_v1.fixture.json`: development / holdout corpus と atomicity probe
- `tests/fixtures/sibling_normalization_controlled_v1.gold.json`: query、used node、credited path、mutation scope
- `tests/fixtures/sibling_normalization_controlled_v1.schedule.json`: baseline / treatment 係数、checkpoint、時刻、run count
- `tests/fixtures/sibling_normalization_controlled_v1.gate.json`: hard gate と conditional holdout stop rule
- `tests/fixtures/sibling_normalization_controlled_v1.manifest.json`: source と protocol hash、exclusive output
- `tests/fixtures/sibling_normalization_controlled_v1.development.observed.json`: development の一回限りの出力
- `tests/fixtures/sibling_normalization_controlled_v1.holdout.observed.json`: development 全 gate 通過時だけ生成する一回限りの出力

観測値と gate 判定は observed JSON を正本とする。observed output 生成後は evaluator、fixture、gold、schedule、manifest、gate、test、runner、docs を変更しない。
