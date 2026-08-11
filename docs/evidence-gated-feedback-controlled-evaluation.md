# Evidence-gated feedback controlled evaluation

## 目的

このprotocolは、explicit opt-inのevidence quorum `3`とsame-source sibling normalization `1.0`を組み合わせたcandidateが、少数feedbackによる早期rank flipを抑え、3件目の独立relation traceで初回activationしながらcontrolを退行させないかを、repository内の新規controlled corpusに限定して評価する。default採用、production品質、external corpusへの一般化は主張しない。

## Result-free freeze

`evidence-gated-feedback-v1`のcorpus、fixture、gold、schedule、gate、audit、manifest、evaluator、runner、testと本書を、development / holdout observed outputが存在しない状態で一つのcommitとしてpushする。manifestは評価対象candidate sourceのcommitとSHA-256、登録artifactのSHA-256、exclusive output path、hard-gate順序を固定する。

過去のobserved result、evaluator、goldは設計・選択・実行入力に使わない。contamination verifierは登録済みprior auditからallowlistのidentity fieldだけをcanonical projectionし、名前に`result`、`metric`、`gate`を含むsubtreeを読取対象から除外する。development / holdoutはcase、cluster、node、document path、source URL、query語彙、credited edge identityを共有しない。

## Schedule

各stageはheadroomとceilingの2caseを持つ。各caseを次の4variant、feedback count `[0, 1, 2, 3, 4, 10]`で実行する。

| variant | evidence quorum | sibling ratio |
| --- | ---: | ---: |
| `current` | 1 | 0.0 |
| `evidence-only` | 3 | 0.0 |
| `local-only` | 1 | 1.0 |
| `combined` | 3 | 1.0 |

各variant/checkpoint/replayは新しいin-memory engineへ同じcorpus、query、used node、config、timestamp、limitを登録し、0から指定件数の実`search_channels().relation` traceを再生する。前checkpoint stateを流用しない。一つのstage run内で同じcheckpointをfresh engineから2回再生し、UUIDを保存対象から除いたrank、score、edge、evidence、control出力の完全一致をdeterminism gateとする。

## Hard gates

- source/hash、artifact hash、UTF-8、split identity、identity-only contamination、exclusive output、run countが登録値と一致する。
- 全feedbackはrelation trace、固定used node、endpoint/typeを含む単一credited pathに射影される。
- `combined`はcheckpoint 1/2でevidence countだけを増やし、weight、reinforced count、target rank、direct / lexical / directional-negative control rankを変えない。
- headroomの`current`と`local-only`はcheckpoint 1でflipし、`combined`はcheckpoint 3で初めてactivationしてflipする。
- checkpoint 10の`combined` relation MRRは`current`から退行せず、ceilingは全variant/checkpointでrank 1を保つ。
- headroomのpre-quorum relation top-k churnは`combined`が`current`より厳密に小さく、control rankは全checkpointで不変とする。
- evidence-onlyはquorum timing、local-onlyはsame-source sibling deltaを独立に示し、combinedのcredited deltaとsibling deltaが各ablation規則に一致する。
- mutationはactual activation時のcredited edgeと、ratioが1.0の場合のsame-source uncredited siblingだけに限定し、directional-negative edgeを変えない。
- credited edgeまたはsibling更新失敗時はevidence、weight、reinforced count、feedback rowをatomic rollbackする。
- fresh replay、score順序、rank、churn再計算が一致する。

## One-time execution

freeze commitをpushした後、次のcommandでdevelopmentを一回だけ実行する。

```powershell
uv run python tools/run_evidence_gated_evaluation.py development
```

developmentの全hard gateがpassした場合だけholdoutを一回実行する。

```powershell
uv run python tools/run_evidence_gated_evaluation.py holdout
```

runnerはexclusive creationを使い、既存outputを上書きしない。失敗outputも正本としてcommitし、観測後はprotocol artifact、evaluator、runner、test、docs、candidate coreを変更せず、再実行・再集計しない。

## Claim boundary

結果は登録したsynthetic repository corpusと固定scheduleの機構検証に限る。quorum `3`、sibling ratio `1.0`、learning rate、maximum weight、fusion、normalization、defaultの採用判断には使わない。
