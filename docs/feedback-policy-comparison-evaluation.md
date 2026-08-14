# Feedback policy comparison evaluation

## 目的と境界

この文書は Issue #85 の result-free 比較 protocol の正本である。source input は corpus-only merge commit `07b4ed48df8a81ba8c91dbaa17eb740774b6951b` にある16文書と manifest の bytes だけであり、runner は working tree の本文ではなく `git show <commit>:<path>` を読む。現行の local candidate `used + evidence quorum 3 + same-source sibling normalization 1.0` と、`confirmed + decay ratio 0.5 + same-source sibling normalization 1.0` を固定 control と比較する。

この評価は library/MCP default、local q3/s1 registration、production adoption を変更しない。controlled corpus の結果を external corpus generalization、Agent end-to-end efficiency、production quality として表現しない。

## Result-free freeze

protocol ID は `policycmp85-feedback-policy-comparison-v1` である。fixture、gold、query、cohort role、graph projection、engine config、event order、checkpoint、metric、gate array、result schema、exclusive writer、verifier、output path を observed result 不在の freeze commit に固定して push する。

development は Amber を `confirmed-use`、Cobalt を `corrected-use` に割り当てる。全 hard gate 通過時だけ開く holdout は Quartz を `confirmed-use`、Willow を `corrected-use` に割り当てる。split 間では node、path、source URL、edge identity、query を共有しない。各 arm は同じ split の8文書、6 edge、query、event order を使う。

freeze 前の writer/verifier 検査は、登録 query、gold、node、output path と無関係な placeholder identity を temporary directory に exclusive create し、非アルファベット順 field の UTF-8 semantic round-trip を確認して削除する。登録 output は freeze 時に存在せず、writer は `O_EXCL` だけを使う。観測後の再実行、上書き、再集計、field reorder、query/gold/ratio/gate の変更を禁止する。

## Arm と schedule

checkpoint は `0 / 1 / 3 / 10`、各 event の cohort order は `confirmed-use`、`corrected-use` とする。event ごとに fresh relation trace を作り、`selected -> validated -> used` と delayed outcome を記録する。同じ idempotency key の replay は同じ receipt semantics になり、独立証拠として数えない。各 arm は fresh in-memory clone でもう一度全 schedule を再生し、UUIDを除いた semantic result の完全一致を要求する。

1. `control`: source-use と outcome は ledger/store に保存するが feedback application を行わない。
2. `used_q3_s1`: `confirmed_outcome_reinforcement=False`、`relation_feedback_evidence_quorum=3`、`sibling_feedback_normalization=1.0`。
3. `confirmed_r05_s1`: `confirmed_outcome_reinforcement=True`、`confirmation_decay_ratio=0.5`、`sibling_feedback_normalization=1.0`。

ratio `0.5` は初回 increment を1とした無限累積を2未満に抑える mechanics candidate であり、採用値または default ではない。

## Projection と metric

各 cluster の明示 link だけから `source -> route -> terminal` と `source -> sibling` を作る。固定初期 weight は順に `0.8 / 1.3 / 0.9` とする。これは checkpoint 0 に headroom を残し、一回の通常 bounded confirmed update が観測可能な事前登録 projection であり、result 観測後に調整しない。

各 checkpoint は relation rank、MRR、Hit@3、raw graph score、split 内で正規化した graph score、hybrid final score と最大競合との差、corrected sibling / direct lookup / reverse direction / unrelated / non-target sibling rank、全 edge の weight / reinforced count / confirmation count / evidence count、mutation count、正の actual delta 累積、top-2 entry / exit、non-target churn を保存する。source-use / outcome receipt、idempotency replay、fresh-clone replay も同じ observed JSON に保存する。

## Hard gate

gate array の順序と membership は固定し、verifier が observed trajectory から再計算する。

1. `protocol-integrity`: source hash、split identity、schedule、result schema、receipt、idempotency、fresh-clone replay、exclusive output 条件が一致する。
2. `confirmed-diminishing`: candidate は checkpoint 1 で正の bounded update を一回適用し、その後の per-edge delta は非増加である。
3. `used-quorum-boundary`: q3 arm は checkpoint 1 で不変、checkpoint 3 で初めて変更する。
4. `corrected-isolation`: candidate の corrected cohort は全 checkpoint で control と同じ serving edge、rank、control metric を保つ。
5. `confirmed-headroom`: baseline MRR が1未満なら candidate checkpoint 1 は control より strict improvement、1なら safety を伴う non-regression とする。
6. `checkpoint-10-safety`: candidate confirmed MRR は q3/s1 に non-regression、direct / reverse / unrelated rank は control に non-regression とする。
7. `mutation-locality`: mutation は事前登録した credited path と同じ source の uncredited sibling だけに限定し、candidate の corrected cohort、unrelated、lexical、zero-hop へ漏らさない。

全 gate pass の場合だけ `支持`、整合性を保った比較 gate のいずれかが通らない場合は `不支持` とする。protocol failure で比較できない場合は結果を捏造せず `判定不能` として停止する。

## 一回限りの実行

freeze commit の push 後だけ development を一回実行する。development の保存済み `all_pass=true` を verifier が確認した場合だけ holdout を一回実行する。

```powershell
python tools/run_feedback_policy_comparison_evaluation.py --stage development
python tools/run_feedback_policy_comparison_evaluation.py --stage holdout
python tools/run_feedback_policy_comparison_evaluation.py --verify development
python tools/run_feedback_policy_comparison_evaluation.py --verify holdout
```

observed JSON の `interpretation_ja` が human-readable な支持・不支持と一般化境界を保持する。normative protocol 本文は観測後も変更しない。
