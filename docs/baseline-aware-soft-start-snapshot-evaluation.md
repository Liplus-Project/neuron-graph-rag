# Baseline-aware soft-start snapshot evaluation

## 目的とsuccessor境界

snapshot comparison v1はdevelopment 7 hard gate中6件を通過したが、`policy-boundaries`を満たさずholdoutを開かなかった。対象edgeがinitial evidence count `1`を持ち、q3/s1がfresh `used_2`でquorum `3`へ到達した一方、v1 gateは空baselineからの`used_3`発火を固定していた。この不支持result、protocol、gate、解釈、private snapshotは凍結したまま保持する。

v2はminor tweakやv1再実行ではない。`baseline_aware_soft_start_snapshot_v2` namespace、fresh transaction-consistent private snapshot、新規fixture / manifest / outputを使う。v1 private snapshotとobserved resultを入力にせず、developmentのcredited edge identityもv1 observed developmentから分離する。

## Baseline contractとcapacity

confirmed / corrected relation caseごとに、snapshot上のcredited edgeについて次を結果前に登録する。

- weight
- reinforced count
- relation feedback evidence count
- confirmation count

q3/s1のexpected first mutation eventは次の式だけで導出する。

`max(1, relation_feedback_evidence_quorum - registered_initial_evidence_count)`

event budgetはfresh trace 4回である。derived first mutationはfinal eventより前に発生し、少なくとも一回の後続fresh evidence activationも観測できなければならない。snapshot actual stateと登録baselineの不一致、式との不一致、capacity不足、selected credited path不一致はresult-free preflight failureである。この場合はregistered resultを作らず、failure reportをIssueへ保存して停止する。

## 固定armとschedule

同じfresh snapshotの別cloneへ次の4 armを適用する。

1. `control`: audit-only。
2. `used_q3_s1`: evidence quorum `3`、sibling normalization `1.0`。
3. `confirmed_r05_s1`: confirmed-only、decay `0.5`、sibling normalization `1.0`。
4. `soft_start_r025_r05_s1`: provisional ratio `0.25`、decay `0.5`、sibling normalization `1.0`。

各caseはbaseline後に、fresh traceで`used_n`、同じtraceに対する`outcome_n`を`n=1..4`の順に実行する。armごとにfresh clone replayを行い、runtime identityを除くsemantic payload一致を要求する。

## Result-freeとprivacy

source databaseはSQLite URI `mode=ro`と`query_only`で開き、backup APIでprivate destinationへ一度だけ複製する。source containerの前後SHA-256一致とsnapshot `integrity_check`を要求し、arm / replayはsnapshot cloneだけを変更する。

public artifactはsource locator、capture timestamp、container / snapshot / schema hash、table name、選択row count、public node / edge identifier、query、baseline数値だけを持つ。snapshot本体、node本文、credential、absolute private pathを含めない。writerはexclusive createし、registered resultの上書き、再実行、再集計を拒否する。

v1のresult-free commitとobserved commitはmainへのsquash mergeで一つのcommitになり得るため、mainのmanifest初回追加commitだけからpre-observation development output不在を推論しない。これは履歴上証明できなくなった事実だけを弱める境界変更である。v1 verifierは同commitのhistorical protocol blob hash、既存result-free audit literal、同commitに含まれるdevelopment outputを既存exclusive result verifierで検証し、未開封holdoutの不在を引き続き明示検証する。v1 protocol、gate、result、audit、manifestのbytes自体は変更しない。

## Hard gate

1. snapshot / protocol / case / privacy / baseline / receipt / idempotency / semantic replay / exclusive output integrity。
2. initial evidenceから導出したq3 first mutationと実測の一致、quorum前不変、後続fresh evidence発火。
3. confirmed-onlyのused時不変、独立confirmedごとの`1.0, 0.5, 0.25, 0.125` decay。
4. soft-start first used provisional、first confirmed remainder、後続fixed decay。
5. soft-startのlearning latency先行と最終q3/s1 relation quality non-regression。
6. corrected cohortのprovisional cost上限、自動negative reinforcement / rollbackなし。
7. lexical、zero-hop、direct、reverse、unrelated controlのrank / mutation safety。
8. credited edgeとconfirmation時same-source siblingだけへのmutation、source snapshot不変、fresh-clone replay。

このIssueはprotocol、fixture、runner、verifier、tests、docsをregistered output不在のresult-free stateで固定するfreeze-only phaseである。developmentとholdoutは実行せず、squash merge後に別Issueの新規registration commitを使ってdevelopmentを一度だけ実行する。その全8 gate pass時だけholdoutを一度開く。不支持または判定不能を保存してもquery、ratio、case、schedule、metric、gateを変更しない。このphase分離により、freeze commitのoutput不在と後続observed commitをmain history上で別々に検証可能にする。

## 解釈境界

development不支持または判定不能ではholdoutを開かずlocal cutoverを支持しない。holdoutまで全gate passしても、支持範囲は固定local snapshot上のcandidateに限定する。このIssue内ではsource database、live config、library defaultを変更しない。

```bash
python tools/run_baseline_aware_soft_start_snapshot_evaluation.py --probe --snapshot "$SNAPSHOT"
python tools/run_baseline_aware_soft_start_snapshot_evaluation.py --stage development --snapshot "$SNAPSHOT"
python tools/run_baseline_aware_soft_start_snapshot_evaluation.py --verify development
```

## 関連

- [Requirements](requirements.md)
- [Soft-start snapshot evaluation v1](soft-start-snapshot-evaluation.md)
- [Confirmed-outcome feedback reinforcement](confirmed-outcome-feedback-reinforcement.md)
- [Historical source verification](historical-source-verification.md)
- [Decision Structure](Decision-Structure.md)
- [Issue #106](https://github.com/Liplus-Project/neuron-graph-rag/issues/106)
