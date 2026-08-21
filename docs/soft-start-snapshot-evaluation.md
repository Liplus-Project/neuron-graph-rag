# Soft-start snapshot evaluation

## 目的と判断範囲

この評価は、現在のlocal NGR databaseから一度だけ取得したtransaction-consistent snapshotを使い、次の4 armを同じquery、node role、event順序、checkpointで比較する。

1. `control`: source-useとoutcomeを監査保存するがserving edgeを変更しない。
2. `used_q3_s1`: relation feedback evidence quorum `3`、sibling normalization `1.0`。
3. `confirmed_r05_s1`: confirmed-only、confirmation decay `0.5`、sibling normalization `1.0`。
4. `soft_start_r025_r05_s1`: soft-start ratio `0.25`、confirmation decay `0.5`、sibling normalization `1.0`。

判断対象は、この固定snapshot上でsoft-startがlocal serving cutover候補になれるかだけである。`0.25`と`0.5`は観測前に固定した比較値であり、library defaultや最終採用値ではない。このissueはsource database、live config、project defaultを変更しない。

## Snapshotとprivacy境界

acquisitionはsource databaseをSQLite URI `mode=ro`と`query_only`で開き、SQLite backup APIでprivate destinationへ一度だけ複製する。複製前後のsource container SHA-256が一致し、snapshotの`integrity_check`が`ok`であることを要求する。各armとfresh replayはsnapshotの別temporary cloneだけを開く。

public manifestにはsource locator、capture timestamp、container / snapshot / schema hash、size、table name、選択tableのrow countだけを登録する。snapshot本体、source / snapshotのabsolute path、node本文、credentialを登録しない。fixtureは既にpublicであるnode / edge identifier、固定query、outcome role、mutation許可集合だけを保持する。runnerとverifierはpublic payloadを再帰走査し、private field、absolute path、credential-shaped valueを拒否する。

## Result-free freeze

`soft_start_snapshot_v1` namespaceはfixture、schedule、gate、result schema、result-free audit、manifestを新規作成し、既存の凍結評価artifactを読み替えも変更もしない。manifest pathを初めて追加したcommitのexact blob bytesがhistorical source of truthであり、登録stage実行とresult verificationはcurrent working treeの同名bytesではなく、そのcommitから検証済みbytesを読む。

freeze commitをpushする前は、登録development / holdout outputが存在しないことを確認する。writer / verifierは登録外のplaceholder identityとtemporary outputだけでround-tripを証明する。registered outputはexclusive createし、上書き、再実行、再集計を拒否する。

developmentはfreezeとCI確認後に一度だけ実行する。developmentの全hard gateがpassした場合だけholdoutを一度実行する。失敗または不支持でもquery、ratio、schedule、metric、gate、caseを変更しない。

## Caseとschedule

developmentとholdoutは、それぞれconfirmed cohort、corrected cohort、lexical control、zero-hop controlを持つ。各caseはbaseline後、異なるfresh search traceを使って次を固定順で実行する。

`used_1` → `outcome_1` → `used_2` → `outcome_2` → `used_3` → `outcome_3`

全eventでsource-use / outcome receiptとidempotency replayを保存する。各armは同じsnapshotのfresh cloneで再実行し、runtime identityを除くsemantic payloadが一致することを要求する。

## Metricとhard gate

各checkpointはrelation rank / MRR / Hit@k、score margin、top-k entry / exit、non-target churn、direct rank、reverse relation rankを保存する。全edgeについてweight、reinforced count、evidence count、confirmation countを保存し、credited edge、same-source sibling、unrelated edgeの変化を再計算可能にする。

hard gateは次の7件である。

1. snapshot、case、schedule、receipt、idempotency、fresh-clone replay、exclusive outputのintegrity。
2. 最初の`used`で正のprovisional deltaを適用し、最初のconfirmedとの合計が通常bounded update一回分で、後続confirmationが固定decayへ従うこと。
3. q3/s1が3回目前に不変、confirmed-onlyがconfirmed前に不変で、controlが不変であること。
4. soft-startがcheckpoint 1で先に学習を開始し、最終confirmed cohortのrelation MRRがq3/s1へnon-regressionであること。
5. corrected cohortのprovisional costを保存し、通常bounded incrementの`0.25`以下に限定し、自動negative reinforcementやrollbackを行わないこと。
6. lexical、zero-hop、direct、reverse controlへmutationまたはrank regressionを漏らさないこと。
7. mutationをcredited edgeとconfirmation時のsame-source siblingだけへ限定し、source snapshotを変更しないこと。

## Result解釈

全hard gateがpassしたstageだけを`支持`とする。一件でも不合格なら`不支持`、protocolを完走できなければ`判定不能`をexclusive observed resultへそのまま保存する。developmentが`支持`でない場合はholdoutを開かず、local cutoverを支持しない。holdoutまで`支持`でも、結論は固定local snapshot上のcandidate支持に限定し、別のconfig変更判断なしにlive configを変更しない。

実行入口は次のとおりである。`SNAPSHOT`はpublic repository外のprivate pathを呼出側だけが渡す。

```bash
python tools/run_soft_start_snapshot_evaluation.py --probe --snapshot "$SNAPSHOT"
python tools/run_soft_start_snapshot_evaluation.py --stage development --snapshot "$SNAPSHOT"
python tools/run_soft_start_snapshot_evaluation.py --verify development
```

holdout commandはdevelopment resultの全hard gate passをpreflightが確認した場合だけ同じ形で一度実行する。

## 関連

- [Requirements](requirements.md)
- [Confirmed-outcome feedback reinforcement](confirmed-outcome-feedback-reinforcement.md)
- [Historical source verification](historical-source-verification.md)
- [Decision Structure](Decision-Structure.md)
- [Issue #104](https://github.com/Liplus-Project/neuron-graph-rag/issues/104)
