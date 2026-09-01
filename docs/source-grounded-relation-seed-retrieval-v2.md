# Source-grounded relation-seed retrieval v2

## 目的

Issue #203は、v1のfreeze identityと観測後のappend-only registryを別責務に分離する。v1のresult-free testが正当なone-time observation artifactを永久に拒否したlifecycle contradictionを解消し、同じv2 protocol、manifest、runner、testを変更せずにresult-freeからconditional holdout完了まで監査できるようにする。

本protocol commitはresult-freeであり、development / holdoutを実行しない。v1のdevelopment attemptはadmissible performance evidenceから除外し、v2のfixture、algorithm、gate、performance比較へ再利用しない。v1のfreeze artifactとIssue #202 branch上のclaim、raw packet、outputは変更も削除もしない。

## Fresh identity

v2はv1と異なるprotocol ID、fixture stem、claim / raw / output pathを持つ。sourceは`Liplus-Project/neuron-graph-rag`のcommit `8b3cdf5052cc382687cf3efbb6d728ddd473d75d`にあるrepository-native controlled v3の15文書を`git show`でread-only取得する。このpath集合はv19、v21、v23、GitHub retrieval parity v1、source-grounded v1の登録source corpusと交差しない。source commit、path、Git blob、content SHA-256、source URLを固定し、relative Markdown linkだけをsource-grounded relationとして抽出する。

development / holdoutは結果観測前に各8件へ分離し、case IDとgold identityを交差させない。v19、v21、v23、GitHub retrieval parity v1、source-grounded v1の固定identityをhash検証し、query similarityとgold signatureの再利用をfail closedにする。candidate algorithm、metric、hard gateはv1から変更しない。

## Freeze identityとregistry lifecycle

freeze identity verifierは、指定commitがmanifestの第一親上の一意な初回導入commitであり、`origin/main`に含まれ、commit内とruntimeのmanifest / artifact bytesが一致し、commitに登録claim / raw / outputが存在しないことだけを検証する。

repository lifecycle auditはcurrent checkoutのappend-only registryだけを検証する。claim、raw packet、outputはcanonical JSONかつmanifest登録pathへのexclusive createとし、protocol commit、stage、arm、run、attempt 1、retry 0を一致させる。raw packetはrunnerの固定順序によるprefixとして部分保存を認め、欠損を埋めるretryや既存byteの上書きを認めない。outputがある場合はdiskから4 packetを再読込してfinalizer結果を再計算し、exact bytesを照合する。

同じauditは次を受理する。

- result-free
- development claimのみ
- development partial raw packet
- development failedまたはcandidate未選択でholdout unopened
- development passedかつcandidate selectedでholdout eligible
- eligibleなholdoutのclaim / partial raw packet / completed output

holdoutはdevelopment outputが全hard gate通過かつcandidate selectedの場合だけ開く。tamper、非canonical JSON、schema / identity mismatch、raw packetのgap、outputとdisk packetの不一致、eligibleでないholdout artifactをfail closedにする。

## One-shot execution boundary

workerはcorpus、relation、queryだけを受け取りgoldへ接触しない。各armのprimary / replayはfresh SQLiteを使い、raw packetを完了直後にexclusive createする。finalizerだけがdisk上の完全な4 packetを再読込してからgoldを開く。shared SQLiteはSHA-256の前後比較だけに使い、SQLiteとして開かない。

developmentがfailedまたはcandidate未選択ならholdoutを開かない。candidate selectedの場合だけ別successor issueでholdoutを一度実行できる。passing resultでもNGR default、MCP config、shared database、physical integrationを変更しない。

## Result-free verification

次のprobeはdevelopment / holdout queryを実行せず、shared databaseを開かない。

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_source_grounded_relation_observation_v2 -v
python tools/acquire_source_grounded_relation_corpus_v2.py --output tests/fixtures/github_source_grounded_relation_v2.corpus.json --verify
python -c "from neuron_graph_rag.source_grounded_relation_observation_v2 import audit_result_free; print(audit_result_free())"
```
