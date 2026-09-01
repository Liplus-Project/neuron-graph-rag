# Source-grounded relation-seed retrieval experiment

## 目的

Issue #200は、query内で明示されたsource documentをrelation seedとして固定source graphへ接続するretrieval-only candidateが、original / full-query NGR defaultよりrelation retrievalを改善し、direct / semantic / negative controlを維持できるかを検証する。cross-encoderは使わず、candidateが全gateを通った場合だけ別successorで再評価する。

## Source acquisition

sourceは`Liplus-Project/neuron-graph-rag`のcommit `74a7ae1b4b9dbe822ef719e4a4b7d0a8b5b3066c`にある20 Markdown filesを`git show <commit>:<path>`でread-only取得する。corpusはrepository、full commit、path、Git blob SHA、content SHA-256、commit-pinned source URL、generatorを保持する。

relationは同じ固定corpus内を指すrelative Markdown linkだけを`markdown-relative-link-regex-v1`で決定論的に抽出する。各edgeはsource path、target path、`markdown_link` edge type、取得方法、source / target content SHA-256を保持する。fixture-authored edgeや外部URL、anchor-only linkはsource-grounded relationに含めない。

## Fresh splitとgold隔離

development / holdoutは結果観測前に各8件へ分離し、direct lexical、semantic paraphrase、relation linked、negative controlを2件ずつ固定する。case ID、gold identityはsplit間で交差させない。v19、v21、v23、GitHub retrieval parity v1とのquery normalized Jaccard similarityが0.72以上のcaseと、同一gold signatureをfail closedにする。

workerはcorpus、source-grounded relation、queryだけを読み、goldを受け取らない。finalizerだけがgoldを開き、protocol validityを性能比較より先に確定する。baselineのrelation失敗はprotocol invalidではなく性能結果である。

## Armsとgate

- `original-full-query-ngr-default`: full queryを変更せず、現行`NeuronGraphRAG()` defaultで検索するbaseline。
- `source-grounded-relation-seed`: relation intentとsource documentのpath、filename、stem、H1 titleがquery内で明示された場合だけ、そのsourceから固定edgeで到達するtargetをpath provenance付きで候補先頭へ置く。複数targetはfull-query NGR default score順、同点時はpath順に固定し、残りのdefault hitを続ける。goldや期待targetは参照しない。

candidateはrelation path completeness、relation MRR、relation hit@5をbaselineから厳密に改善しなければならない。direct / semanticはper-case rank、cohort MRR、hit@5を退行させず、negative forbidden countを増やさない。primary / fresh replayは完全一致を要求する。全hard gate通過時だけcandidateを選択する。

## One-shot boundary

各armのprimary / replayはそれぞれfresh SQLiteを使う。shared SQLiteはbyte hashの前後確認だけを行い、開かない。runnerはworkerへcorpus、source-grounded relation、queryだけを渡し、全worker完了後にfinalizerが初めてgoldを開く。primary / replay raw packetは最終outputへ残す。

各stageはworker開始前に登録claimをexclusive createする。claimは失敗時も削除せず、claimまたはoutputが既にあれば再実行を拒否するため、attemptは1回、retryは0である。freeze PRではdevelopment / holdoutを実行せず、claim / outputを作らない。

freezeがmainへmergeされた後のsuccessorだけがdevelopmentを一度実行できる。developmentでcandidateが全gateを通った場合だけholdoutを一度開く。同一protocolの再実行、output上書き、観測後のquery、gold、edge、metric、gate変更は禁止する。passing resultでもNGR default、MCP config、physical integrationを自動変更しない。

## Result-free verification

freeze時は次を確認する。

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_source_grounded_relation_observation -v
python tools/acquire_source_grounded_relation_corpus.py --output tests/fixtures/github_source_grounded_relation_v1.corpus.json --verify
python -c "from neuron_graph_rag.source_grounded_relation_observation import audit_result_free; print(audit_result_free())"
```

これらはsource objectとprotocol contractだけを検証し、development / holdout armを実行しない。
