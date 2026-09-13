# Cross-encoder precision one-shot observation v1

## 範囲

Issue #139 は、freeze merge commit `b681b2a8073e13dfc940b6f3e0c55f86556a21c0` だけをprotocol inputとして、`github-ngr-cross-encoder-precision-v1` のdevelopmentを一度実行した。凍結artifact、candidate順、threshold、query/gold、gate、dependency lock、evaluator、NGR defaultは変更していない。shared database、既存experiment database、github-rag-mcp、feedback/outcome、production serviceへSQLite接続していない。

## Preflight

development claim前に、専用venvへhash付きlockをinstallし、次のexact revisionを専用cacheへ取得した。

- `BAAI/bge-reranker-base@2cfc18c9415c912f9d8155881c133215df768a70`
- `BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`

凍結registryのrequired file全件についてsizeとLFS SHA-256またはgit blob IDを検証した。cacheは `3,427,616,927` bytesである。検証後は `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、local-files-only、`trust_remote_code=False`へ固定し、CPU / float32 / `eval()` / inference mode、batch size 8のsynthetic probeをmodelごとに1回実行した。probeは登録queryを含まない。専用17 tests、audit/probe、変更対象Ruff、project venvのfull 347 testsはclaim前にgreenだった。

shared `C:/Users/smile/.ngrdb/knowledge.db` のraw SHA-256はpreflight前後とも `84a3fc590eee990579e3ef8130294129934fe93e25f18ac249eece19813c261e` である。

## One-shot development result

development claimをexclusive-createし、baseline primary/replay、bge-base primary/replay、bge-v2-m3 primary/replayを6個のfresh processと6個のfresh SQLite databaseで各一度だけ実行した。8 registered queriesを各processで一度実行したため、NGR searchは合計48回である。model側は4 processで合計7,812 query/chunk pair、1,348 forward batchを実行した。primary/replayは各modelでcase bytes相当、ranking hash、activation hashが一致し、database identityはすべて分離している。

6 worker完了後、凍結evaluatorが最初のcandidate cohortを算出する途中で `KeyError: 'ranked_hits'` を返した。candidate caseの `returned_source_paths` が空配列の場合、凍結 `_cohorts()` は空配列を有効な結果として扱わず、candidate caseに存在しない `ranked_hits` へfallbackする。このためresult、candidate selection、case/cohort metric、11 hard gateは生成されておらず、pass/failとして評価しない。これはmodel scoreに基づく調整ではなく、凍結protocolの再現可能なevaluator defectである。

phaseは `development=archived-error`、`holdout=unobserved` である。claim後の再試行、再評価、evaluator修正、threshold/query/gold/gate/selection変更は行っていない。developmentの全hard gate passを証明できないためholdout claimとholdout inferenceは0件である。default変更とGitHub RAG parity v2の開始条件も成立しない。

## Evidence

claim/errorはruntimeからarchiveへbyte-preservingに移送した。6 workerのraw JSONも外部experiment pathから専用archiveへbyte-preservingに複写し、model weight、cache、venv、fresh databaseはgitへ追加していない。raw archive manifestはoriginal/archive path、size、SHA-256、byte identity、execution countを固定する。

- preflight: `f3fd585cac05cc90bb307a92745479b31a88dff185bd74913429044632113d35`
- model verification: `f0be0cde73a75a2d84995a35cda8579c3bd04fdc756a6e2b5c8a1dac166a7512`
- dependency report: `4704d07a3f73b11a600b959e6800c5825a0251a7feb8d08ea93d29b75e77494d`
- development claim: `be3ad76fc7e0e0e2749fddb0111e059812226b616049e4559973dacdda0864e9`
- development error: `8503b1970c74dbd7bd8fc8a8c4240188f8a83af34af37c53b1ab6d394aef0f9f`
- development transport: `97c0292fdcf37682eb675bc0e03eb5f0e2a302cdf56cd66e975a40ada7fb0fbe`
- execution error: `01247f1082552b863e19bfaea173eadaf8e3290e4a0271eda29a2c0707889cbc`
- raw archive manifest: `667cde2474bfbbd5da00c82270bdb9dfbb8dd532c6b9252cf0abb441faddeabf`

successor protocolは、candidateが空の返却集合を持つsynthetic raw resultをfreeze前の同じevaluator round-tripへ通し、空配列とfield欠落を区別してから新しいresult-free versionとして凍結する必要がある。本versionのerror evidenceを修正・置換して再利用しない。
