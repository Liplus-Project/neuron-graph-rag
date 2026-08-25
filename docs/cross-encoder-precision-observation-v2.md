# Cross-encoder precision one-shot observation v2

## 範囲

Issue #143 は、freeze merge commit `36c17aac3b49587c97d96bac51db668bf834177b` だけをprotocol inputとして、`github-ngr-cross-encoder-precision-v2` をone-shot観測する。v2凍結artifact、candidate順、threshold、query/gold、gate、dependency lock、evaluator、NGR defaultは変更しない。

v1 raw packet、evidenceのsemantic content、database、run outputはv2観測の入力にしない。v1 evidenceはv2 manifestが固定したbyte hashとの一致だけを確認する。既存model cacheから再利用できるのは凍結registryに列挙されたweight、config、tokenizer bytesだけであり、claim前にrequired file全件のsizeとhashを再検証する。shared database、既存experiment database、github-rag-mcp、feedback/outcome、production serviceへ接続しない。

## Preflightと専用実行面

観測はv2専用venv、run root、fresh database、fresh process、claim/result/error/transportを使用する。model cache検証後は `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、local-files-only、`trust_remote_code=False`へ固定する。synthetic probeは登録queryを含まず、観測stageのmodel inference countに含めない。

preflightは凍結protocol、model bytes、dependency、offline probe、専用/full tests、audit/probe、Ruff、shared databaseの前後SHA-256を検証し、claim前のevidenceとして独立commitへ保存する。preflightを通過するまでdevelopment claimを作らない。

## One-shot lifecycle

development claimをexclusive-createし、baseline primary/replay、2 modelのprimary/replayを6個のfresh processとfresh SQLite databaseで各一度だけ実行する。worker raw packetを評価前にv2専用archiveへbyte-preservingに保存する。全development hard gateがpassした場合だけholdoutを一度開く。

claim後の例外またはgate failureではevidenceを保存し、同じprotocol versionの再試行、再評価、threshold/query/gold/gate/selection変更を行わない。model weight、cache、venv、fresh databaseはgitへ追加しない。観測結果がpassでもNGR default、MCP config、GitHub RAG parity v2を変更しない。

## 状態

preflightは完了した。2 exact model revisionのrequired file全件を検証し、再利用cacheは `3,427,616,927` bytesだった。offline/local-files-only、CPU / float32 / `eval()` / inference mode、batch size 8のsynthetic probeをmodelごとに1回実行した。probe forward countは2であり、登録queryを含まない。v2 freeze 15 tests、observation 7 tests、full 370 tests、audit/probe、変更対象Ruffはgreenだった。

shared `C:/Users/smile/.ngrdb/knowledge.db` のraw SHA-256はpreflight前後とも `84a3fc590eee990579e3ef8130294129934fe93e25f18ac249eece19813c261e` である。preflight完了時点のclaim/query/observed-stage inference countは `0/0/0`、phaseはdevelopment/holdoutともに `unobserved` である。

- preflight: `4c35e1846486cba6ac8b001df62f79fa4f39df5df3df0711a7769ed9f10df703`
- model verification: `2c78d8df182e1a53127984d09161b21f1fe7dd8709f78df793abb653bb92b3cc`
- dependency report: `5b80fa6c2470cf99004294b2b0d03bfcd15c102d62dda45594c8d1d8d44e8ad7`
- preflight commands: `77c29f50eb8368e5dc59bdecbbd08c6e9e681b84777aeca2b928267cdf58eadc`

## One-shot development result

preflight evidence commit `2e6c234f43f52b23871b297ea638ab0b61e0dead` をpushした後、development claimをexclusive-createした。baseline primary/replay、bge-base primary/replay、bge-v2-m3 primary/replayを6個のfresh processと6個のfresh SQLite databaseで各1回だけ実行した。8 registered queriesを各processで一度実行したため、NGR searchは合計48回である。model側は4 processで合計7,812 query/chunk pairを評価した。primary/replayは各modelでcase、ranking hash、activation hashが一致し、6 database identityはすべて分離している。claim/worker arm/retry countは `1/6/0`、holdout claim/worker armは `0/0` である。

4 candidateはすべて11 hard gateを通過せず、selected candidateはない。共通するfailed gateは `positive-case-rank-non-regression`、`positive-cohort-mrr-hit-at-5-non-regression`、`positive-expected-source-top-5-completeness`、`relation-source-edge-only-provenance` の4つである。phaseは `development=archived-failed`、`holdout=unobserved` である。停止規則に従いholdoutは開封せず、retry、再評価、threshold/query/gold/gate/selection変更を行っていない。NGR defaultとMCP configは変更せず、GitHub RAG parity v2開始条件は成立しない。

shared databaseのraw SHA-256は実行前後とも `84a3fc590eee990579e3ef8130294129934fe93e25f18ac249eece19813c261e` である。claim/resultはruntimeからarchiveへbyte-preservingに移送し、6 raw worker packetも専用external run rootからv2 archiveへbyte-preservingに保存した。model weight、cache、venv、fresh databaseはgit変更に含めない。

観測後のv2専用23 tests、full 371 tests、変更対象Ruffはgreenである。専用testはresultのfrozen evaluator再検証、全11 gateとcandidate結論、primary/replay determinism、fresh database identity、48 query、shared database不変性、claim/result/raw packetのbyte hash、retry/error/holdout artifact不在を固定する。

- development claim: `437450a4e8fdcc488b4409ac14cff9133c152c8945a11081e268f93ae08efdbc`
- development result: `83e7cbbc7e09db2189edc535372d317ce69810c5601149d6acf5b2e308bae007`
- development transport: `7eafcb3a442bc3a5da94a25c0867b4bb283b468fd60dcd11444c2bb60e9d0838`
- raw archive manifest: `7e9b3bc45fa6a7c65fad0f9c45414cad18fe17dc6559ea998179911be054aca7`
- execution report: `d41ba9a93b8048c0be15bb3fcf7d830a744ade6813f663e727c2a62e226496cd`
