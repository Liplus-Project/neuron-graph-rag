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

one-shot resultは未生成である。生成後は、claim/query/inference/arm/retry count、development/holdout phase、selected candidate、gate failure、shared databaseの不変性、archive hashを追記する。
