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

preflight evidenceとone-shot resultは未生成である。生成後は、claim/query/inference/arm/retry count、development/holdout phase、selected candidate、gate failure、shared databaseの不変性、archive hashをこの文書へ追記する。
