# Cross-encoder precision observation v19

## 目的

Issue #189は、v18 developmentをterminalにしたstage directory生成境界をfresh v19 protocol / root / runtimeで
固定し、rank performanceをexactly once観測する。v18で実行済みのsource、test、fixtures、runnerとraw terminal
evidenceは変更せず、v19固有moduleを設定、stage initialization、worker identity、actual count auditへ限定する。

## Predecessor境界

実行基盤はv18 merge commit `5106e341522bd6cd9d79a7de48800c607eedc455`とaccepted v8 image
`ngr-cross-encoder-precision-v8:freeze` / image ID
`sha256:136ad9466799109bf32b4b96b611c9db9a099bcc47cf78243f26c7227bc16742`である。v18の実行source、test、
fixtures、runner、docs、raw terminal evidence、append-only count clarificationをhost側SHA-256 registryで固定する。
v18 terminal evidenceのsemantic content、packet、resultを観測入力として開かず、v18 terminal volumeをmount、
read、copy、reuseしない。

v10 runtime / cache-freeze、v11 root-freeze、v12 runtime、v13 commit-freeze、v14 runtime、v15
root-normalization-freeze、v16 source-root-propagation-freeze、v17 runtime、v18 runtimeの各volumeはmount、read、
copy、reuseしない。旧`/opt/ngr-v8/runtime`もcreate、mount、readしない。唯一のmutable rootは開始時absentを
確認してexclusive-createする`github-cross-encoder-precision-v19-runtime`で、全mutable pathを
`/opt/ngr-v19/runtime`配下へ置く。accepted image rebuildは0、networkは`none`、container内git / subprocess
invocationは0とする。

## Stage directory契約

developmentまたはholdoutのclaimを開く前に、専用stage initialization processで次の2 directoryがどちらも
存在しないことを確認し、exclusive-createする。

- `/opt/ngr-v19/runtime/databases/{stage}`
- `/opt/ngr-v19/runtime/runs/{stage}`

作成後はexact path、directory種別、create count 2を検証してからclaimとworkerを開始する。既存directory、
片側だけの作成、path divergenceはterminal errorとし、同じv19 protocolを再試行しない。developmentの全hard
gateがpassしてselected candidateがある場合だけholdoutへ同じ契約を適用する。

## Actual count契約

planned worker slot cardinalityとactual execution countを別fieldで保持する。terminal count auditはraw command log
から次を集計し、`stage_process_count`や`6 * claim_count`をactual countへ代入しない。

- `planned_worker_slot_count`: claim済みstageごとのfrozen 6 slots
- `actual_worker_launch_count`: raw logへ記録されたworker command数
- `actual_successful_worker_count`: return code 0のworker command数
- `actual_observed_result_count`: successful workerが生成したresult数
- `actual_finalize_count`: return code 0のfinalize command数
- `stage_directory_initialization_count`: return code 0のstage initialization command数

互換field `worker_process_count` / `observed_result_count`も、それぞれactual launch / actual observed resultを返す。
error、timeout、OOMをplanned slot完了へ補完せず、terminal evidence manifestへexact file-setとSHA-256を固定する。

## Preflight

`tools/run_cross_encoder_precision_v19_observation_wslc.ps1 preflight`は、prebuild implementation commitを
pushしたPRのCore / Optional MCP CIがgreenになった後だけ一度実行する。fresh runtime volume create、source
initialization、read-only Windows cacheからのmodel exact exclusive-copy、dependency report、synthetic probeは
各最大一度とする。shared Windows SQLiteは接続せず、FILE_SHARE-aware SHA-256の前後一致だけを確認する。

prebuild PR前はv19 dedicated / related targeted、Ruff、audits、full suiteを一度だけlocal実行する。preflight内は
dedicated / related targeted、Ruff、auditsだけを再検証し、full suiteは再実行しない。successful preflight
evidenceをcommit / pushし、そのremote CI green後だけdevelopment claimを開く。preflight errorはその場で
terminal化・hash固定し、同じv19 protocolを再試行しない。

### Preflight実測

prebuild implementation commit `0d17a767baaf7edd24acce0a9d07a3810fe2c3f6`に対するGitHub Actions run
`33344937227`はCore / Optional MCPともgreenだった。その後、2026-08-31T00:37:41Zにv19 preflightを
exactly once実行し、成功した。同一v19 preflightのretryは0である。

runtime volume createは1、accepted image rebuildは0、networkは`none`だった。model 2 revisionsのrequired
filesは12 files / 3,427,616,927 bytesで、read-only source cacheからのexclusive-copy後に全bytes一致した。
synthetic probe forwardは2、development claim、holdout claim、registered query、result、retryはいずれも0で
ある。shared Windows SQLiteはopenせず、前後SHA-256はともに
`84a3fc590eee990579e3ef8130294129934fe93e25f18ac249eece19813c261e`だった。v10からv18までの禁止volumeは
mount、read、reuseせず、旧`/opt/ngr-v8/runtime`もcreate、mount、readしていない。

preflight evidence manifest SHA-256は
`dc296667d9a1c8632574021e526f85a82598880d6f822c72be4780e9c7d3be86`である。performanceは引き続き
`not assessed`であり、developmentはこのevidence commitのremote CI greenまで閉じたままとする。

## Developmentとholdout

`tools/run_cross_encoder_precision_v19_observation_wslc.ps1 run`はpushed remote HEADとCI greenを検証してから
developmentをexactly once開く。baseline primary / replay、base primary / replay、v2-m3 primary / replayを
frozen順序の6 fresh worker process / 6 fresh observation SQLiteで実行する。developmentの全hard gateがpassし
selected candidateがある場合だけholdoutをexactly once開く。selectedなし、fail、error、timeout、OOMでは
holdoutを開かず、result / errorのいずれもretry 0とする。同一v19 protocolとterminal volumeは再実行・再利用
しない。

### Development実測

preflight evidence commit `af0cb59d3f4a5dc71debb2eaeee7826c676d9354`に対するGitHub Actions run
`33353194262`はCore / Optional MCPともgreenだった。その後、v19 developmentをexactly once実行した。
stage directory initializationは1、development claimは1、planned worker slotsは6、actual worker launch / successful
worker / observed resultは各6、finalizeは1だった。全commandのreturn codeは0で、shared Windows SQLiteはopenせず、
前後SHA-256はともに`84a3fc590eee990579e3ef8130294129934fe93e25f18ac249eece19813c261e`だった。
同一v19 developmentのretryは0である。

1,953 pairsに対する`BAAI/bge-reranker-base`のprimary / replay latencyは130,854.886 ms /
127,312.911 ms、peak RSSは1,387,855,872 bytes / 1,404,841,984 bytesだった。
`BAAI/bge-reranker-v2-m3`のprimary / replay latencyは430,410.446 ms / 423,496.594 ms、peak RSSは
2,377,826,304 bytes / 2,413,035,520 bytesだった。これはprotocol環境における実測値であり、production性能を
表すものではない。

4 candidateはいずれも全hard gateを満たさず、development statusは`failed`、selected candidateは`null`だった。
したがってholdout claim / holdout workerは0のまま開いていない。performanceはdevelopment範囲で`observed`だが、
GitHub RAG / NGR retrieval parityは成立しておらず、default surface変更も行わない。observation evidence manifest
SHA-256は`83044dcbcbb586440a2055bd8f80bb09b404bb8aa8b76cc47894e3609e8e527e`、count audit SHA-256は
`9a85dea0d16deb66adfb64795fc99b1f7f79b39efb1fc32b6fc2b4eb9cd43be3`、terminal evidence manifest SHA-256は
`953c856ec0160e56843e6bf5ab461a280fe7305e8b4501253c7da922c0c98907`である。

## 主張境界

観測前のperformanceは`not assessed`である。terminal evidence後も、性能、GitHub RAG / NGR retrieval parity、
physical integrationの判断は実測development / holdout evidenceの範囲だけに限定する。production performanceや
NGR default変更をこのprotocol単独から主張しない。evidence追加後のlocal検証は専用test、関連targeted、audits
だけとし、full保証はGitHub CIに置く。
