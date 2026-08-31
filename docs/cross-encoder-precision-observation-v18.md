# Cross-encoder precision observation v18

## 目的

Issue #187は、v17をpreflight terminalにしたself-audit module identityとpost-terminal audit scopeを修正し、
fresh v18 protocol / root / runtimeでrank performanceをexactly once観測する。v16 source-root propagationと
v17 rank-observation lifecycleを再利用し、v18固有moduleは設定、literal module identity、scoped terminal
audit、v18 worker identityへ限定する。

## Predecessor境界

実行基盤はv17 merge commit `7a4b63d65c5abc84e7550856a965572837b238b0`とaccepted v8 image
`ngr-cross-encoder-precision-v8:freeze` / image ID
`sha256:136ad9466799109bf32b4b96b611c9db9a099bcc47cf78243f26c7227bc16742`である。
v17 source、test、fixtures、runner、docs、terminal evidenceの13 artifactsをhost側SHA-256 registryで固定する。
v17 terminal evidenceのsemantic content、packet、resultを観測入力として開かず、実行済みv17 bytesを変更しない。

v10 runtime / cache-freeze、v11 root-freeze、v12 runtime、v13 commit-freeze、v14 runtime、v15
root-normalization-freeze、v16 source-root-propagation-freeze、v17 runtimeの各volumeはmount、read、copy、reuse
しない。旧`/opt/ngr-v8/runtime`もcreate、mount、readしない。唯一のmutable rootは開始時absentを確認して
exclusive-createする`github-cross-encoder-precision-v18-runtime`で、全mutable pathを
`/opt/ngr-v18/runtime`配下へ置く。accepted image rebuildは0、networkは`none`、container内git / subprocess
invocationは0とする。

## Literal module identity

preflightのself-audit commandはruntimeの`__name__`を参照せず、literal
`neuron_graph_rag.cross_encoder_precision_v18_performance_observation`を使う。v17で発生した
`python -m __main__ audit`をcommand生成時に拒否する専用testを置く。

actual dependency / probe / claim / worker / finalize processはv16共通source-root propagation verifierを
distinct 6 surfacesへbindする。claim source `/opt/ngr-v18/runtime/source`をexact確認した後、verification時だけ
`/opt/ngr-v18/runtime/frozen-source`へ正規化し、23 protocol artifactsと24 corpus documentsのexact bytesを
git / subprocessなしで検証する。worker container identityは`ngr-v18-*`、SQLiteとoutputはv18 rootだけを使う。

## Terminal audit

terminal auditはv14 / v10 predecessor scopeの内側へv18 protocol identityを最終bindし、wrapper、nested
lifecycle、manifest verifierのprotocol / freeze / root / evidence identityを一致させる。preflight errorまたは
development result / errorのいずれでもcount auditをexclusive-createし、既存stage evidenceを含むexact file-set
とSHA-256を`terminal-evidence-manifest.json`へappend-onlyで固定する。既存terminal evidenceがある場合は再固定を
拒否する。

## Preflight

`tools/run_cross_encoder_precision_v18_observation_wslc.ps1 preflight`は、prebuild implementation commitを
pushしたPRのCore / Optional MCP CIがgreenになった後だけ一度実行する。fresh runtime volume create、source
initialization、read-only Windows cacheからのmodel exact exclusive-copy、dependency report、synthetic probeは
各最大一度とする。shared Windows SQLiteは接続せず、FILE_SHARE-aware SHA-256の前後一致だけを確認する。

prebuild PR前はv18 dedicated / related targeted、Ruff、audits、full suiteを一度だけlocal実行する。preflight内は
dedicated / related targeted、Ruff、auditsだけを再検証し、full suiteは再実行しない。successful preflight
evidenceをcommit / pushし、そのremote CI green後だけdevelopment claimを開く。preflight errorはその場で
terminal化・hash固定し、同じv18 protocolを再試行しない。

### Preflight実測

prebuild implementation commit `51605f4efb64b45f3aaae5f04602890b701a1d58`に対するGitHub Actions run
`33342936425`はCore / Optional MCPともgreenだった。その後、2026-08-30T23:56:06Zにv18 preflightを
exactly once実行し、成功した。同一v18 preflightのretryは0である。

runtime volume createは1、accepted image rebuildは0、networkは`none`だった。model 2 revisionsのrequired
filesは12 files / 3,427,616,927 bytesで、read-only source cacheからのexclusive-copy後に全bytes一致した。
synthetic probe forwardは2、development claim、holdout claim、registered query、result、retryはいずれも0で
ある。shared Windows SQLiteはopenせず、前後SHA-256はともに
`84a3fc590eee990579e3ef8130294129934fe93e25f18ac249eece19813c261e`だった。v10からv17までの禁止volumeは
mount、read、reuseせず、旧`/opt/ngr-v8/runtime`もcreate、mount、readしていない。self-auditはliteral
`neuron_graph_rag.cross_encoder_precision_v18_performance_observation`で実行され、`__main__`は使われなかった。

preflight evidence manifest SHA-256は
`f28fe36099564db2d578c1bc65dd9afcc15e7c58d8a32cdea41fce8165fcd2aa`である。performanceは引き続き
`not assessed`であり、developmentはこのevidence commitのremote CI greenまで閉じたままとする。

## Developmentとholdout

`tools/run_cross_encoder_precision_v18_observation_wslc.ps1 run`はpushed remote HEADとCI greenを検証してから
development claimをexactly once開く。baseline primary / replay、base primary / replay、v2-m3 primary /
replayをfrozen順序の6 fresh container process / 6 fresh observation SQLiteで実行する。developmentの全hard
gateがpassしselected candidateがある場合だけholdoutをexactly once開く。selectedなし、fail、error、timeout、
OOMではholdoutを開かず、result / errorのいずれもretry 0とする。同一v18 protocolとterminal volumeは再実行・
再利用しない。

## Terminal実測

preflight evidence commit `43944ff56ea6dfb9b8e36cb8d7c2fcfa187bd24c`に対するGitHub Actions run
`33343200534`がCore / Optional MCPともgreenになった後、v18 developmentをexactly once実行した。development
claimは1だったが、最初の`ngr-v18-development-baseline-primary` workerが
`sqlite3.OperationalError: unable to open database file`でterminal errorになった。

失敗したdatabase pathは
`/opt/ngr-v18/runtime/databases/development/baseline-primary.sqlite3`で、未作成だった親directoryは
`/opt/ngr-v18/runtime/databases/development`である。source initializationは
`/opt/ngr-v18/runtime/databases`までを作成した一方、stage hostは`development` child directoryを作成せずに
SQLite pathをworkerへ渡した。実行済みv18 source、test、fixture、runner bytesは変更せず、このroot causeと
terminal evidenceだけを固定する。

holdout claim、registered query、retryは0で、shared Windows SQLiteはopenせず前後SHA-256も不変だった。
execution evidence上で実行されたdevelopment worker commandは最初の1件だけで、performance result / finalizeは
生成されていない。immutable count auditの`worker_process_count` / `observed_result_count`はfrozen worker slotsを
各6と記録しているため、これを実測済みperformance result数とは解釈しない。terminal statusとperformanceは
それぞれ`error` / `not assessed`である。

count audit、observation evidence manifest、terminal evidence manifestのSHA-256は順に
`1b0ad113f6ca1e404eb87e06b29ab22fd9bedfc6131577036fa78c16b571b224`、
`dad351e5591607d1b3c5d28616bfa4e63aedbd40ef206969d7374417f67c8149`、
`ec2c90506d0bec979f64fe33cd052db5726ce4baffb15d356815a9fc0e89816f`である。predecessor artifacts 13件は
byte不変で、同一v18 protocolのretryは0、terminal runtimeは再利用不可である。

## 主張境界

観測前のperformanceは`not assessed`である。terminal evidence後も、性能、GitHub RAG / NGR retrieval parity、
physical integrationの判断は実測development / holdout evidenceの範囲だけに限定する。production performanceや
NGR default変更をこのprotocol単独から主張しない。evidence追加後のlocal検証は専用test、関連targeted、audits
だけとし、full保証はGitHub CIに置く。
