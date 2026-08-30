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

## Developmentとholdout

`tools/run_cross_encoder_precision_v18_observation_wslc.ps1 run`はpushed remote HEADとCI greenを検証してから
development claimをexactly once開く。baseline primary / replay、base primary / replay、v2-m3 primary /
replayをfrozen順序の6 fresh container process / 6 fresh observation SQLiteで実行する。developmentの全hard
gateがpassしselected candidateがある場合だけholdoutをexactly once開く。selectedなし、fail、error、timeout、
OOMではholdoutを開かず、result / errorのいずれもretry 0とする。同一v18 protocolとterminal volumeは再実行・
再利用しない。

## 主張境界

観測前のperformanceは`not assessed`である。terminal evidence後も、性能、GitHub RAG / NGR retrieval parity、
physical integrationの判断は実測development / holdout evidenceの範囲だけに限定する。production performanceや
NGR default変更をこのprotocol単独から主張しない。evidence追加後のlocal検証は専用test、関連targeted、audits
だけとし、full保証はGitHub CIに置く。
