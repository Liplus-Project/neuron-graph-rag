# Cross-encoder precision observation v17

## 目的

Issue #185は、v16で固定したexact source-root propagationと6 surfaceのgit-free byte verificationを、
fresh rank observation lifecycleへ組み込む。frozen benchmark inputs、hard gate、expected distribution、
candidate順序、top-kは変更しない。v17固有moduleは設定とscoped compositionだけを持ち、preflight、
development、holdout、terminal evidenceの共通制御は`rank_observation_lifecycle.py`へ置く。

## 凍結境界

実行基盤はv16 merge commit `041233ab6267e883fdf9d519609bbe615c79645b`とaccepted v8 image
`ngr-cross-encoder-precision-v8:freeze` / image ID
`sha256:136ad9466799109bf32b4b96b611c9db9a099bcc47cf78243f26c7227bc16742`である。
v16で追加・固定したsource、test、fixtures、tool、docs、terminal evidenceの16 artifactsをhost側の
SHA-256 registryでbyte検証するが、v16 terminal evidenceのsemantic content、packet、resultは観測入力として
開かない。accepted image rebuildは0、networkは`none`、container内git / subprocess invocationは0とする。

v10 runtime / cache-freeze、v11 root-freeze、v12 runtime、v13 commit-freeze、v14 runtime、v15
root-normalization-freeze、v16 source-root-propagation-freezeの各volumeはmount、read、copy、reuseしない。
旧`/opt/ngr-v8/runtime`もcreate、mount、readしない。唯一のmutable rootは開始時absentを確認して
exclusive-createする`github-cross-encoder-precision-v17-runtime`で、source、frozen-source、model-cache、
databases、runs、archive、transportを`/opt/ngr-v17/runtime`配下のdistinct absolute POSIX pathへ置く。

## Source-root-safe composition

各container processはv16共通`SourceRootFreezeSpec.bind_verifier`をactual v8 wrapper object graphへ適用する。
wrapper、distinct `_BASE`、evaluation wrapper / base、nested protocol evaluator / baseの6 surfaceを同じ
verifierへbindする。claimがprotocolを`/opt/ngr-v17/runtime/source`からloadしたことを明示検証した後、
verification時だけrootを`/opt/ngr-v17/runtime/frozen-source`へ正規化し、23 protocol artifactsと24 corpus
documentsのexact bytesをgit-freeで検証する。NGR defaultと凍結済みv14 / v16 moduleのdefaultは変更しない。

## Preflight

`tools/run_cross_encoder_precision_v17_observation_wslc.ps1 preflight`は、prebuild implementation commitを
pushしたPRのCore / Optional MCP CIがgreenになった後だけ一度実行する。fresh runtime volume create、source
initialization、model-cache exact exclusive-copy、dependency report、synthetic probeは各最大一度とする。
model cache入力はread-only Windows cacheだけで、2 revisions / 12 required files / 3,427,616,927 bytesの
post-copy hashを検証する。shared Windows SQLiteは接続せず、FILE_SHARE-aware SHA-256の前後一致だけを確認する。

prebuild PR前のlocal verificationはv17専用test、関連targeted、Ruff、auditsに加えてfull suiteを一度だけ行う。
preflight内はv17専用test、関連targeted、Ruff、auditsだけを再検証し、full suiteは再実行しない。successful
preflight evidenceをcommit / pushし、そのremote CIがgreenになった後だけdevelopment claimを開く。preflight
errorはterminal evidenceとして固定し、同じv17 protocolを再試行しない。

## Developmentとholdout

`tools/run_cross_encoder_precision_v17_observation_wslc.ps1 run`は、pushed remote HEADとremote CI greenを
再検証してからdevelopment claimをexactly once開く。baseline primary / replay、base primary / replay、
v2-m3 primary / replayをfrozen順序の6 fresh container process / 6 fresh observation SQLiteで実行する。
developmentの全hard gateがpassしselected candidateがある場合だけholdoutをexactly once開く。selectedなし、
fail、error、timeout、OOMではholdoutを開かず、success / fail / errorのいずれもretryは0とする。同一v17
protocolとterminal volumeを再実行・再利用しない。

## 証跡と主張境界

preflight前のperformanceは`not assessed`である。terminal evidence、count audit、manifest、shared DB invariance、
v16 predecessor byte identityをSHA-256付きで固定する。evidence追加後はv17専用test、関連targeted、auditsだけを
local実行し、full保証はGitHub CIに置く。共通 / core codeを変更した場合だけlocal full suiteを再実行する。

性能、GitHub RAG / NGR retrieval parity、physical integrationについての判断は、実際に得たdevelopment / holdout
evidenceの範囲だけに限定する。このprotocol単独からproduction performanceやNGR default変更を主張しない。

## Preflight実測

prebuild implementation commit `f92c658aaa5dff29d7532b5da603623e85a16893`のCore / Optional MCP CI
run `33320500289`がgreenになった後、v17 preflightをexactly once実行した。fresh runtime volumeは開始時absentで
exclusive-createされ、source initialization、read-only Windows cacheからのmodel exact copy、synthetic probe
2 forward、dependency report、Ruff、101 targeted tests、v8 / v10 / v13 / v16 auditまでは成功した。

最後のv17 self-audit commandは、moduleが`python -m`でentryされた際の`__name__`をcommandへ使用したため
`python -m __main__ audit`となり、`ValueError: __main__.__spec__ is None`でterminal errorになった。これは
performance observation前のhost-side staged verification failureである。development / holdout claim=`0/0`、
registered query=`0`、observed result=`0`、retry=`0`で、performanceは`not assessed`、同一v17 protocolの
再試行とterminal volumeの再利用は禁止する。literal module-name修正は別のfresh successor protocolへ送る。

shared DB SHA-256はpreflight前後とも
`84a3fc590eee990579e3ef8130294129934fe93e25f18ac249eece19813c261e`で不変だった。raw preflight error
SHA-256は`3602cadcdb4b24f131d0ea8090e9d9ec3cd8292ac9ecc992625d9ab822af1bd3`、preflight terminal
SHA-256は`b88249b8680dc590d3519c2e4d7af55e3a788caf55918e75e4616f25b8fb4ed9`、stage manifest
SHA-256は`b44018a94aacbee8998c4d54e2bc1670be985dbe1f0de60b432c29d323675af4`である。count audit
SHA-256は`aec92f0248113675b4b1370183bfa0d778b1b1502a29e404894411435edcdbf2`、これら4 filesを固定する
terminal evidence manifest SHA-256は
`37e4e0b2ddc4896436a76a5632db1ed9cccf0974d3f1c70ce4027fd73b354191`である。accepted image rebuild、
旧volume mount / read / reuse、旧v8 root create / mount / readはいずれも0である。
