# Cross-encoder precision observation v9

## 実行境界

Issue #167 の one-shot observation はpath-freeze squash merge commit
`25790b5218ccc7a5741dbdf6a19d1f7723d7afeb`と、accepted v8 image
`ngr-cross-encoder-precision-v8:freeze` / image ID
`sha256:136ad9466799109bf32b4b96b611c9db9a099bcc47cf78243f26c7227bc16742`
だけを実行基盤とする。accepted imageは再buildせず、保存済みruntime content report、
attestation、fingerprint、METADATA correspondence、exact 29 distribution registryをread-onlyで
照合する。

v9 path-freeze volume `github-cross-encoder-precision-v9-path-freeze`はmount、read、copy、再利用しない。
成功済みpath-smoke、count-audit、evidence-manifestを含む12 artifactsはgit上のSHA-256だけを
検証する。v1-v8 predecessor evidenceも29-file manifest registryでbyte immutabilityだけを検証し、
semantic content、raw packet、result、error、model cache、weightsを観測入力にしない。

唯一のmutable run rootは、開始時absentを確認してexclusive-createする専用volume
`github-cross-encoder-precision-v9-runtime`である。Windows host pathは`Path`、container pathは
`PurePosixPath`として分離し、全container destinationをv9 strict serializerに通す。
source、model cache、fresh SQLite、worker output、claim / result / error / archive / transportは
`/opt/ngr-v9/runtime`配下のdistinct absolute POSIX pathへ置く。

## Preflight

`tools/run_cross_encoder_precision_v9_observation_wslc.ps1 preflight`は専用runtime volumeを
一度だけ作り、development claim前に次を保存する。

- exact path-freeze merge commitのexportとv9 / v8 immutable registry
- accepted image tag / ID、保存済みfreeze reports、exact 29 inventory contract
- 2 exact model revisionのWindows cache discovery、required-file verification、volume copy後の再hash
- CPU / float32 / eval / inference-mode / batch8のmodel別synthetic probe
- v9 observation tests、v9 path audit、v8 evaluator audit / probe、full suite、変更対象Ruff
- shared Windows DBのFILE_SHARE-aware SHA-256前後一致
- development / holdout claim=`0/0`、registered query / observed-stage inference / result=`0/0/0`

共有DB `C:\Users\smile\.ngrdb\knowledge.db`はSQLite接続せず、hash readerだけを使う。preflight
evidenceをcommit / pushし、そのremote check-runがすべてgreenになった後だけdevelopment claimを
作れる。preflight errorもexclusive evidenceとして保存し、同じv9を再試行しない。

## One-shot lifecycle

`tools/run_cross_encoder_precision_v9_observation_wslc.ps1 run`はpushed remote HEADとCI greenを
再検証し、development claimをexactly once作る。baseline primary / replay、base primary / replay、
v2-m3 primary / replayを固定順の6 fresh container process / 6 fresh SQLite DBで各一度だけ実行する。
workerはfrozen v8 corpus、queries / gold、models、candidate order、prefilter exact top20、CE / RRF
rank-only、top5、tie-breakを使い、frozen v8 evaluatorが11 hard gateを一意にrecomputeして即archiveする。

developmentで`all_hard_gates_pass=true`かつselected candidateがある場合だけ、holdout claimを
exactly once開く。selectedなし、failed、errorではholdoutを開かない。claim後の例外、native signal、
OOM、dependency / runtime / container failureはerror archiveし、timeoutをretryへ変換しない。
success / fail / errorのいずれでもretry countは0である。

terminal evidenceはraw worker packetとgit evidenceのbyte identity、container / process / DB identity、
primary / replay determinism、state immutability、shared DBの観測後hashを含む。performance判定は
frozen development / holdout gateだけに基づき、環境成立だけでretrieval parityや物理統合可能性を
主張しない。

## 観測結果

未実行。preflight evidenceをcommit / pushし、remote CI greenを確認した後にone-shot結果をこの節へ
固定する。
