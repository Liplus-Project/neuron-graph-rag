# Cross-encoder precision observation v8

## 実行境界

Issue #163 の one-shot observation は freeze merge commit
`d2fdf7720e2a9dde7e8d666cf4fd9f314fd3d12f`と、そのcommitでacceptedとなったbuild A
`ngr-cross-encoder-precision-v8:freeze`だけを入力とする。accepted imageは再buildせず、
保存済みruntime content report、attestation、fingerprint、METADATA correspondence、
exact 29 distribution registryをread-onlyで検証する。v1-v7 evidenceはmanifest registryで
byte immutabilityだけを検証し、semantic contentやpacketを入力にしない。

mutable run rootは専用volume `github-cross-encoder-precision-v8-runtime`だけである。
source、model cache、fresh SQLite、worker output、claim/result/error/archive/transportを
volume内のdistinct pathへ置く。共有Windows DBはSQLite接続せず、FILE_SHARE-aware byte
readerでSHA-256だけを取得する。frozen sourceと検証前のWindows model sourceはread-only、
git evidenceは専用outputだけをbindする。

## Preflight

`tools/run_cross_encoder_precision_v8_wslc.ps1 preflight`はtarget volumeがabsentであることを
確認してexclusive-createし、次をclaim前に記録する。

- exact freeze commit、v8 hash registry、v1-v7 predecessor evidenceのbyte immutability
- accepted image tag / IDと保存済みfreeze report hash registry（追加build/reportは0回）
- exact 26 dependency artifactとexact 29 installed distribution contract
- 2 exact model revisionのWindows source検証、volume copy、全required file再hash
- CPU / float32 / eval / inference-mode / batch8のmodel別synthetic forward
- v8 test、audit/probe、full suite、変更対象Ruff
- development claim/query/observed inference/result=`0/0/0/0`

preflight evidenceをcommit/pushし、そのheadのremote check-runがすべてgreenになった後だけ
development claimを作る。preflight failureはexclusive error evidenceとして保存し、同versionを
再試行しない。

## One-shot lifecycle

`tools/run_cross_encoder_precision_v8_wslc.ps1 run`はremote CIを再検証してからdevelopment
claimを一度だけ作る。frozen順序のbaseline/base/v2-m3 primary+replayを6 fresh container
process / 6 fresh SQLite DBで実行し、raw packetをv8 rank-only evaluatorで一意に再計算して
即archiveする。11 hard gateをすべて通る最初のcandidateがある場合だけholdoutを同じ方法で
一度開く。selectedなし、failed、errorではholdoutを開かない。

claim後の例外、native signal、OOM、dependency/runtime/container failureはerror archiveし、
timeoutをretryへ変換しない。pass/fail/errorのいずれでもraw packetとgit evidenceのbyte
identity、fresh container/process/DB identity、primary/replay determinism、state不変、shared
DB SHA不変をterminal evidenceへ残す。

## 観測結果

Preflightおよびone-shot結果は未記録である。観測後、この節を保存evidenceに基づいて更新する。
