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

implementation commit `165b57fa7c04a0b8cfa1ad41586317a5c113b3f4`からpreflightを
一度開始した。accepted imageのread-only inspectと専用volumeのexclusive create、exact freeze
commitのarchiveまでは成功したが、archiveをcontainerへ展開する最初の`wslc run`で停止した。

Windows host上でPOSIX container root `/opt/ngr-v8/runtime`が
`\\opt\\ngr-v8\\runtime`へ文字列化され、named volume指定が
`github-cross-encoder-precision-v8-runtime:\\opt\\ngr-v8\\runtime`となった。WSLCはcontainer
pathが`/`から始まるabsolute pathではないとして`E_INVALIDARG`を返し、preflightはclaim前に
fail-closedした。raw failure evidence
`tests/evidence/github_cross_encoder_precision_v8/preflight.error.json`のSHA-256は
`df97b812b052cc421408cdab3b89cbe25529e3167bdc4903c68c892f3c451280`である。

development / holdout claim=`0/0`、registered query=`0`、preflight / observed-stage model
inference=`0/0`、result=`0`である。accepted image rebuild、runtime report rerun、attestation
rerunも`0/0/0`である。専用volumeは1回作成されたが、同じv8のpreflightまたはone-shotへ
再利用しない。shared Windows DBはSQLiteでopenせず、preflight開始時SHA-256
`84a3fc590eee990579e3ef8130294129934fe93e25f18ac249eece19813c261e`だけを記録した。
post-error hashは保存packetに含まれないため、観測時点の前後不変性を追加主張しない。

developmentとholdoutは未観測、性能は`not assessed`である。同version retry、volume再利用、
result生成、tuning、再評価を行わず停止する。後続観測が必要な場合は、このfailure evidenceを
不変に保ち、container path serializationを修正したsuccessor result-free protocolを先に固定する。
