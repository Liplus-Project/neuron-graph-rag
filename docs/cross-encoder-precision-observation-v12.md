# Cross-encoder precision observation v12

## 実行境界

Issue #175 の one-shot observation は、successful parameterized root freeze v11 の
squash merge commit `39f2cebc6b3b43ac1060a2ce519e8906fa598f57` と accepted v8 image
`ngr-cross-encoder-precision-v8:freeze` / image ID
`sha256:136ad9466799109bf32b4b96b611c9db9a099bcc47cf78243f26c7227bc16742`
だけを実行基盤とする。accepted image は再buildせず、保存済み runtime content report、
attestation、fingerprint、METADATA correspondence、exact 29 distribution registry を read-only で
照合する。

v11 root-freeze volume `github-cross-encoder-precision-v11-root-freeze`、v10 runtime volume
`github-cross-encoder-precision-v10-runtime`、v10 cache-freeze volume
`github-cross-encoder-precision-v10-cache-freeze` は mount、read、copy、再利用しない。v11 の source、
tests、fixture、tool、docs、evidence からなる 14 artifacts と、その registry が閉じる v1-v10
predecessor は git 上の SHA-256 だけを検証する。predecessor evidence の semantic content、raw packet、
result、error、model cache、weights は観測入力にしない。

唯一の mutable run root は、開始時 absent を確認して exclusive-create する専用 volume
`github-cross-encoder-precision-v12-runtime` である。source、frozen protocol source、model cache、
fresh SQLite、worker output、claim / result / error / archive / transport は
`/opt/ngr-v12/runtime` 配下の distinct absolute POSIX path へ置く。旧
`/opt/ngr-v8/runtime` は create、mount、read しない。

## Parameterized harness binding

v12 は v11 の `bind_frozen_harness_root` を actual dependency report、synthetic model probe、claim、
worker、finalize、fail-stage の各 container process で実行する。v8 wrapper surface だけでなく distinct
frozen v5 `_BASE`、evaluation wrapper / base、`direct_git_bytes` の root を、同じ v12 root、source、
model cache、frozen protocol source、evidence path へ bind する。binding は v12 command の scoped
lifecycle に限定し、NGR の既定 retrieval path と predecessor module の host-side default は変更しない。

## Preflight

`tools/run_cross_encoder_precision_v12_observation_wslc.ps1 preflight` は、実装 commit を push した PR の
Core / Optional MCP CI が green になった後だけ実行する。専用 runtime volume を一度だけ作り、
development claim 前に次を exclusive evidence として保存する。

- exact v11 merge commit の export と v11 / v1-v10 immutable registry
- accepted image tag / ID、保存済み freeze reports、exact 29 inventory contract
- source initialization 後も model-cache が absent であること
- frozen v10 verifier だけによる 2 revisions / 12 required files / 3,427,616,927 bytes の exclusive copy
- v11 parameterized binding を経由する CPU / float32 / eval / inference-mode / batch8 synthetic probe
- v8 evaluator audit / probe、v10 / v11 audit、v12 targeted tests、full suite、変更対象 Ruff
- shared Windows DB の FILE_SHARE-aware SHA-256 前後一致
- development / holdout claim=`0/0`、registered query / observed-stage inference / result=`0/0/0`

共有 DB `C:\Users\smile\.ngrdb\knowledge.db` は SQLite 接続せず、FILE_SHARE_READ / WRITE / DELETE を
許可する hash reader だけを使う。hash を取得できない場合は DB に接触せず claim 前 error として停止する。
preflight evidence を commit / push し、その remote check-run がすべて green になった後だけ
development claim を作れる。preflight error も terminal evidence として固定し、同じ v12 を再試行しない。

## One-shot lifecycle

`tools/run_cross_encoder_precision_v12_observation_wslc.ps1 run` は pushed remote HEAD と CI green を
再検証し、development claim を exactly once 作る。baseline primary / replay、base primary / replay、
v2-m3 primary / replay を固定順の 6 fresh container process / 6 fresh SQLite DB で各一度だけ実行する。
worker は frozen v8 corpus、queries / gold、models、candidate order、prefilter exact top20、CE / RRF
rank-only、top5、tie-break を使う。raw logit の符号や絶対値で hit を除外せず、frozen v8 evaluator が
11 hard gate を一意に recompute して即 archive する。

development で `all_hard_gates_pass=true` かつ selected candidate がある場合だけ holdout claim を
exactly once 開く。selected なし、failed、error では holdout を開かない。claim 後の例外、native signal、
OOM、dependency / runtime / container failure は error archive し、timeout を retry へ変換しない。
success / fail / error のいずれでも retry count は 0 である。

## 観測結果

実環境 one-shot 前である。prebuild implementation、tests、docs を commit / push し、PR の Core / Optional
MCP CI green を確認するまでは preflight を開始しない。実測後は terminal evidence の範囲だけをここへ追記し、
環境成立だけで retrieval parity や物理統合可能性を主張しない。
