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

prebuild implementation commit `ced0af7aa9cb5d3e5300ba8012372b7edafb34ad` の Core / Optional MCP
CI green を確認した後、preflight を一度だけ実行した。専用 v12 runtime volume の absence、accepted image、
v11 / v1-v10 immutable registry、shared DB の開始時 hash を検証し、volume を exclusive-create した。
frozen v10 verifier だけが 2 revisions / 12 required files / 3,427,616,927 bytes を model-cache へ
exclusive-copyし、v11 binder を通る synthetic probe は2 forward inferenceを完了した。preflight 前後の shared
DB SHA-256 はともに `84a3fc590eee990579e3ef8130294129934fe93e25f18ac249eece19813c261e`、
development / holdout claim=`0/0`、result=`0`、retry=`0` である。

preflight evidence commit `287c1bbbe6ed83e8677f787edd5ec678d55c65cd` の Core / Optional MCP CI
green を確認した後、development one-shot を exactly once 開始した。preflight evidence sync と claim 前の
shared DB hash 一致は成功したが、最初の development claim container で terminal error となった。v11 binder は
wrapper と frozen base harness を v12 root へ適用した一方、frozen evaluator の
`verify_protocol_commit` は `git` subprocess を要求し、accepted v8 image に `git` executable が存在しないため
`FileNotFoundError: [Errno 2] No such file or directory: 'git'` で fail-closed した。

claim file の exclusive-create 前に停止したため development / holdout claim=`0/0`、worker process=`0`、
observed-stage inference=`0`、result=`0`、retry=`0` である。holdout は開いていない。shared DB の観測後
SHA-256 も `84a3fc590eee990579e3ef8130294129934fe93e25f18ac249eece19813c261e` で、
preflight 前から不変である。accepted image rebuild、runtime report / attestation rerun、v10 / v11 volume の
mount / read / reuse、旧 v8 root の create / mount / read はすべて0である。

raw terminal error の SHA-256 は
`14b0d4ddeead9a155c26b80553ceccf6e86b460e06f1775c2a9f49564e0d6c64`、terminal evidence manifest の
SHA-256 は `cf84eb12d15b1fc906ac9947267b2d48bc3657feee9b5aa592b5b17f7fb3e954` である。
同じ v12 の preflight、claim、development、holdout は再実行しない。development とholdoutのrank性能は
`not assessed` であり、retrieval parity、物理統合可能性、NGR default変更を支持しない。次候補軸は、accepted
image内でgit executableを要求せずにfrozen commit identityを検証できるsuccessor protocol、またはgitを含む
新しい実行基盤をresult-freeに固定してから別protocolで観測する境界である。
