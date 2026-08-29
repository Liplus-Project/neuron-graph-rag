# Cross-encoder precision observation v10

## 実行境界

Issue #171 の one-shot observation は、successful result-free model-cache freeze v10 の
squash merge commit `e75d1e065441b794ce83b68f62d55747741052e5` と accepted v8 image
`ngr-cross-encoder-precision-v8:freeze` / image ID
`sha256:136ad9466799109bf32b4b96b611c9db9a099bcc47cf78243f26c7227bc16742`
だけを実行基盤とする。accepted image は再buildせず、保存済み runtime content report、
attestation、fingerprint、METADATA correspondence、exact 29 distribution registry を read-only で
照合する。

v10 cache-freeze volume `github-cross-encoder-precision-v10-cache-freeze` は mount、read、copy、
再利用しない。成功済み freeze の source、tests、fixture、tool、docs、evidence からなる 15 artifacts は
git 上の SHA-256 だけを検証する。v1-v9 evidence も frozen registry の byte immutability だけを検証し、
semantic content、raw packet、result、error、model cache、weights を観測入力にしない。

唯一の mutable run root は、開始時 absent を確認して exclusive-create する専用 volume
`github-cross-encoder-precision-v10-runtime` である。Windows host path は `Path`、container path は
`PurePosixPath` として分離し、全 container destination を v10 strict serializer に通す。source、
model cache、fresh SQLite、worker output、claim / result / error / archive / transport は
`/opt/ngr-v10/runtime` 配下の distinct absolute POSIX path へ置く。

## Preflight

`tools/run_cross_encoder_precision_v10_observation_wslc.ps1 preflight` は専用 runtime volume を一度だけ
作り、development claim 前に次を exclusive evidence として保存する。

- exact cache-freeze merge commit の export と v10 / v1-v9 immutable registry
- accepted image tag / ID、保存済み freeze reports、exact 29 inventory contract
- 2 exact model revision の Windows cache discovery、required-file verification、runtime volume copy 後の再hash
- model-cache を source initialization で作らず、frozen verifier だけが exclusive-create する ownership
- CPU / float32 / eval / inference-mode / batch8 の model 別 synthetic probe
- v10 observation tests、v10 cache-freeze audit、v8 evaluator audit / probe、full suite、変更対象 Ruff
- shared Windows DB の FILE_SHARE-aware SHA-256 前後一致
- development / holdout claim=`0/0`、registered query / observed-stage inference / result=`0/0/0`

共有 DB `C:\Users\smile\.ngrdb\knowledge.db` は SQLite 接続せず、FILE_SHARE_READ / WRITE / DELETE を
許可する hash reader だけを使う。hash を取得できない場合は DB に接触せず claim 前 error として停止する。
preflight evidence を commit / push し、その remote check-run がすべて green になった後だけ
development claim を作れる。preflight error も terminal evidence として保存し、同じ v10 を再試行しない。

## One-shot lifecycle

`tools/run_cross_encoder_precision_v10_observation_wslc.ps1 run` は pushed remote HEAD と CI green を
再検証し、development claim を exactly once 作る。baseline primary / replay、base primary / replay、
v2-m3 primary / replay を固定順の 6 fresh container process / 6 fresh SQLite DB で各一度だけ実行する。
worker は frozen v8 corpus、queries / gold、models、candidate order、prefilter exact top20、CE / RRF
rank-only、top5、tie-break を使う。raw logit の符号や絶対値で hit を除外せず、frozen v8 evaluator が
11 hard gate を一意に recompute して即 archive する。

development で `all_hard_gates_pass=true` かつ selected candidate がある場合だけ holdout claim を
exactly once 開く。selected なし、failed、error では holdout を開かない。claim 後の例外、native signal、
OOM、dependency / runtime / container failure は error archive し、timeout を retry へ変換しない。
success / fail / error のいずれでも retry count は 0 である。

terminal evidence は raw worker packet と git evidence の byte identity、container / process / DB identity、
primary / replay determinism、state immutability、shared DB の観測後 hash を含む。性能判定は frozen
development / holdout gate だけに基づき、環境成立だけで retrieval parity や物理統合可能性を主張しない。

## 観測結果

implementation commit `a7b8867056543a47a5a2f64ffd3977f92f59d2d3` から preflight を一度開始した。
runtime volume absence、WSLC version、accepted image identity、v10 cache-freeze / v1-v9 immutable registry、
shared DB の開始時 SHA-256 を検証し、専用 runtime volume を exclusive-create した。exact cache-freeze
commit と current harness source を distinct strict POSIX destination へ展開し、source initialization 後も
model-cache が absent であることを確認した。frozen v10 verifier による 2 model revisions / 12 required
files / 3,427,616,927 bytes の runtime model-cache exclusive copy は一度成功した。

続く synthetic probe の最初の container process で terminal error となった。v10 wrapper が frozen v8
observation module の surface globals を v10 runtime path へ更新した一方、underlying v5 harness の
`direct_git_bytes` は旧 `/opt/ngr-v8/runtime/frozen-source` を保持していた。probe は model import / load /
forward より前の protocol source validation で
`FileNotFoundError: /opt/ngr-v8/runtime/frozen-source/corpora/repository-native-controlled-v3/README.md`
として fail-closed した。

raw failure evidence `tests/evidence/github_cross_encoder_precision_v10_observation/preflight.error.json` の
SHA-256 は `88f8e0b71be42751ff5414a3c793d5f08e2c52d7c8689bc6e27e00cf20f4f038` である。runtime
volume create=`1`、model-copy verifier run=`1`、synthetic probe process=`1`、development / holdout
claim=`0/0`、registered query / preflight forward / observed-stage inference / result=`0/0/0/0`、retry=`0`
である。accepted image rebuild、runtime report rerun、attestation rerun、v10 cache-freeze volume の mount /
read / reuse はいずれも 0 である。

shared Windows DB の preflight 前と error 後の SHA-256 はともに
`84a3fc590eee990579e3ef8130294129934fe93e25f18ac249eece19813c261e` であり、SQLite では open して
いない。development と holdout は未観測、performance は `not assessed` である。同じ v10 の
preflight、runtime volume、model copy、probe、development を再試行しない。

次候補軸は、successor protocol を先に freeze し、v8 wrapper surface だけでなく frozen v5 harness の
source / frozen-source root まで新 runtime root へ一貫して bind してから synthetic probe を開く境界である。
v10 raw failure と terminal evidence は byte 不変の predecessor として保存する。
