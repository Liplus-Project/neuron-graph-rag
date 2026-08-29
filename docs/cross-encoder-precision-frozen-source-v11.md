# Cross-encoder precision frozen-source root v11

この文書はv11 successorの分割requirements specであり、目的、前提、制約、受入境界の正本である。
中央`docs/requirements.md`はfrozen v8 protocolのbyte registryに含まれるため変更しない。

## 目的

v10 one-shot preflightはmodel-cache copy後の最初のsynthetic probeでterminal errorとなった。frozen v8
wrapperのsurface globalsは`/opt/ngr-v10/runtime`へ更新されていたが、別module objectとしてloadされた
underlying v5 harness `_BASE`の`direct_git_bytes`は旧
`/opt/ngr-v8/runtime/frozen-source`を保持していた。v10 runtime / cache-freeze volumeは再利用せず、raw
failure、terminal summary、evidence manifest、実装をbyte不変のpredecessorとして保存する。

v11は性能観測ではない。wrapperと`_BASE`の両方へ新rootを明示bindし、new frozen-sourceからexact
protocol / corpus bytesを検証できるかだけを固定するresult-free one-shot protocolである。

## parameterized root境界

container pathは`PurePosixPath`からstrict absolute POSIX serializerを通す。専用volume
`github-cross-encoder-precision-v11-root-freeze`は
`/opt/ngr-v11/root-freeze`だけへmountし、次のpathを明示的に分離する。

- harness source: `/opt/ngr-v11/root-freeze/source`
- exact predecessor protocol source: `/opt/ngr-v11/root-freeze/frozen-source`
- model cache sentinel: `/opt/ngr-v11/root-freeze/model-cache`
- verifier report: `/opt/ngr-v11/root-freeze/root-binding-verification.json`

binderはfrozen v8 wrapperとdistinct `_BASE`へ`VOLUME`、`CONTAINER_ROOT`、`CONTAINER_SOURCE`、
`CONTAINER_CACHE`、`CONTAINER_PROTOCOL_SOURCE`、`ROOT`、`EVIDENCE`を同じ値で設定する。その後
`_BASE._bind_container_harness()`を呼び、nested evaluator rootもnew protocol sourceへ固定する。
`direct_git_bytes`はparameterized `CONTAINER_ROOT / frozen-source`だけを読む。

旧`/opt/ngr-v8/runtime/frozen-source`はabsence probe以外に使用せず、directory create、mount、content readを
行わない。v10 runtime volumeとcache-freeze volumeもinspect、mount、read、copy、reuseしない。

## exact byte verifier

exact predecessor merge commitは`6a511dbb3289dc83c6a7a7a1ec16593a0110b539`である。このarchiveをnew
frozen-sourceへ展開し、prebuild implementation commitのarchiveをdistinct sourceへ展開する。accepted
imageは`ngr-cross-encoder-precision-v8:freeze` / image ID
`sha256:136ad9466799109bf32b4b96b611c9db9a099bcc47cf78243f26c7227bc16742`で、再buildせず
`--network none`で使う。WSLCは`2.9.4.0`に固定する。

root-binding verifierはnew protocol sourceからfrozen v8 protocolをloadし、23 exact protocol artifactsと
24 corpus Markdown documentsを検証する。corpus bytesはrebound evaluator `_git_bytes`を通してnew
frozen-sourceから読み、登録SHA-256と一致させる。model-cacheはverifier前後ともabsentで、model-cache
copy、model import / load / forward、registered query、development / holdout claim、observed resultを一切
行わない。shared Windows SQLiteはopenもhash readもしない。

## result-free one-shot

`tools/run_cross_encoder_precision_v11_freeze_wslc.ps1 freeze`は、prebuild commitがpush済みでCore CI /
Optional MCPともgreen、かつroot-freeze volumeとfuture runtime volume
`github-cross-encoder-precision-v11-runtime`がabsentの場合だけ開始する。root-freeze volume createと
root-binding verifier runは各exactly once、retryは0である。future runtime volumeはfreeze前後ともabsentを
維持する。

success / errorのどちらでもroot-freeze volumeはterminalで再利用しない。failure evidenceを保存した場合も
同じv11を再試行しない。development / holdout claim、registered query、model-cache copy、model import /
load / forward、observed result、shared database openはすべて0、performanceは`not assessed`である。

このfreezeのpassはparameterized root bindingとexact offline source-byte validationだけを支持する。retrieval
performance、retrieval parity、物理統合可能性、NGR default変更は支持しない。後続developmentは別Issueで
exactly once実行する。

## 観測結果

prebuild implementation commit `085b66133e8acc68930273d1d8f306494f04af43`をpushし、GitHub Actions
run `33262111375`のCore CI / Optional MCPがともにgreenであることを確認してから、v11 freezeをexactly
once実行した。結果はpassである。同じprotocolのretryは行わず、専用root-freeze volumeはterminalかつ
non-reusableとして保持する。future runtime volumeは実行前後ともabsentである。

root-freeze volume createは1、root-binding verifier runは1、retryとverifier retryは0だった。wrapperと
distinct `_BASE`のbindingは一致し、新frozen-sourceから23 protocol artifacts / 6,555,670 bytesと24 corpus
documents / 151,585 bytesを登録SHA-256どおり検証した。v10 predecessor 23 artifactsは実行前後でbyte
不変だった。旧rootのcreate / mount / read、v10 runtime / cache-freeze volume mount、accepted image rebuild、
model-cache copy、model import / load / forward、registered query、development / holdout claim、observed result、
shared SQLite openはすべて0である。

主要evidence SHA-256は次のとおりである。

- `root-binding-verification.json`: `af35fa36a1e1be2ed1ef22790dbcc7a3943d351fad892c18e852c947566c8a89`
- `root-freeze.pass.json`: `ee86431e6603dd1eabd557778366ff05183a843c79d890f0d97fb9bcc9b26387`
- `count-audit.json`: `806c0f0f2c72d57e6e1d0a755086cdb14a8a8812d9e3dc3e0ab723b87f2ddc10`
- `evidence-manifest.json`: `522e223e57a8855371e18623952404156c593853ba32b91e93039152b2befae6`

このpassのperformanceは`not assessed`であり、retrieval performance、retrieval parity、物理統合可能性を
主張しない。後続のmodel-cache development / queryはこのterminal volumeを再利用せず、別Issueの新protocolで
実施する。
