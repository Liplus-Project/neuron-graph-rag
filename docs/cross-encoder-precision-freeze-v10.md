# Cross-encoder precision model-cache freeze v10

## 目的

v9 one-shot preflightは、source初期化が
`/opt/ngr-v9/runtime/model-cache`を先に作成し、frozen model-copy verifierが要求する
exclusive-create前提と衝突したため、development claim前にterminal errorとなった。v9は再試行せず、
raw failure、terminal summary、evidence manifestをbyte不変のpredecessorとして保存する。

v10は性能観測ではない。source初期化とmodel-cache ownershipを分離し、read-only Windows cacheから
freeze専用volumeへ2 exact model revisionのrequired filesを一度だけcopy / hash verificationできるかを
固定するresult-free protocolである。

## ownership境界

container pathは`PurePosixPath`で保持し、strict serializerを通したabsolute POSIX pathだけをWSLCへ
渡す。Windows source cacheだけをhost `Path`として扱い、`/input/models:ro`へmountする。

source初期化は次のdirectoryだけを作る。

- `source`
- `databases`
- `runs`
- `archive`
- `transport`

dedicated destination `/opt/ngr-v10/cache-freeze/model-cache`は作らず、source展開後にもabsentを確認する。
model-copy verifierはsource required filesをsizeとLFS SHA-256またはGit blob IDで先に検証し、その後に
target directoryをexclusive-createする。copy先全fileを再hashし、source / destination SHA-256一致を
同一verifier processで保存する。既存targetの削除、上書き、merge copyは行わない。

## frozen入力

exact predecessor merge commitは
`aefe0123d48b762445c1a58e5ae6056cc02feab0`である。current implementation commitと別々にarchiveし、
freeze volumeのdistinct strict POSIX destinationへ展開する。accepted imageは
`ngr-cross-encoder-precision-v8:freeze` / image ID
`sha256:136ad9466799109bf32b4b96b611c9db9a099bcc47cf78243f26c7227bc16742`
で、再buildしない。WSLCは`2.9.4.0`に固定する。

v9 raw failure SHA-256は
`cc3c57682dd25df86d8aa0122efee9ef081b18ae2d08f216e87672d2ffff4426`、terminal summaryは
`676480a3af07c09a0041623ed483d885884dea3d26a2c9f722e5d30a3ba0786e`、evidence manifestは
`75f581c1b07520c4c55cbcb9cd49805b2c29919d25fcb94204074038f8c292d6`である。v9 path-freezeと
observationの20-file closureをprebuild / post-freezeで照合し、semantic contentを観測入力にしない。

## result-free one-shot

`tools/run_cross_encoder_precision_v10_freeze_wslc.ps1 freeze`は、prebuild implementation commitが
push済みでremote CI green、かつ次の両volumeがabsentの場合だけ開始する。

- cache freeze: `github-cross-encoder-precision-v10-cache-freeze`
- future runtime: `github-cross-encoder-precision-v10-runtime`

cache-freeze volumeだけをexclusive-createし、accepted imageを`--network none`で使用する。v9 runtime /
path-freeze volumeはmount、read、copy、reuseしない。model-copy verifierは一度だけ実行し、success / error
のどちらでもretryは0、freeze volumeは再利用しない。failure evidenceを保存した場合も同じv10を再試行
しない。

model copy後もregistered query、NGR retrieval、SQLite、transformer model import / load、synthetic probe、
model forward、observed resultを実行しない。development / holdout claim=`0/0`、registered query / model
import / load / forward / observed result=`0/0/0/0/0`、performance=`not assessed`を維持する。shared Windows
DBはSQLite接続もhash読取も行わない。

## outcome

implementation commit `45704eefb6610501d100c18ebc024e820aa428dd`のCore CI / Optional MCP
adapterがgreenになった後、cache freezeをexactly once実行してpassした。cache-freeze volume create=`1`、
model-copy verifier run=`1`、retry=`0`である。source初期化直後までmodel-cache targetはabsentで、verifierが
exclusive-createした。

2 model revisions / 12 required files / 3,427,616,927 bytesをsourceで検証し、copy後の全file SHA-256一致を
確認した。model verification SHA-256は
`380ae8c602d1c0049adc495eaea97f31aca25984cbf6ed286594df35bf07e0c9`、pass summaryは
`cca6ee778acb18b7dd921b754657f92bda39af2ccdf6827a86f052634ab41910`、evidence manifestに登録した
count auditは`f55da694e51f8d4e7296cdb03e157452c78410eb40030d55165a30a4c908eaa7`である。

v9 predecessor 20 filesはprebuild / post-freezeでbyte identityが一致し、future runtime volumeはabsentである。
development / holdout claim=`0/0`、registered query / model import / load / forward / observed result=
`0/0/0/0/0`、performance=`not assessed`でterminalとする。cache-freeze volumeは後続観測に再利用しない。

このsuccessful result-free freezeだけが、別Issueでdevelopmentを一度開始する前提となる。retrieval parityや
物理統合可能性は性能観測まで支持しない。
