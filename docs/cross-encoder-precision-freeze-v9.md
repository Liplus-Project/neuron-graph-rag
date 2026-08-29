# Cross-encoder precision path freeze v9

## 目的

v8 one-shot observationはclaim作成前にterminal failureとなった。Windows hostで
`pathlib.Path("/opt/ngr-v8/runtime")`を文字列化したため、container destinationが
`\\opt\\ngr-v8\\runtime`へ変わり、WSLCが`E_INVALIDARG`でfail-closedした。v8は再試行せず、
raw failure evidenceのSHA-256
`df97b812b052cc421408cdab3b89cbe25529e3167bdc4903c68c892f3c451280`を不変に保つ。

v9はこのtransport境界だけを固定するresult-free successorである。corpus、queries / gold、
models、candidate順、prefilter exact top20、CE / RRF rank-only、top5、tie-break、11 hard gateを
変更せず、性能を観測しない。

## path型とserializer

Windows host filesystem pathは`pathlib.Path`、container filesystem pathは
`pathlib.PurePosixPath`または検証済みPOSIX stringとして分離する。
`serialize_container_path`は次をfail-closedで拒否する。

- leading `/`を持たないpath、空path、`.` / `..` traversal、重複separator
- backslash、drive prefix、UNC、NUL、colonを含むpath
- container pathとして渡されたhost `Path`

named volumeは`named_volume_spec`だけが
`<volume>:<absolute POSIX container path>[:mode]`を生成する。v8 failure commandはnegative
fixtureとして保持し、v9 path-freeze mountは
`github-cross-encoder-precision-v9-path-freeze:/opt/ngr-v9/path-freeze`に固定する。

## prebuild contract

`tools/run_cross_encoder_precision_v9_freeze_wslc.ps1 prebuild`はv8のsource、tests、fixtures、
docs、tool、evidenceからなる29-file hash registryを検証する。v8 accepted image tag / ID、保存済み
runtime report / attestation hash、v8 raw failure evidenceをbyte identityとしてだけ扱い、semantic
content、query、model、共有DBを入力にしない。

targeted tests、full suite、Ruff、JSON UTF-8、diff checkを通したprebuild commitをpushし、その
remote check-runがすべてgreenになった後だけ`smoke`を一度実行できる。working treeとremote HEADが
一致しない場合、またはremote CIが未完了の場合は実行前に停止する。

## exactly-once path smoke

開始時にfreeze専用volume `github-cross-encoder-precision-v9-path-freeze`と将来の観測専用volume
`github-cross-encoder-precision-v9-runtime`の両方がabsentであることを確認する。前者だけを
exclusive-createし、accepted v8 imageを次の境界で一度だけ起動する。

- `--network none`
- host bind mountなし
- model cacheなし
- query、NGR retrieval、SQLite、model import / load / forwardなし

exact mount destination内にsentinel directory / fileを作成して読み戻し、container path、
`/proc/self/mountinfo`上のmount identity、sentinel hash、command / return code、stdout / stderr hashを
evidenceへ保存する。success / errorのどちらでもretryは0で、path-freeze volumeを後続観測に
再利用しない。smoke後もfuture runtime volumeはabsentでなければならない。

## outcome

prebuild commit `9c8f649bb2f5e5ce759634e82f4ec32b66df69ab`のremote CIが2 checksともgreenに
なった後、path smokeを一度だけ実行してpassした。run command SHA-256は
`9517832cd15138ae8c895668c0d10691f56f528ce8486c369b98e9ce7c54bc28`、return codeは`0`、
stdout / stderr SHA-256は
`ee08f9d372df9cc4d131d6a23fd98d4f23bac012c3edacdd81968d9ecf70bee4` /
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`である。

containerは`/opt/ngr-v9/path-freeze`を認識し、mount identity
`91|8:32|/volumes/github-cross-encoder-precision-v9-path-freeze/_data|/opt/ngr-v9/path-freeze|ext4`
を返した。sentinelはexact destinationで作成・読取され、そのSHA-256は
`f3fad15f0c91aa25cfbe4ba32c647584045aadf425305f7c2c10536006c65ce2`である。

path smoke run=`1`、retry=`0`でterminalとし、path-freeze volumeを再利用しない。future runtime
volumeはsmoke後もabsentである。registered query / model import / load / forward / observed
result=`0/0/0/0/0`、development / holdout claim=`0/0`、performance=`not assessed`を維持した。
v8 raw failure evidenceは同じSHA-256でbyte不変である。成功したresult-free transport freezeだけを
後続v9 observationの必要条件とし、このPRはretrieval parityまたは性能を支持しない。
