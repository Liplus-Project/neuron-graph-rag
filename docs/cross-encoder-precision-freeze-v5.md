# Cross-encoder precision benchmark freeze v5

## 目的と境界

v4 one-shot observation はclaim前のLinux dependency setupで停止し、rank-only性能は`not assessed`のままである。v5はv4へ追加調整や再試行を行わず、successor observationの起動mediumとLinux CPU dependency再現契約だけをWindowsのWSL built-in container (`wslc`)へ切り替える。

本freezeで実装済みなのはprotocol、WSLC image、dependency artifact registry、synthetic validation、evaluator、lifecycle verifier、tamper testである。登録query、model forward / inference、development / holdout resultの生成は未実装ではなく意図的に実行対象外であり、v5 freeze merge commitを入力とする別Issueだけが実行できる。

freeze auditは次を固定する。

- 登録query実行=`0`
- model inference / forward=`0`
- observed result=`0`
- v1-v4 evidence semantic content、model cache / weight、既存venv、既存v4 run root、共有Windows SQLiteをopenしない
- predecessor artifactはmanifestのSHA-256だけでbyte immutabilityを検証する

query / inference / resultのcount scopeはv5 freezeだけであり、過去versionの観測を含めない。container側のcountはfreeze期間全体の試行回数ではなく、最終固定contractに採用したimage build=`1`とoffline synthetic validation=`1`だけを表す。Containerfileやvalidatorを最終化する前のiterative build / validationはこのaccepted countから明示的に除外する。

## 不変のrank-only意味

v5のcorpus、development / holdout各8件のbilingual query / gold、2 exact model revision、480/80 passage projection、batch size 8、NGR top24、model prefilter exact top20、4 candidate順、CE / RRF式、top5、tie-break、selection rule、11 hard gateはv4と同一である。corpus sourceは`c32b3049fd3daaa2190faf5e3e85955a195ee88c`へ固定する。

v5 evaluatorはv3の凍結rank-only evaluatorをisolated moduleとして読み、v5 identityとartifact surfaceだけをbindする。v4/v5 semantic diff testはprotocol identity / path、container platform、dependency source contract以外の差を拒否する。負の全raw logit、一様shift、positive / negative / mixed、5件未満、0件、同点、prefilter、CE / RRF、tie-break、returned path、11 derived gateのround-tripとtamper rejectionは同じ基底testで再現する。

## WSLC container contract

- WSLC: `2.9.4.0`
- base: `python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3`
- platform: Linux amd64 / CPython 3.11.15
- resolver: uv `0.12.3 (507230998)` / `x86_64-pc-windows-msvc`
- installer: base imageのpip `24.0`、`--only-binary=:all:`、`--require-hashes`、`--ignore-installed`
- PyPI route: `https://pypi.org/simple`をdefault indexに一つだけ固定
- torch route: `https://download.pytorch.org/whl/cpu/torch-2.4.1%2Bcpu-cp311-cp311-linux_x86_64.whl`へのdirect URLだけを許可
- torch SHA-256: `2b03e20f37557d211d14e3fb3f71709325336402db132a1e0dd8b47392185baf`
- built image: `ngr-cross-encoder-precision-v5:freeze` / `sha256:bc105cebf12e144ef0e178b18b3ff95367bf7567113fdfe524c6c7c2de2b4dd2`

exact 26 Linux wheelのfilename、version、URL、SHA-256は`tests/fixtures/github_cross_encoder_precision_v5.dependency-artifacts.json`へ固定する。torch以外は選択済み`files.pythonhosted.org` artifactだけ、torchは上記PyTorch CPU direct URLだけを許可する。extra index、index fallback、source distribution、CUDA / triton / nvidia distributionを拒否する。

build commandは次のexact arrayをplatform registryへ保存する。

```text
wslc build --no-cache --file containers/github_cross_encoder_precision_v5/Containerfile --tag ngr-cross-encoder-precision-v5:freeze .
```

## Offline synthetic validation

validationはhost bind mountなし、networkなしのfresh containerで実行する。

```text
wslc run --rm --network none ngr-cross-encoder-precision-v5:freeze
```

validatorが確認するのはPython / OS / architecture、全26 dependency version、`torch.version.cuda is None`、小さなCPU float32 tensor加算、CUDA系distribution不在、outbound network failure、container filesystem上のexclusive-createだけである。model repository、cache、weight、登録query、model forward、SQLiteには触れない。

runtime、fresh SQLite、worker output、claim / result / error archive、transportはfuture observationでv5専用WSLC volume `github-cross-encoder-precision-v5-runtime`配下にexclusive-createする。Windows共有DBや既存v4 pathは接続対象にしない。frozen sourceを渡す必要がある場合だけread-only bindを許可し、runtime outputはbind mountへ書かない。

## Lifecycle

manifestはdevelopment / holdoutそれぞれにv5専用runtime、archive、transport pathを持つ。phase verifierはsynthetic unobserved、development archived pass / fail / error、holdout archivedを同じ検証器へ通し、重複、部分archive、hash不一致、claimなしresult、development gate不通過時のholdoutを拒否する。

successor observationはv5 freezeのsquash merge commitだけをprotocol inputにできる。preflight evidenceをcommit / pushした後にdevelopmentをexactly once実行し、全gate通過時だけholdoutをexactly once開く。claim後例外はerror archiveし、同versionを再試行しない。container buildまたはoffline validationを再現できない場合はerror evidenceを保存して停止し、performanceを`not assessed`のまま維持する。
