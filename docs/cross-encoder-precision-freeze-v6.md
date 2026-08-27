# Cross-encoder precision benchmark freeze failure v6

## 目的とresult-free境界

v5 one-shot observationはWSLC `--no-cache` rebuildに成功したが、freeze時とrebuild時のlocal image IDが異なったため、volume作成前にfail-closed停止した。v6はv5へ追加調整や再試行を行わず、独立buildの同値判定だけをlocal image IDからnormalized runtime content fingerprintとoffline runtime attestationへ切り替える。

本freeze中の登録query、model forward / inference、observed resultは`0/0/0`である。v1-v5 evidenceのsemantic content、raw packet、model cache / weight、既存venv / run root、共有Windows SQLiteはopenしていない。predecessorはmanifestのSHA-256 registryだけでbyte immutabilityを検証する。build後にoffline attestationがexact installed distribution setを証明しないことが判明したため、v6は`fail_closed_offline_attestation_not_exact`で停止し、performanceは`not assessed`である。

## 不変のrank-only意味

v6の24 corpus identity、development / holdout各8件のbilingual query / gold、2 exact model revision、passage projection、batch size 8、NGR top24、model prefilter exact top20、4 rank-only candidate順、CE / RRF式、top5、tie-break、selection rule、11 hard gateはv5と同一である。corpus source bytesとrelationshipは`c32b3049fd3daaa2190faf5e3e85955a195ee88c`へ固定する。

v6 evaluatorはv3の凍結rank-only evaluatorをisolated moduleとして読み、v6 identity/pathとimage同値契約だけをbindする。v5/v6 semantic diff testはこの境界外の差を拒否する。全raw logitが負、一様shift、positive / negative / mixed、5件未満、0件、同点、prefilter、CE / RRF、tie-break、returned path、11 derived gateのround-tripとtamper rejectionは同じ基底testで再現する。

## WSLC one-shot build

共通inputは次である。

- WSLC `2.9.4.0`
- base `python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3`
- base local image ID `sha256:f0c05afecbd16040caff4c000954567c7e3b56fc6c1f783fa10a55cba3ccfbfc`
- exact 26 Linux wheel、PyPI route、PyTorch CPU direct URL/hashはv5と同一

同じpinned Containerfile/inputを`--no-cache`で各1回だけbuildした。

- build A: tag `ngr-cross-encoder-precision-v6:freeze`、return code `0`、local image ID `sha256:03134b1593ee804ca3d03c90aee2cc40d64e4e81c87282ac2626ec83ba33e222`
- build B: tag `ngr-cross-encoder-precision-v6:rebuild-check`、return code `0`、local image ID `sha256:1d259121e8184d9342ae71a9c483904fe875e19f0991ebbffd03a97ede88b6ec`

local image IDはbuilt artifactの識別子として保存するが、独立build間の内容同値判定には使用しない。Aは当初successor observation用candidateとして記録したが、後述のattestation failureによりaccepted imageは存在しない。build A/Bは各1回で封印し、追加buildしない。

## Normalized runtime content fingerprint v1

algorithmは`ngr.wslc-runtime-content/v1`である。Python executable、`/usr/local/lib/python3.11/site-packages`、v6 requirements lock、dependency report、validator、fingerprint toolだけを対象とし、relative POSIX pathのUTF-8 byte順へ正規化する。regular fileはcontent SHA-256とsize、symlinkはtargetとsizeを持つ。`__pycache__`、`.pyc`、cache、temporary file、log、mutable run rootをliteral exclusion registryで除外する。

directory/file timestamp、ctime、inode、UID/GID、container/layer ID、build timestamp、tar順、host pathは入力にしない。content / symlink target差、missing / extra / duplicate path、path traversal、case collisionを拒否する。canonical JSONはkey sort、compact separator、UTF-8、newlineなしである。

build A/Bは各17,435 entries、fingerprint `238fda19d59c723d4b7f0535c5fd55e94fec3ced5707be2f128fe6a677dcd975`で完全一致した。canonical report bytesのSHA-256も両方`c4a0310df4c23700a76a3abd29aa1d5b5403b94f4e296845664517dceacae7ba`である。この一致はcontent fingerprint比較の成功だけを表し、offline attestation contractの成功を意味しない。

## Offline runtime attestation

build A/Bをhost bind mountなし、`--network none`のfresh containerで各1回検証した。validatorは期待26 distributionのversion、Linux amd64 / CPython 3.11.15、`torch==2.4.1+cpu`、`torch.version.cuda is None`、CUDA / triton / nvidia distribution不在、CPU float32 synthetic tensor、outbound network disabled、container filesystem exclusive-create、query / inference / result=`0/0/0`を報告した。

canonical attestation bytesのSHA-256はA/Bとも`43dd98cefc6b40ebeeb9cb01ab1374e1f845e94b8f4622acf09f684e5b2d47fe`で一致した。しかし保存済みruntime content reportの`site-packages/*.dist-info/METADATA`は29件であり、attestationが列挙しない`pip-24.0.dist-info`、`setuptools-79.0.1.dist-info`、`wheel-0.46.3.dist-info`を含む。したがってattestationはexact installed distribution setではなくallowlist subsetの投影であり、A/B一致でも必須gateを満たさない。

この発見後はaccepted imageを追加runせず、validator修正、追加build、追加report取得を行っていない。実測とfailure理由は`tests/evidence/github_cross_encoder_precision_v6/freeze-attestation.error.json`へ保存する。

## Lifecycle

manifestはdevelopment / holdoutそれぞれにv6専用runtime、archive、transport pathを持つ。phase verifierはsynthetic unobserved、development archived pass / fail / error、holdout archivedを同じ検証器へ通し、重複、部分archive、hash不一致、claimなしresult、development gate不通過時のholdoutを拒否する。

v6はsuccessful freezeではなく、accepted imageを持たないfail-closed記録である。successor observationは許可せず、development / holdoutを開始しない。v6専用runtime / archive / transport pathは未観測のまま保持し、既存packetを移送・変換・再利用しない。
