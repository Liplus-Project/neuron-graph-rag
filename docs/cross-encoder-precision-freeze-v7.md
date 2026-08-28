# Cross-encoder precision benchmark freeze v7

## 目的とresult-free境界

v7はv6の再試行ではない。v6で29 installed distributionsに対して26件だけを列挙したoffline attestationをsuccessor contractの前提に限定し、expected registry、`importlib.metadata` actual inventory、filesystem METADATA inventoryの三者完全一致を要求する。v1-v6のobserved packet、semantic evidence、model cache / weight、既存run root、共有Windows SQLiteをcorpus、query、gold、model、candidate、gate設計に使わない。predecessor artifactはmanifestのSHA-256 registryだけでbyte immutabilityを検証する。

freeze中のregistered query、model forward / inference、observed resultは`0/0/0`であり、performanceは`not assessed`である。production service、NGR default、SQLite schema、dependency default、MCP config、feedback / outcomeは変更しない。

## 不変のrank-only意味

24 corpus identity、development / holdout各8 bilingual query / gold、2 exact model revision、passage projection、batch size 8、NGR top24、model prefilter exact top20、4 rank-only candidate、CE / RRF式、top5、tie-break、selection rule、11 hard gateはv6と同一である。corpus source bytes / relationshipは`c32b3049fd3daaa2190faf5e3e85955a195ee88c`へ固定する。v7 evaluatorはv3 rank-only evaluatorをisolated moduleとして読み、identity、path、installed-distribution contractだけをbindする。

## Exact installed-distribution contract

`tests/fixtures/github_cross_encoder_precision_v7.expected-distributions.json`はPEP 503相当の`[-_.]+`を`-`へ置換してlowercase化したcanonical name、version、origin classを持つ。v6と同じ26 ML/runtime artifactsに`pip==24.0`、`setuptools==79.0.1`、`wheel==0.46.3`を加え、exact 29を固定する。canonical nameの重複、extra、missing、version mismatch、空Name / Versionを拒否する。

inside-image validatorは`importlib.metadata.distributions()`の全件を列挙し、Name、Version、canonical nameをcanonical attestationへ保存する。Linux amd64、CPython 3.11.15、CPU-only torch、CUDA / triton / nvidia distribution不在、network disabled、container filesystem exclusive-create、synthetic CPU float32 probe、query / inference / result=`0/0/0`も必須である。

`ngr.wslc-runtime-content/v2`はv1のnormalized file / symlink entries、base digest、dependency registry、exclusion registryに加え、全`site-packages/*.dist-info/METADATA`を直接parseする。filesystem inventoryはmetadata path、Name、Version、canonical name、METADATA content SHA-256を持つ。missing / malformed metadata、duplicate canonical name、duplicate / case-colliding / traversal path、normalized contentに対するextra / missing METADATA pathを拒否する。expected registry SHAはruntime reportとattestationの両方へbindする。

## WSLC one-shot lifecycle

baseは`python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3`、WSLCは`2.9.4.0`である。同じpinned Containerfile / inputを`--no-cache`でbuild A `ngr-cross-encoder-precision-v7:freeze`、build B `ngr-cross-encoder-precision-v7:rebuild-check`として各一回だけbuildする。各imageはhost bind mountなし、`--network none`のfresh containerでruntime content reportとattestationを各一回だけ取得する。

local image IDは各built artifactの識別だけに用いる。A/Bのnormalized content、filesystem inventory、actual attestation、expected registry照合が完全一致した場合だけbuild Aをaccepted imageにする。失敗時はevidenceを保存し、accepted imageを持たず、追加build、追加report、同v7のvalidator調整、successor observationを禁止する。成功時もsuccessor observationは本freezeのsquash merge commitだけをinputとする別Issueで行い、accepted build Aを再buildしない。

## Lifecycle isolation

development / holdoutはv7専用runtime、archive、transport、claim / result / error pathをexclusive-createする。phase verifierはunobserved、development archived pass / fail / error、holdout archivedを検証し、重複、部分archive、hash mismatch、claimなしresult、development gate不通過時のholdoutを拒否する。本freeze PRではdevelopment / holdoutを実行しない。

## Freeze record

static/unit、CLI audit/probe、JSON/compile/lint、full unittestを先にgreenへ収束させた後、WSLC one-shotを実行した。build A/Bは各1回とも成功した。続くbuild Aのruntime content reportは1回だけ実行され、normalized contentとfilesystem METADATA inventoryの不一致を検出してreport出力前に終了した。その時点で停止したため、runtime content report A/B=`1/0`、attestation report A/B=`0/0`、additional build/report=`0/0`である。

部分出力と例外は`tests/evidence/github_cross_encoder_precision_v7/`へ保存した。expected registryは29件だが、actual inventoryとfilesystem inventoryはreportが成立しなかったため件数も三者一致も未確立である。fingerprintとattestationも未確立であり、accepted imageはない。同v7のvalidator調整・追加build・追加report・successor observationを禁止し、failure outcomeへfail-closedした。performance=`not assessed`、query / inference / result=`0/0/0`を維持する。
