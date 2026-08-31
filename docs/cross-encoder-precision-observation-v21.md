# Cross-encoder precision observation v21

## 目的

Issue #194は、v20で固定したintent-aware rank fusionとgate ownershipを、fresh development / holdout
identity、fresh protocol root、fresh runtime volumeで実測する。否定・除外意図、positive retrieval、relation pathを
同時に維持または改善できるcandidateの有無と、latency / peak RSSを観測する。

## 前提

v19 performance evidenceはmerge commit `2f5f5d7d658681a495bcaab05e8729b567db9fc7`、v20 result-free
contractはmerge commit `33b465c7422e8eeae1153e323a46a662a97f8fee`で固定済みである。v19 cases / resultsは
identity非再利用の監査対象にだけ使い、v21のquery、gold、result、candidate選択へ再利用しない。v20のproduction signal
schema、fusion weights、EN / JA marker contract、gate ownership、positive per-case non-regressionは変更しない。

## Protocolとidentity

protocol IDは`github-ngr-cross-encoder-precision-v21`、runtime volumeは
`github-cross-encoder-precision-v21-runtime`、container rootは`/opt/ngr-v21/runtime`である。development
identityは`github-ngr-cross-encoder-precision-v20-development-7b6b4f9d`、holdout identityは
`github-ngr-cross-encoder-precision-v20-holdout-c31e958a`を使う。corpus、query、goldはv21専用fixtureで分離し、
workerはquery fixtureとproduction corpusだけを読み、gold / forbidden / expected valueをranking入力へ渡さない。

## Gate順序

`protocol-source-contract-integrity`、`identity-separation`、`baseline-prefilter-validity`、
`relation-source-edge-only-provenance`、`production-signal-only`、`default-surface-immutability`を先に評価する。
一つでもfailした場合はcandidate gateを評価せず、developmentをterminal failureとしてholdoutを開かない。

protocol validityがすべてpassした場合だけ、v20の6 candidate-controllable gateを順番どおり評価する。positive caseは
baseline top 5にexpected sourceがある場合にcandidate rankが同順位以下であることを要求し、cohort平均でcase regressionを
隠さない。negative controlはcase non-worseningに加えてforbidden top 5 aggregateのstrict improvementを要求する。
全candidate gateを満たす最初のcandidateだけをselected candidateとし、その場合だけholdoutをexactly once開く。

relation pathはschemaと各field/valueのexact一致を検証する。JSON key orderやwhitespaceを含むbyte identityは主張しない。

## One-shot境界

implementation commitのremote CI green後だけpreflightをexactly once実行する。successful preflight evidence commitの
remote CI green後だけdevelopmentをexactly once実行する。error、timeout、OOM、protocol gate failure、candidate gate
failureをretryへ変換せず、同一v21 protocolとterminal volumeを再実行または再利用しない。holdoutはdevelopmentでselected
candidateが全candidate gateをpassした場合だけexactly once実行する。

accepted v8 imageはrebuildせず、networkはnoneとする。read-only Windows model cacheをfresh runtimeへexclusive-copyし、
shared Windows SQLiteはread-only hash以外で開かず、preflight前後・development前・terminal後のSHA-256一致を要求する。
planned worker slots、actual launches、successful workers、observed results、finalizeを別countとして記録する。

## Evidenceと主張境界

preflight / development / optional holdoutのclaim、raw worker output、observed result、transport、command log、count audit、
terminal manifestはappend-onlyで固定する。candidate別にdirect / semantic / relation MRR・hit@5、negative forbidden top 5、
failed hard gate IDs、latency、peak RSS、pair countを記録する。

実測主張はfrozen v21 evidenceに限定する。GitHub RAG parity、production performance、physical integration、NGR default変更へ
一般化しない。

