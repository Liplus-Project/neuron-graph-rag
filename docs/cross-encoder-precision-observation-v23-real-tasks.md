# Cross-encoder precision observation v23 real tasks

## 目的

Issue #197は、provenanceを固定したreal GitHub checkoutのretrieval tasks上で、同一protocolの四つのarmを比較する。

- A: original / full queryをそのまま渡す現行NGR default。candidate quality gateの実baselineである。
- B: positive clauseだけをNGRへ渡すprefilter ablation。現行defaultとは呼ばない。
- C: base cross-encoderを使うintent-aware fusion。
- D: v2-m3 cross-encoderを使うintent-aware fusion。

shared intent-aware observation engineへarm identity、query mode、model identity、gate、weights、selection policyを注入する。
protocol非依存のshared runtimeがworker / finalizer adapterとfailure / transport / evidence plumbingを組み立てる。v23 version
moduleはprotocol constants、fixture freshness contract、verification commands、runtime config、公開dispatch aliasだけを持つ。
stage initialization、actual count、terminal auditは既存rank observation lifecycle / stage contractを再利用する。

## Corpus acquisition provenance

corpusは`Liplus-Project/neuron-graph-rag`のcommit
`79b456d620f1b37746669ea1fe1e57c385f5e4ed`にある12ファイルを、
`git show <commit>:<path>`でread-only取得したexact UTF-8 contentである。fixtureはrepository URL、full commit、全path、
path別content SHA-256、取得method、generator pathを持つ。generatorの`--verify`は固定Git objectから再取得したcanonical
fixture bytesとの一致だけを検証し、working treeの同名fileをsourceとして使わない。

development / holdoutは結果を見る前に各8 casesへ分離した。各stageはdirect lexical、semantic paraphrase、relation linked、
negative controlを2件ずつ持つ。normalized query similarityをv19 / v21の全caseと比較し、exact reuseまたは0.72以上を
copy / 言い換え / 近接変形としてfail closedにする。

## Gateとselection

protocol validityをquality比較より先に確定し、一つでもfailならcandidate gateを評価しない。production signal、fusion
weights、protocol / candidate gate ID、positive per-case non-regressionはv20 contractのまま変更しない。workerはcorpusと
queryだけを読み、goldはfinalizerだけが読む。

candidate selection policyは`lowest-development-primary-latency`として実行前にliteral freezeする。全candidate hard gateを
passしたdevelopment candidateのうちprimary worker latencyが最小のものを選び、同値時はcandidate ID lexical順で決める。
この順序はquality優越を意味しない。selected candidateが全gateをpassした場合だけholdoutを一度開く。

## One-shotとisolation

fresh protocol IDは`github-ngr-cross-encoder-precision-v23-real-tasks`、fresh runtime volumeは
`github-cross-encoder-precision-v23-real-tasks-runtime`、fresh rootは`/opt/ngr-v23-real/runtime`である。accepted v8 imageの
rebuild countは0、networkはnone、model cacheはread-only host cacheからfresh volumeへexclusive-copyする。各workerはfresh
SQLiteを使い、shared Windows SQLiteはhash確認以外で開かない。

implementation commitのremote CI green後にpreflightをexactly once実行し、そのevidence commitのremote CI green後に
developmentをexactly once実行する。developmentがselected candidateを持つ場合だけholdoutをexactly once実行する。
error、gate failure、timeout、OOMはretry 0のterminal outcomeであり、同一protocol / volumeを再実行しない。

## Evidenceと解釈境界

claim、raw packet、observed result、transport、actual count、command log、hash manifestはappend-onlyで保存する。A / B / C / Dを
分けてquality、latency、peak RSS、pair countを報告する。実測結果はこの12-file real corpusと16 queriesの範囲に限定し、
v21 parity、NGR default rollout、production performance、physical integrationを自動的に主張しない。
