# Cross-encoder precision observation v23 real-task results

## Outcome

developmentは一度だけ実行され、protocol validityでfail closedになった。`relation-source-edge-only-provenance`が
failしたためcandidate gatesは評価せず、selected candidateは`null`、holdoutは未開封のまま終了した。同一protocolの
retryは0であり、この結果を再実行で置き換えない。

失敗箇所は`v23-real-dev-relation-home-migration`である。現行NGR full-query baselineはexpected source
`docs/decision-wiki-pilot-migration.md`をrank 3へ返したが、hitにはgoldで固定した
`docs/Home.md -> docs/decision-wiki-pilot-migration.md`の`informs` relation pathが付かなかった。もう一つの
relation-linked caseは期待したrelation pathを保持した。

## Provenance boundary

real GitHub由来なのは、固定commit `79b456d620f1b37746669ea1fe1e57c385f5e4ed`のGit blobsから取得した12 documentsの
path、text、content SHA-256である。4本の`informs` relation edgesとdevelopment / holdoutのquery / goldは、
このprotocolの評価構造としてfixture authorが固定したものである。GitHub上のhyperlinks、imports、commit historyから
relationを抽出または検証したものではない。

したがって今回のrelation gate failureは、real repository text上に固定したsynthetic evaluation edgeについてtraversal
provenanceを測った結果であり、実repositoryのdependency discovery性能を示さない。source-grounded relation acquisitionを
評価する場合は、既存v23 bytesを変更せず、取得provenanceを固定したsuccessor protocolとして別に設計する。

## Development metrics

以下はprotocol gateより前に生成されたdescriptive evidenceである。candidate gate pass、candidate selection、quality優越を
意味しない。latencyとpeak RSSは各armのprimary worker実測値である。

| Arm | Query / model | Direct MRR / hit@5 | Semantic MRR / hit@5 | Relation MRR / hit@5 | Forbidden top 5 | Latency ms | Peak RSS MiB | Pairs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | original full-query NGR default | 0.2500 / 0.5000 | 0.7500 / 1.0000 | 0.6667 / 1.0000 | 2 | 706.058 | 31.941 | 0 |
| B | positive-clause NGR ablation | 0.2500 / 0.5000 | 0.7500 / 1.0000 | 0.6667 / 1.0000 | 2 | 713.917 | 31.699 | 0 |
| C | base intent-aware | 0.3500 / 1.0000 | 0.6667 / 1.0000 | 0.6667 / 1.0000 | 0 | 42281.749 | 1134.281 | 720 |
| D | v2-m3 intent-aware | 0.5000 / 0.5000 | 1.0000 / 1.0000 | 0.6667 / 1.0000 | 0 | 128533.780 | 2076.340 | 720 |

C / DはAよりprimary latencyがそれぞれ約59.88倍 / 182.04倍、peak RSSが約35.51倍 / 65.00倍だった。
negative forbidden top 5 countはA / Bの2からC / Dの0へ減ったが、protocol validity failureにより
candidate nonregression / strict-improvement gatesは未評価である。

## Protocol and lifecycle audit

- protocol gates: source contract `pass`、identity separation `pass`、baseline prefilter `pass`、relation provenance
  `fail`、production-signal-only `pass`、default-surface immutability `pass`
- actual worker launch / success: `8 / 8`; finalize: `1`; observed cases: `8`
- development / holdout claims: `1 / 0`; stage initialization: `1`; retry: `0`
- accepted image rebuild: `0`; model cache copy: `1`; shared database open: `0`
- shared database SHA-256 before preflight / before claim / after observation:
  `84a3fc590eee990579e3ef8130294129934fe93e25f18ac249eece19813c261e`
- predecessor immutable artifacts: `44 / 44` unchanged
- corpus source commit: `79b456d620f1b37746669ea1fe1e57c385f5e4ed`; corpus fixture SHA-256:
  `6db9689017235968b191a3844f243c5b54996ee92aeab8b4c79ca08afb095ae5`
- development observed SHA-256: `81c54010876bb361c9dbf8f14192f021bcaa5ad0ec9ffc56d9e250b0e41d17a6`
- terminal evidence manifest SHA-256:
  `6487342f8b33e3643a9e65832d193c818f32ef15dc760dba4cb41766b874f01c`

この観測は固定12-file document corpusとfixture-authored evaluation structureのdevelopment 8 casesに限られる。holdout
evidence、NGR default変更、production性能、physical integration、実repository dependency discovery性能を主張しない。
