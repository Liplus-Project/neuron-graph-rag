# Cross-encoder precision observation v21 results

## 観測境界

prebuild commit `fcea358d1e9d0bac9d888bf0e5169bdc21eeac4a`のremote CI run `33374613368`がgreenに
なった後、preflightを1回だけ実行した。preflight evidence commit
`701ff5f0e9426f15f19aab54a9c75d5655665452`のremote CI run `33375147651`がgreenになった後、
developmentを1回だけ実行した。developmentはprotocol validity 6 gateをすべてpassし、2 candidateともcandidate
gate 6件をすべてpassした。順序上最初の`base-intent-aware`をselected candidateとし、holdoutを1回だけ開いた。
holdoutもprotocol validityと両candidateの全candidate gateをpassした。failed hard gate IDは両stage・両candidateとも
空である。

## 品質

品質値は`MRR / hit@5`で示す。

| stage | candidate | direct | semantic | relation | forbidden top 5 |
| --- | --- | ---: | ---: | ---: | ---: |
| development | current-ngr-prefilter | 1.0 / 1.0 | 0.75 / 1.0 | 1.0 / 1.0 | 2 |
| development | base-intent-aware | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1 |
| development | v2-m3-intent-aware | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0 |
| holdout | current-ngr-prefilter | 1.0 / 1.0 | 0.75 / 1.0 | 1.0 / 1.0 | 2 |
| holdout | base-intent-aware | 1.0 / 1.0 | 0.75 / 1.0 | 1.0 / 1.0 | 1 |
| holdout | v2-m3-intent-aware | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0 |

## 性能

性能値はprimary / replayの順で示す。baselineはcross-encoder pairを生成しないためpair countは0である。

| stage | candidate | latency ms | peak RSS MiB | pair count |
| --- | --- | ---: | ---: | ---: |
| development | current-ngr-prefilter | 608.22 / 556.15 | 31.94 / 31.93 | 0 / 0 |
| development | base-intent-aware | 10498.60 / 7386.81 | 911.83 / 913.52 | 240 / 240 |
| development | v2-m3-intent-aware | 22121.32 / 19115.43 | 1712.35 / 1712.14 | 240 / 240 |
| holdout | current-ngr-prefilter | 550.96 / 517.71 | 31.52 / 31.43 | 0 / 0 |
| holdout | base-intent-aware | 7367.12 / 7352.16 | 913.68 / 913.62 | 240 / 240 |
| holdout | v2-m3-intent-aware | 18876.03 / 18951.45 | 1712.51 / 1712.25 | 240 / 240 |

## 証跡監査

terminal countはplanned worker slots 12、actual launches 12、successful workers 12、observed results 12、
finalize 2、development claim 1、holdout claim 1、retry 0である。shared SQLite SHA-256はpreflight前、claim前、
terminal後とも`84a3fc590eee990579e3ef8130294129934fe93e25f18ac249eece19813c261e`で一致した。

主要証跡SHA-256は、development observed
`06acd4212045dfd94dc5f7cfef4276d122d85c5d14e9c930c1ecc0dbbae6da88`、holdout observed
`0c0310671d39d189830bce382adef14ce95e8c7f93ff3b0b09729a205b98ab21`、count audit
`63b3992a61f617313e841d2ef3e4eb2a91592a1467a0a73242895717a7475986`、observation manifest
`e0c7957fa7195341dad7c3e014bbcc1d4c8c26c9715551bfdf9b93eadfa03083`、terminal manifest
`0cbf5235dc312c67db5fa2f9d8f4116b8ff5d09141b085e7abf15fe6cb43693c`である。

## 主張境界

実測主張はfrozen v21 synthetic corpus、fresh development / holdout identity、CPU-only accepted imageに限定する。
両intent-aware candidateがこのfixture上で全gateをpassした事実を記録するが、GitHub RAG parity、production performance、
physical integration、NGR default変更へ一般化しない。
