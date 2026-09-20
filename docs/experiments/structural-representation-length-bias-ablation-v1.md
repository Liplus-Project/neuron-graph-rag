# Structural representation / length-bias ablation v1

## result-free診断

有効なfull-corpus oracle v2 evidenceとexact 93文書corpusだけを読み、登録queryを再実行せずに診断した。baseではchunk数とbest scoreのPearson相関が0.5631、Spearman相関が0.4182、v2-m3ではそれぞれ0.5149、0.5742だった。chunk数とrankのSpearman相関は符号が逆で同じ絶対値となり、chunk数が多い文書ほどmaximum raw logitで上位になりやすい信号がある。

baseのtop distractorは96 chunksの `docs/2.-Evolution.md`、正解は3 chunksの `rules/model/axis-separation.md` だった。v2-m3のtop distractorは正解と同じ3 chunksの `rules/model/layer-definition.md` である。このためlength biasだけでは説明できず、構造表現とaggregationを独立させたfactorial比較が必要である。

診断evidenceは `tests/evidence/structural_representation_length_bias_ablation_v1/result_free_diagnostic.json` に固定し、入力hash、相関、正解とtop distractorのwinning chunk本文・offset・hash・構造fieldを保存する。登録query実行数、GitHub RAG request、共有DB accessはいずれも0である。

## 事前固定protocol

Issue #238本文と `tests/fixtures/structural_representation_length_bias_ablation_v1.manifest.json` に、4 armの2x2 factorial、NLME temperature 1.0、cutoff 20、成功条件、literal structural prefix、欠損・whitespace・fence・cap規則、body chunk同一性、tokenizer truncation診断を固定した。

registered executionはfresh holdout-absent WSLC volumeで行う。workerはgoldを持たず、全93文書へ同じquery非依存transformを適用する。preflightまではsynthetic forwardだけとし、登録query実行数0を維持する。

## 観測結果

fresh volume `github-structural-length-bias-ablation-v1-runtime` でpreflightを通過した後、source commit `f73373a2c8ca4bd728a7ea8fc46b14e9f6c4b020` に対して登録queryを1回だけ実行した。retryは0であり、workerにはgoldを渡していない。93文書、2 model、4 armを事前固定どおり評価し、host auditは `observed_valid` となった。

| model | body/max | structural/max | body/NLME | structural/NLME |
| --- | ---: | ---: | ---: | ---: |
| base rank | 64 | 41 | 43 | 21 |
| v2-m3 rank | 45 | 39 | 20 | 19 |
| base Spearman(chunk count, score) | 0.4182360021 | 0.3807203866 | 0.0764353592 | -0.0104637423 |
| v2-m3 Spearman(chunk count, score) | 0.5742361339 | 0.5637051727 | 0.3005357064 | 0.2630947786 |

primary successはfalseである。同じnonbaseline armで両modelがcutoff 20以内になる条件を満たさず、最良の `structural_nlme` でもbase 21、v2-m3 19だった。一方、`structural_max`、`body_nlme`、`structural_nlme` は両modelで `body_max` よりrankを改善したため、directional evidenceは成立した。NLMEによる相関絶対値の低下はbody表現とstructural表現の双方で両modelに観測され、length-bias attenuationは両方trueとなった。この結果は既定retrievalやproduction設定を変更しない。

## truncation診断

各model・各representationで2065 query-chunk pairをquery非依存に記録した。body/structuralとも512 tokenを超えたpairはbase、v2-m3のいずれも0であり、structural prefixによって新たにtruncationが増えたpairも0だった。

structural prefixのcodepoint長はmin 74、max 269、mean 144.3486682809である。prefix token長はbaseがmin 27、max 104、mean 52.1510895884、v2-m3がmin 26、max 103、mean 51.1510895884だった。したがって今回のrank差をprefix追加によるtokenizer truncationの増加では説明できない。

## evidence

append-only evidenceは次のSHA-256で固定する。

- preflight: `41355b6d2f17a3b046af34d079c137c3c6aca0bce392c315d88346ba4f97f01f`
- claim: `e14743bb9ceb3eda8c8b7911678fb12da4af9fe4652e7a770ff606d59ec8d5ba`
- observed result file: `da37bf495c6061a0f48119b6d679a7f42daaee6b625e41446ba9d361022c70e8`
- observed payload: `91bb1bff94fa52d8ebdd1f01e7ab085bda5314cf90ee79a3dc5c8099d5220fb8`

実行時間はbase 411.326秒（peak RSS 2,462,584,832 bytes）、v2-m3 1365.897秒（peak RSS 3,613,126,656 bytes）だった。
