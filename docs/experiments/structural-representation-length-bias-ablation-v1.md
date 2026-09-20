# Structural representation / length-bias ablation v1

## result-free診断

有効なfull-corpus oracle v2 evidenceとexact 93文書corpusだけを読み、登録queryを再実行せずに診断した。baseではchunk数とbest scoreのPearson相関が0.5631、Spearman相関が0.4182、v2-m3ではそれぞれ0.5149、0.5742だった。chunk数とrankのSpearman相関は符号が逆で同じ絶対値となり、chunk数が多い文書ほどmaximum raw logitで上位になりやすい信号がある。

baseのtop distractorは96 chunksの `docs/2.-Evolution.md`、正解は3 chunksの `rules/model/axis-separation.md` だった。v2-m3のtop distractorは正解と同じ3 chunksの `rules/model/layer-definition.md` である。このためlength biasだけでは説明できず、構造表現とaggregationを独立させたfactorial比較が必要である。

診断evidenceは `tests/evidence/structural_representation_length_bias_ablation_v1/result_free_diagnostic.json` に固定し、入力hash、相関、正解とtop distractorのwinning chunk本文・offset・hash・構造fieldを保存する。登録query実行数、GitHub RAG request、共有DB accessはいずれも0である。

## 事前固定protocol

Issue #238本文と `tests/fixtures/structural_representation_length_bias_ablation_v1.manifest.json` に、4 armの2x2 factorial、NLME temperature 1.0、cutoff 20、成功条件、literal structural prefix、欠損・whitespace・fence・cap規則、body chunk同一性、tokenizer truncation診断を固定した。

registered executionはfresh holdout-absent WSLC volumeで行う。workerはgoldを持たず、全93文書へ同じquery非依存transformを適用する。preflightまではsynthetic forwardだけとし、登録query実行数0を維持する。

## 観測結果

登録one-shotはまだ実行していない。
