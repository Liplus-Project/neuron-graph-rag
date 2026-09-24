# RTX 3080 上の v2-m3 CPU/CUDA 探索的観測 v1

## 何ができるようになったか

固定 v3 の3問について、同じ Windows ホスト・同じ候補 pair を CPU と RTX 3080 の CUDA FP32 へ投入し、再順位付け時間・source 順位・GPU peak memory を比較できた。NGR の既定検索、wheel の依存関係、凍結済み v3 artifact は変更していない。再実行手順は [実験 README](../../experiments/gpu-v2-m3-probe.md)、全 source 順位・logit・環境・hash を含む観測値は [JSON](gpu-v2-m3-probe-v1.observed.json) に残した（LF 正規化後の SHA-256 `748a542eb9db21b1e80ab79443c6a817f6d696ce5045ac2934eb64e25dc9636b`）。

## 実測条件

- 2026-09-24、Windows build 26200（Python の `platform.platform()` 表記は `Windows-10-10.0.26200-SP0`）、Python 3.11.15、NVIDIA GeForce RTX 3080 10,240 MiB、driver 617.14、PyTorch 2.4.1+cu121 / CUDA 12.1。
- `transformers 4.44.2`、`tokenizers 0.19.1`、`huggingface_hub 0.36.2`、`onnxruntime 1.23.2`、`numpy 2.4.6`。CPU 4 thread、CPU/GPU とも FP32、`max_length=512`、batch size 8。CUDA 計時は synchronize を含む wall clock。
- v3 の93文書と3問、E5 shortlist K=50、文書内 top2 chunk、pair 数100・100・99。E5 cache copy の fingerprint と pair 数を v3 の記録に照合した。corpus、manifest、E5 ONNX/tokenizer、v2-m3 weights/tokenizer の SHA-256 は v3 fixture と一致した。モデル revision も一致した。
- 測定直前 `nvidia-smi` は 10,240 MiB 中 1,373 MiB 使用を表示した。GPU 推論には `torch.cuda.is_available()` と RTX 3080 の device name を確認し、CUDA model・tensor を使用した。

| 問い | pair | CPU warm | GPU warm | 速度比 | 期待 source 順位 CPU/GPU | 全 source 順位 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sheepdog-three-axes | 100 | 47.163 秒 | 1.728 秒 | 27.29 倍 | 1 / 1 | 一致 |
| codex-hook-trust | 100 | 46.507 秒 | 1.658 秒 | 28.04 倍 | 3 / 3 | 一致 |
| source-wrap-levels | 99 | 47.033 秒 | 1.670 秒 | 28.16 倍 | 1 / 1 | 一致 |

Model load は CPU 0.82 秒、GPU 1.29 秒。最初の全 pair cold 推論は CPU 48.44 秒、GPU 1.83 秒で、上表の warm 推論には含めない。CPU/GPU の対応 logit の最大絶対差は各問 `1.91e-5`、`9.54e-6`、`1.81e-5`。全 source 順位の一致はこの3問に対する観測であり、一般の入力での順位同一性を保証しない。

GPU の PyTorch peak allocated は 2,571,252,736 byte（約 2.39 GiB）、peak reserved は 2,589,982,720 byte（約 2.41 GiB）。これはプロセスによる CUDA allocation であり、デスクトップ等による GPU 使用量は含めない。CPU 区間の process peak RSS は 2,313,900,032 byte、GPU 区間終了時の process peak RSS は 3,485,962,240 byte。ただし RSS は累積 peak なので、区間ごとの増分や同時常駐量を示さない。

## 解釈と範囲

このホスト・単一実行・固定3問では、約 47 秒を占める v2-m3 CPU 再順位付けを約 1.7 秒に短縮できた。比較対象は同一 run の CPU 区間であり、過去の v3 約39〜43秒を速度比の分母にしていない。CPU 区間を先に測ったため、順序、OS cache、GPU の他プロセス、熱状態の影響は分離できない。推論部分のみの速度比であり、E5 cache 構築、検索全体、起動やモデル download、通常配布の費用を含まない。GPU 使用には約2.4 GiB の追加 CUDA allocation が必要だった。既定検索への GPU 導入や wheel への CUDA 依存同梱は、この観測からは決めない。
