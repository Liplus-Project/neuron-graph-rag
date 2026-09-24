# CUDA shortlist 公開 API の実機確認 v1

## 観測

2026-09-24、Windows build 26200、RTX 3080、PyTorch 2.4.1+cu121（CUDA 12.1）で `attach_cuda_shortlist_retriever()` と `engine.search_cuda_shortlist()` を使った。固定 v3 の93文書・3問を in-memory NGR engine に登録し、既存 E5 cache copy を使った。CUDA model を初回検索でロードして1問を warm-up した後、同じ backend instance で3問を測定した。CUDA model と入力 tensor は `cuda:0` に置かれ、PyTorch の CUDA peak allocation が記録された。

| 問い | pair | 公開 API 全体 wall | v2-m3 区間 | 期待 source 順位 | peak allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| sheepdog-three-axes | 100 | 1.773 秒 | 1.719 秒 | 1 | 2,571,252,736 byte |
| codex-hook-trust | 100 | 1.707 秒 | 1.672 秒 | 3 | 2,511,751,680 byte |
| source-wrap-levels | 99 | 1.694 秒 | 1.656 秒 | 1 | 2,500,856,832 byte |

全 source 順位は[探索的 #253](gpu-v2-m3-probe-v1.md)の CUDA 結果と3問とも一致した。cache fingerprint は `d168d7a473ed5bede19f882c297ce64076405326c7fbd84979e5ee045557daa4` で固定 v3 と一致し、pair 数は100・100・99だった。各検索の peak reserved は 2,589,982,720 byte。最大 process peak RSS は 3,798,282,240 byte。RSS は累積 peak であり、この検索だけの増分ではない。

探索的 #253 の v2-m3 区間は1.728・1.658・1.670秒で、今回の公開 API 区間との差はそれぞれ -0.009・+0.014・-0.014秒。公開 API 全体 wall は E5 query、shortlist、cache照合、CUDA再順位付けを含むため、v2-m3 区間より長い。短い差の原因はこの1 run から分離できない。#253 の CPU/GPU 比を公開 API の速度比とは呼ばない。モデルロード、E5 cache 構築、download は warm 測定に含めない。`retriever.close()` 後の PyTorch allocated は 8,519,680 byte で、CUDA context 等は残り得る。

## 再実行と範囲

`experiments/cuda_shortlist_api_probe.py` を使い、`--cache` に固定 v3 E5 cache の repository 外 copy、`--e5-snapshot` と `--reranker-snapshot` に repository 外の pinned snapshot、`--output` に repository 外の新規 JSON path を渡す。script は corpus と4 model file の SHA-256 / revision path を固定 manifest に照合し、cache fingerprint、forward pair 数、期待順位を検査する。[観測 JSON](cuda-shortlist-retrieval-v1.observed.json) の LF 正規化後の SHA-256 は `087368a1cd91edfbb05fd2fec4430071dc04bc7ce6c81a841abcf4b8a01d5fc6`。model files と CUDA 依存は repository に置かない。

この結果は当該 Windows host、固定 corpus、3問、単一 run、FP32 batch 8 に限る。一般の順位一致、速度、必要 VRAM を保証しない。`cuda_peak_allocated_bytes` は当該 PyTorch process の allocation であり、desktop と他 process の VRAM は含まない。
