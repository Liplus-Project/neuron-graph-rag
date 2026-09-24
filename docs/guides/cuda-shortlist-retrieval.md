# CUDA shortlist 検索を試す

NVIDIA GPU を持つ利用者が、E5 の CPU cache と v2-m3 の CUDA 再順位付けを明示的に選ぶ方法。通常の `engine.search()` と CPU 用 `search_cpu_shortlist()` は変わらない。固定93文書・3問の API 経由実測は[実験記録](../experiments/cuda-shortlist-retrieval-v1.md)に記す。速度と VRAM はその環境の値に限る。

## 準備

NGR の通常 wheel には CUDA 依存と model weights は入らない。GPU 対応 PyTorch は OS、driver、CUDA wheel の組み合わせに合うものを [PyTorch の公式手順](https://pytorch.org/get-started/locally/) で別環境に導入する。E5 に必要な `numpy`、`onnxruntime`、`tokenizers` と、v2-m3 に必要な `transformers` もその環境に導入する。CUDA が PyTorch で利用可能か `python -c "import torch; print(torch.cuda.is_available())"` で確認する。

モデルは自動取得しない。[CPU guide](cpu-shortlist-retrieval.md) の明示 download と hash 確認手順で、E5 revision `614241f622f53c4eeff9890bdc4f31cfecc418b3` と v2-m3 revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` のローカル snapshot を用意する。実測の再現には[固定 manifest](../../tests/fixtures/cpu_shortlist_benchmark_v3.json)に記載された E5 ONNX / tokenizer と v2-m3 weights / tokenizer の SHA-256 照合も必要。API はファイル hash を検査しない。

## 選択と検索

```python
from neuron_graph_rag.engine import NeuronGraphRAG
from neuron_graph_rag.cpu_shortlist_retrieval import LocalPinnedE5
from neuron_graph_rag.cuda_shortlist_retrieval import (
    CudaShortlistRetriever,
    LocalPinnedCudaV2M3,
    attach_cuda_shortlist_retriever,
)

retriever = CudaShortlistRetriever(
    "shortlist.db",
    LocalPinnedE5(r"C:\models\e5\<revision>", threads=4),
    LocalPinnedCudaV2M3(r"C:\models\v2-m3\<revision>"),
)
try:
    with NeuronGraphRAG("documents.db") as engine:
        attach_cuda_shortlist_retriever(engine, retriever)
        engine.add_document("refund", "返金は購入から30日以内に申請できます。")
        receipt = engine.update_cuda_shortlist_cache()
        trace = engine.search_cuda_shortlist("返金の期限は？", timeout_seconds=120)
        print([hit.node.node_id for hit in trace.hits])
        print(receipt.cache_fingerprint, trace.diagnostics)
finally:
    retriever.close()
```

同じ retriever を使う間は CUDA model が VRAM に常駐し、次の検索で再利用される。`close()` は model の参照と PyTorch の再利用 cache を解放する。CUDA context や他の参照・プロセスの VRAM まで回収する操作ではない。検索結果の `cuda_peak_allocated_bytes` と `cuda_peak_reserved_bytes` は当該 PyTorch process の検索区間 peak を表す。

CUDA 不可、モデル欠落、OOM、timeout、cancel は例外として呼び出し元へ伝わる。timeout と cancel は batch 境界で確認し、実行中の CUDA kernel を直ちに停止しない。自動 CPU fallback はない。失敗後に CPU を選ぶ場合は、呼び出し元が別途 `CpuShortlistRetriever` を構築して明示的に呼ぶ。[仕様](../specifications/cuda-shortlist-retrieval.md)を参照。
