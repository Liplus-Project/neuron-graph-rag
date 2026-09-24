# 新規実験コード

新しい実験専用 Python script はこのディレクトリに置く。既存の凍結済み実験 source file は source path と hash の証拠を維持するため移動しない。通常 wheel に必要な API は `src/neuron_graph_rag` に実装し、`setup.py` の明示的な配布リストと隔離 wheel テストを同じ変更で更新する。

- [v2-m3 CPU/CUDA 探索的比較](gpu-v2-m3-probe.md): RTX 3080 上で固定 v3 の shortlist pair を CPU と GPU に再投入する。既定検索・wheel には変更を加えない。
