# v2-m3 CPU/CUDA 探索的比較の再実行

この実験は [固定 v3](../docs/experiments/cpu-shortlist-retrieval-v3.md) の93文書・3問・E5 shortlist K=50・文書内 top2 chunk から、同じ候補 pair を作り、`BAAI/bge-reranker-v2-m3` の CPU と CUDA FP32 推論を同じ Windows ホストで比較する。既定の NGR 検索や配布依存は変更しない。`tests/fixtures` と `tests/evidence` の v3 artifact は読み取り専用の照合元として使う。

## 入力と隔離

実行環境、モデル、cache copy、観測 JSON は repo 外の `D:\Users\hal\Codex\workspace\experiments\ngr-gpu-probe-2026-09-24` 以下に置く。元の v3 cache (`workspace/experiments/cpu-shortlist-246-run-v3/cache.db`) があればコピーして使う。なければ runner が指定した新規 cache を E5 から構築する。runner は corpus・manifest・4 model file の SHA-256、snapshot revision、cache fingerprint と pair 数を v3 記録に照合する。出力ファイルは既存 path を拒否するので、反復実験では別名を指定する。

モデルを新規取得する場合は、固定 revision を指定して以下のファイルだけを対象ディレクトリへ保存する。既存 snapshot からコピーしてもよい。runner は hash が異なれば測定前に停止する。

```python
from huggingface_hub import snapshot_download
snapshot_download("intfloat/multilingual-e5-small", revision="614241f622f53c4eeff9890bdc4f31cfecc418b3", local_dir=r"D:\Users\hal\Codex\workspace\experiments\ngr-gpu-probe-2026-09-24\models\e5\614241f622f53c4eeff9890bdc4f31cfecc418b3", allow_patterns=["onnx/model.onnx", "tokenizer.json"])
snapshot_download("BAAI/bge-reranker-v2-m3", revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", local_dir=r"D:\Users\hal\Codex\workspace\experiments\ngr-gpu-probe-2026-09-24\models\v2-m3\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"])
```

| モデル | revision | 必要なファイル |
| --- | --- | --- |
| `intfloat/multilingual-e5-small` | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | `onnx/model.onnx`, `tokenizer.json` |
| `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `sentencepiece.bpe.model` |

## Windows PowerShell 手順

```powershell
$probe = 'D:\Users\hal\Codex\workspace\experiments\ngr-gpu-probe-2026-09-24'
py -3.11 -m venv "$probe\venv"
& "$probe\venv\Scripts\python.exe" -m pip install 'torch==2.4.1+cu121' --index-url https://download.pytorch.org/whl/cu121
& "$probe\venv\Scripts\python.exe" -m pip install 'transformers==4.44.2' 'tokenizers==0.19.1' 'huggingface_hub==0.36.2' 'onnxruntime==1.23.2' 'numpy==2.4.6' safetensors
Copy-Item 'D:\Users\hal\Codex\workspace\experiments\cpu-shortlist-246-run-v3\cache.db' "$probe\cache-copy.db" # 既存 cache がある場合のみ
$env:PYTHONPATH = 'src'
$env:PYTHONDONTWRITEBYTECODE = '1'
& "$probe\venv\Scripts\python.exe" experiments\gpu_v2_m3_probe.py `
  --cache "$probe\cache-copy.db" `
  --e5-snapshot "$probe\models\e5\614241f622f53c4eeff9890bdc4f31cfecc418b3" `
  --reranker-snapshot "$probe\models\v2-m3\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e" `
  --output "$probe\observed.json"
```

実行前にモデルを `$probe\models` の上記パスへ配置する。GPU memory は実行直前の `nvidia-smi` と PyTorch の peak allocated / reserved を記録する。各 device の model load、最初の全 pair cold 推論、3問の warm 推論を別々に測り、CUDA は wall clock の区間末で synchronize する。各問で source 全順位、期待 source 順位、CPU/GPU logits の最大絶対差を残す。失敗時も stage と traceback を出力する。観測 JSON は探索的データであり、既定動作の性能保証ではない。
