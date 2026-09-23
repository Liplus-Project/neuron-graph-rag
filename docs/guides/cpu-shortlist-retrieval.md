# CPU shortlist検索を試す

CPUだけで文書を絞り込み、E5とv2-m3 rerankerで順位を付ける明示的なopt-in APIである。既定の `engine.search()` と既存の意味検索設定は変わらない。GPUも生成用LLMも使わないが、モデルの読み込みと検索には相応の時間とメモリが必要になる。

公開条件を固定した[実測 v3](../experiments/cpu-shortlist-retrieval-v3.md)では、Windows native・CPU 4 thread・93文書・2065 chunkのcold indexが63.641秒、3つのwarm queryが38.770〜43.343秒、process peak RSSが最大約2.93 GiBだった。3問の期待sourceはすべてtop5に入った。この値はその固定corpusと環境での結果であり、手元の文書数やCPUで同じ待ち時間・品質になる保証ではない。

## 準備

Python 3.11の環境でNGRと、実測に使用した依存版をインストールする。既にこれらの依存がある環境を使うなら、新しいvenvは不要。以下はPowerShellでの新規環境の例である。

```powershell
uv venv .cpu-shortlist-env --python 3.11
uv pip install --python .cpu-shortlist-env/Scripts/python.exe -e . "numpy==2.4.6" "onnxruntime==1.23.2" "tokenizers==0.19.1" "torch==2.4.1" "transformers==4.44.2" "huggingface-hub==0.36.2"
```

モデルは自動取得しない。既にあるsnapshotを共有してもよい。取得が必要な場合だけ、次の**明示操作**を一度実行し、表示された `e5` と `v2-m3` のsnapshot pathを控える。モデルのコピーを実験フォルダごとに作る必要はない。

```powershell
.cpu-shortlist-env/Scripts/python.exe -c "import sys; from neuron_graph_rag.cpu_shortlist_retrieval import download_cpu_shortlist_models; print(download_cpu_shortlist_models(sys.argv[1]))" "$env:LOCALAPPDATA\NeuronGraphRAG\models"
```

`LocalPinnedE5` は指定されたsnapshotの `onnx/model.onnx` と `tokenizer.json`、`LocalPinnedV2M3` は `model.safetensors`、`config.json`、`tokenizer.json` を読む。**実測と同じモデルとして使う前に**、次の4ファイルのSHA-256を[固定manifest](../../tests/fixtures/cpu_shortlist_benchmark_v3.json)と照合する。API自体はローカルファイルの内容ハッシュを検査しないため、異なるsnapshotを指定しても自動検出されない。モデル取得も推論もAPIを明示的に呼ぶまで始まらない。

```powershell
$e5 = '表示されたE5のsnapshot path'
$v2m3 = '表示されたv2-m3のsnapshot path'
Get-FileHash -Algorithm SHA256 (Join-Path $e5 'onnx/model.onnx')
Get-FileHash -Algorithm SHA256 (Join-Path $e5 'tokenizer.json')
Get-FileHash -Algorithm SHA256 (Join-Path $v2m3 'model.safetensors')
Get-FileHash -Algorithm SHA256 (Join-Path $v2m3 'tokenizer.json')
```

## 検索する

次の例のsnapshot pathを、上で表示されたパスか既存の共有snapshotへ置き換える。`shortlist.db` はE5索引用のSQLiteで、次回起動時も再利用できる。文書は通常のNGR databaseへ登録し、検索前にcacheを更新する。既存cacheと文書・モデル・構造化規則が一致しなければ明示的に失敗する。

```python
from pathlib import Path

from neuron_graph_rag.cpu_shortlist_retrieval import (
    CpuShortlistRetriever,
    LocalPinnedE5,
    LocalPinnedV2M3,
    attach_cpu_shortlist_retriever,
)
from neuron_graph_rag.engine import NeuronGraphRAG

e5_snapshot = Path(r"ここをE5のsnapshot pathに置き換える")
v2_m3_snapshot = Path(r"ここをv2-m3のsnapshot pathに置き換える")
retriever = CpuShortlistRetriever(
    "shortlist.db",
    LocalPinnedE5(e5_snapshot, threads=4),
    LocalPinnedV2M3(v2_m3_snapshot, threads=4),
)

with NeuronGraphRAG("documents.db") as engine:
    attach_cpu_shortlist_retriever(engine, retriever)
    engine.add_document("refund", "返金は購入から30日以内に申請できます。", metadata={"path": "docs/refund.md"})
    engine.add_document("shipping", "配送には通常3営業日かかります。", metadata={"path": "docs/shipping.md"})
    receipt = engine.update_cpu_shortlist_cache()
    trace = engine.search_cpu_shortlist("返金の期限は？", limit=2, timeout_seconds=60)
    for hit in trace.hits:
        print(hit.node.node_id, hit.score)
    print(receipt.cache_fingerprint, trace.diagnostics)
```

索引の初回構築は文書全体をE5で埋め込む。以後の `update_cpu_shortlist_cache()` は追加・変更・削除分だけを処理する。検索は固定のK=50候補、各文書の上位2 chunk、最大100 forward pairでv2-m3を実行する。`timeout_seconds` を超えた場合やキャンセル時は例外を返し、既定検索の順位に黙って切り替えない。長い処理を表示したい呼び出し元は `progress` callbackを渡せる。検索結果の `diagnostics` にはE5・v2-m3別時間、forward pair数、peak RSS、モデルrevisionとcache fingerprintが入る。

スコア尺度が異なるため、この結果と既定のdense/sparse/graph順位は自動合成しない。現時点では独立した試用APIであり、既定検索や常駐アプリの検索画面への採用は別判断とする。[仕様](../specifications/cpu-shortlist-retrieval.md)を参照。
