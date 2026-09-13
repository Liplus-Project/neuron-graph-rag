# CPU で多言語意味検索を試す

FastEmbed の [custom model 例](https://github.com/qdrant/fastembed#%EF%B8%8F-dense-text-embeddings)を使う。生成用LLMやGPUは不要。最初の検索では Hugging Face からモデルを取得するためネット接続が必要。取得済みモデルはcacheへ保存され、質問・本文の埋め込み計算はCPUで行う。

```powershell
uv venv .semantic-env --python 3.12
uv pip install --python .semantic-env/Scripts/python.exe -e . 'fastembed>=0.8,<0.9'
$env:HF_HOME = "$PWD/.semantic-hf"
$env:HF_XET_CACHE = "$PWD/.semantic-hf/xet"
```

```python
from tempfile import TemporaryDirectory
from pathlib import Path
from neuron_graph_rag.engine import NeuronGraphRAG
from neuron_graph_rag.semantic_retrieval import (
    MultilingualE5, SemanticRetriever, attach_semantic_retriever,
)

retriever = SemanticRetriever(MultilingualE5(".semantic-model-cache", threads=4))
with TemporaryDirectory() as directory:
    with NeuronGraphRAG(Path(directory) / "trial.db") as engine:
        attach_semantic_retriever(engine, retriever)
        engine.add_document("refund", "Customers can get their money back within thirty days.")
        engine.add_document("parcel", "Delivery takes three working days.")
        print(engine.search("代金を返してもらえる期限は？").hits[0].node.node_id)
```

学習済み検索だけを比較する場合は `retriever.score(query, nodes)` の値を降順に並べる。接続後の `engine.search` は既存の疎検索とグラフも合成する。元に戻す場合は接続前の `engine.dense_retriever` を保存し、同じhelperへ渡す。

診断を再実行する場合は出力に新しいパスを指定する。既存出力は上書きできない。

```powershell
.semantic-env/Scripts/python.exe tools/run_semantic_diagnostic.py --model-cache .semantic-model-cache --output artifacts/semantic-trial-new.json
```

[要件・制約](../specifications/semantic-retrieval.md)と[実測結果](../experiments/semantic-retrieval-cpu.md)を参照。
