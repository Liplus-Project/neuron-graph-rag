# Neuron Graph RAG

Neuron Graph RAG は、ハイブリッド検索を入口にし、型付き知識グラフへ活性を伝播し、実際に利用されて成功した経路だけを強化する、観測可能な RAG エンジンです。

この MVP は次の縦切りを一つのローカル実行で成立させます。

1. 文書ノードと型付きエッジを SQLite に取り込む
2. BM25 と dense cosine similarity から入口ノードを決める
3. エッジ重み、事実性、hop decay を使って活性を伝播する
4. 入口スコアとグラフ活性を統合して順位付けする
5. 検索 trace と成功利用を別々に記録する
6. 成功した結果へ至る経路だけを強化する
7. 同じ query を再検索し、経路の活性変化を説明する

## Architecture

```text
documents
   |
   +--> BM25 -------------------+
   |                            |
   +--> dense encoder + cosine -+--> entry score --> seed nodes
                                                    |
typed edges + factuality + weight ------------------+
                                                    |
                                           activation propagation
                                                    |
entry score + graph activation --> ranked results + path explanation
                                                    |
                                      explicit success feedback
                                                    |
                                reinforce successful path edges only
```

永続化層は次の軸を分離します。

- `nodes.confidence`: 知識内容に対する確信度
- `edges.factuality`: 関係が事実である度合い
- `edges.weight`: 探索経路としての結合強度
- `activation_state`: 検索時に生じた時間減衰する動的活性
- `retrievals` / `retrieval_results`: 何が検索されたか
- `success_feedback` / `success_nodes`: 何が実際に利用され成功したか

活性が減衰しても、`confidence` と `factuality` は変更されません。検索だけでも `edge.weight` は変更されません。

## Terminology

- Entry score: BM25 と dense score を正規化して統合した入口スコア
- Seed node: entry score 上位の活性伝播開始点
- Graph activation: seed から重み付きエッジを通って届いた活性の合計
- Factuality: edge の事実性。学習対象の結合重みとは別の値
- Trace: query、rank、各スコア、説明経路をまとめた検索記録
- Success feedback: 利用され成功した node を呼び出し側が明示するイベント
- Reinforcement: 成功 node の上位説明経路に含まれる edge weight だけを増やす処理

## Setup

Python 3.11 以上を使います。runtime dependency は Python 標準ライブラリだけです。

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

`uv` を使う場合は、repository root で次の一行でも実行できます。

```bash
uv run --python 3.11 --with-editable . python -m unittest discover -s tests -v
```

## Vertical slice demo

```bash
python -m neuron_graph_rag demo
```

永続化結果も確認する場合:

```bash
python -m neuron_graph_rag demo --db demo.db
```

JSON 出力には次が含まれます。

- feedback 前後の target rank と graph activation
- target の entry score と伝播経路
- 強化された edge の旧 weight と新 weight
- retrieval record 数と success feedback record 数

demo は `implementation` node が利用され成功したと申告します。最も寄与した `policy -> decision -> implementation` 経路だけが強化され、再検索時の説明に新しい edge weight が現れます。

## Minimal eval

```bash
python -m neuron_graph_rag eval
```

同じ 5 文書 corpus と 3 query に対して、次を比較します。

- `baseline_hybrid`: graph weight を 0 にした BM25 + dense retrieval
- `graph_rag`: 一つの入口 node から最大 2 hop 伝播する graph-integrated retrieval

指標は mean reciprocal rank、hit at 3、baseline より expected node の rank が改善した query 数です。この eval は品質ベンチマークではなく、graph 経路が baseline と異なる順位信号を生むことを検証する最小 smoke test です。

## Public API

```python
from neuron_graph_rag import NeuronGraphRAG

with NeuronGraphRAG("knowledge.db") as rag:
    rag.add_document(
        "decision-17",
        "Decision D17 accepted five retry attempts.",
        metadata={"kind": "decision"},
        confidence=0.95,
    )
    rag.add_document(
        "pr-42",
        "Pull request 42 implemented D17.",
        metadata={"kind": "pull_request"},
    )
    rag.add_edge(
        "decision-17",
        "pr-42",
        "implemented_by",
        weight=0.7,
        factuality=1.0,
    )

    trace = rag.search("How was D17 implemented?", limit=5)
    for hit in trace.hits:
        print(hit.explain())

    rag.record_success(trace.trace_id, ["pr-42"])
```

特定データ源の model は公開 API に含みません。GitHub、Decision Structure、Graphify などは、`add_document` と `add_edge` を呼ぶ将来の adapter として追加できます。

既定 dense encoder は依存なしで再現可能な feature hashing です。実運用の埋め込みは callable を差し替えます。

```python
def my_encoder(text: str) -> list[float]:
    ...

rag = NeuronGraphRAG("knowledge.db", dense_encoder=my_encoder)
```

同じ encoder 呼び出し内で常に同じ次元数を返す必要があります。

## Explanation model

各 `SearchHit` は次の情報を保持します。

- sparse score
- dense score
- entry score
- raw graph activation
- final score
- seed node
- path contribution
- path 上の edge type、weight、factuality

伝播は path 内の node 再訪を禁止し、`max_hops` と `max_propagation_expansions` で上限を設けます。同一 node に複数経路が到達した場合は活性を合算し、説明には寄与上位の経路を残します。

## Limits

- 既定 dense encoder は learned semantic embedding ではありません。意味検索品質が必要な環境では差し替えが必要です。
- SQLite を使う単一 process 向け MVP です。分散 ingestion や高並行 write は扱いません。
- edge は有向です。逆方向探索が必要なら逆 edge を明示的に登録します。
- 成功判定は自動化しません。呼び出し側が利用 node を明示します。
- reinforcement は credit assignment の最小実装として、成功結果の最大寄与経路を選びます。
- graph propagation は決定論的な重み付き探索であり、GNN 学習ではありません。
- eval corpus は機構確認用の小規模 fixture であり、一般的な retrieval 品質を保証しません。

受け入れ要件の詳細は [docs/requirements.md](docs/requirements.md) にあります。
