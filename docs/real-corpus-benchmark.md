# Real-corpus benchmark

## 目的

実際の Li+ D1 corpus 上で、現行 NGR MVP の graph-integrated retrieval が baseline hybrid より有用かを再現可能に観測する。順位だけでなく、説明経路と success feedback の局所性を同じ固定契約で検査する。

## 固定入力

- fixture: `tests/fixtures/d1_liplus_benchmark.json`
- provenance: `tests/fixtures/d1_liplus_benchmark.provenance.json`
- gold: `tests/fixtures/d1_liplus_benchmark.gold.json`
- 12 wiki node / 26 `mention` edge の弱連結 graph
- provenance の `known_gaps` は空配列。取得対象は `wiki_doc` だけであり、取得時点でこの固定 subset に関係する未解消 gap はない
- direct lookup / relation / negative control 各4件
- relation は独立した endpoint 組の one-hop 2件と two-hop 2件
- baseline は `entry_weight=1.0`、`graph_weight=0.0`
- graph は既存 smoke eval と同じ `entry_weight=0.25`、`graph_weight=0.75`
- 両者とも同じ feature-hashing encoder、`seed_count=1`、`max_hops=2`

fixture の node 集合、gold query、許容 rank、期待 path、判定規則は品質結果を見る前に固定する。結果を見た後に変更が必要になった場合は、旧契約と旧結果を残し、別 version として理由を記録する。

初回観測後の provenance 再監査でも fixture SHA-256 は `b3b305aabb57803c2782c3998215e1cbcf9b5e6cdef0f641abc98520d4400cf9`、gold SHA-256 は `a10af7a1a4ec5f0ef66a228b05e14fe0160e4ff5f9b3209073f38fdb90028c71` のまま byte-identical だった。再監査では、wiki-only capture と無関係な diff tracker を `known_gaps` から除き、取得時点で関連する未解消 gap だけを記録する #7 の契約へ戻した。freeze commit `befe245` と初回 result は変更していない。

## 仮説判定

- H1: relation MRR が上昇し、改善件数が悪化件数を上回れば支持。逆条件なら不支持、それ以外は判定不能。
- H2: direct / negative-control がすべて許容 rank 内で悪化ゼロなら支持。許容外または2 rank以上の悪化があれば不支持、それ以外は判定不能。
- H3: 全 relation case で固定 endpoint / edge type の完全な path が説明に存在すれば支持。一件でも欠ければ不支持。
- H4: credited edge が1本以上変化し、credited 外 edge と非対象 case rank が不変なら支持。汚染があれば不支持、edge 変化がなければ判定不能。

## 実行

```bash
python -m neuron_graph_rag benchmark \
  --fixture tests/fixtures/d1_liplus_benchmark.json \
  --gold tests/fixtures/d1_liplus_benchmark.gold.json \
  --output tests/fixtures/d1_liplus_benchmark.result.json
```

CI は gold schema、fixture / provenance、metric 計算、path 照合、feedback isolation、固定 result の再生成一致を検証する。観測品質の数値自体は CI 合格閾値にしない。

## 観測結果

gold / fixture / 判定規則を commit `befe245` で固定した後に初回実行した。機械可読な全結果は `tests/fixtures/d1_liplus_benchmark.result.json` に保存する。

| cohort | baseline MRR | graph MRR | baseline Hit@3 | graph Hit@3 | 改善 / 同値 / 悪化 |
|---|---:|---:|---:|---:|---:|
| direct lookup | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 / 4 / 0 |
| relation | 0.2583 | 0.3833 | 0.5000 | 0.7500 | 3 / 1 / 0 |
| negative control | 1.0000 | 0.7083 | 1.0000 | 1.0000 | 0 / 2 / 2 |
| overall | 0.7528 | 0.6972 | 0.8333 | 0.9167 | 3 / 7 / 2 |

- H1: **支持**。relation は3件改善・0件悪化で、MRR と Hit@3 が上昇した。
- H2: **不支持**。negative-configuration は rank 1 → 3、negative-installation は rank 1 → 2 に悪化した。両方とも許容 rank 3 内だが、事前規則の「悪化ゼロ」を満たさず、前者は2 rank 悪化の不支持条件にも該当する。
- H3: **支持**。one-hop 2件と two-hop 2件の全4件で、固定した endpoint / `mention` path が説明に存在した。
- H4: **支持**。`1.-Model` → `2.-Evolution` の credited `mention` edge だけが weight 1.0 → 1.14 に変化した。非 credited edge と非対象 case rank は不変だった。対象 case 自体の rank は2 → 2であり、強化は起きたが即時 rank 改善は観測されなかった。

relation cohort では graph 統合の有用性が観測された一方、negative control の悪化により overall MRR は低下した。したがって「graph 統合は常に baseline より良い」という結論は支持しない。

## 適用限界

D1 は content truncation と diff 欠損を含み得る損失あり検索 snapshot であり、GitHub が byte-exact history の正本である。既定 dense encoder は learned semantic embedding ではない。この結果は固定した public Li+ subset と現行 MVP 構成の評価であり、一般的な corpus、embedding、GraphRAG 実装へ外挿しない。
