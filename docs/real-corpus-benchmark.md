# Real-corpus benchmark

## 目的

実際の Li+ D1 corpus 上で、現行 NGR MVP の graph-integrated retrieval が baseline hybrid より有用かを再現可能に観測する。順位だけでなく、説明経路と success feedback の局所性を同じ固定契約で検査する。

## 固定入力

- fixture: `tests/fixtures/d1_liplus_benchmark.json`
- provenance: `tests/fixtures/d1_liplus_benchmark.provenance.json`
- gold: `tests/fixtures/d1_liplus_benchmark.gold.json`
- 12 wiki node / 26 `mention` edge の弱連結 graph
- direct lookup / relation / negative control 各4件
- relation は独立した endpoint 組の one-hop 2件と two-hop 2件
- baseline は `entry_weight=1.0`、`graph_weight=0.0`
- graph は既存 smoke eval と同じ `entry_weight=0.25`、`graph_weight=0.75`
- 両者とも同じ feature-hashing encoder、`seed_count=1`、`max_hops=2`

fixture の node 集合、gold query、許容 rank、期待 path、判定規則は品質結果を見る前に固定する。結果を見た後に変更が必要になった場合は、旧契約と旧結果を残し、別 version として理由を記録する。

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

初回実行前。gold freeze commit 後に、生成 JSON の数値と H1-H4 の支持 / 不支持 / 判定不能をそのまま追記する。

## 適用限界

D1 は content truncation と diff 欠損を含み得る損失あり検索 snapshot であり、GitHub が byte-exact history の正本である。既定 dense encoder は learned semantic embedding ではない。この結果は固定した public Li+ subset と現行 MVP 構成の評価であり、一般的な corpus、embedding、GraphRAG 実装へ外挿しない。
