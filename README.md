<img width="627" height="512" alt="ネオン輝くニューロングラフ・ロゴ" src="https://github.com/user-attachments/assets/04930bad-f493-4b55-a450-e6f57f74b962" />

##
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
- `nodes.created_at` / `nodes.last_used_at`: node を最初に取り込んだ時刻と、最後に利用され成功した時刻（UNIX 秒）

活性が減衰しても、`confidence` と `factuality` は変更されません。検索だけでも `edge.weight` と `last_used_at` は変更されません。

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

### Shared database home

optional MCP server は database path を次の優先順位で解決します。

1. `--database <path>`
2. `NGR_DATABASE`
3. `~/.ngrdb/knowledge.db`

したがって、同じ端末・同じユーザーで動く複数の AI client は、client 固有 directory ではなく user-owned な既定 database を共有できます。明示 path は従来どおり利用でき、core library の `NeuronGraphRAG()` と `SQLiteStore()` は引き続き `:memory:` が既定です。file-backed database は WAL と 5 秒の bounded busy timeout を使い、local process 間の read/write contention を SQLite transaction で直列化します。network drive、cloud sync、remote deployment はこの保証範囲に含みません。

既存 database は自動移動しません。source、destination、backup を明示し、既存 output を上書きしない一回性 migration を実行します。

```powershell
python tools/migrate_database.py `
  --source C:\path\to\existing\knowledge.db `
  --destination ~/.ngrdb/knowledge.db `
  --backup C:\path\to\backups\knowledge.before-shared-home.db
```

migration は source の integrity を確認し、SQLite backup API で backup と destination を作成して両方を再検証します。既存 destination または backup がある場合は、何も上書きせず停止します。

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

## Node expiry

Li+ の Knowledge 層 Tier 2（エージェント間で共有するローカルの棚）として使うため、最後に利用されてから一定日数が過ぎた node を消せます。自動では消しません。明示したときだけ実行します。

```bash
# 消える node を確かめる（削除しない）
python -m neuron_graph_rag expire-nodes --older-than-days 90 --dry-run

# 実際に消す
python -m neuron_graph_rag expire-nodes --older-than-days 90 --database ~/.ngr/db/knowledge.db
```

`--database` を省くと、MCP server と同じく `NGR_DATABASE`、設定ファイル、shared database home の順で解決します。存在しない database は作らずにエラーにします。

- 期限は `last_used_at` から数えます。一度も利用されていない node は `created_at` から数えます。
- `last_used_at` が進むのは、成功した利用を記録したときだけです（`record_success`、`record_source_use` の `used`、soft-start / evidence quorum の success feedback。confirmed-only でも `used` で進みます）。検索に当たっただけでは進みません。古い時刻の feedback で巻き戻ることもありません。
- 削除は一つの SQLite transaction（`BEGIN IMMEDIATE`）で行います。途中で失敗したら全体を戻します。
- node を消すと、その edge、activation、retrieval result、source-use、success / outcome の node 記録、edge に紐づく feedback 記録も cascade で消えます。BM25 と dense は検索のたびに `nodes` から計算するので、消した node はどちらの入口にも残りません。
- judgment の node は対象外です。judgment は archive / hard delete の lifecycle で扱います。
- library からは `rag.expire_unused_nodes(90, dry_run=True)` で呼べます。CPU / CUDA shortlist cache を attach している engine では、消した node を cache からも取り除きます。attach していない別 process の shortlist cache は、次の検索で stale として止まるので、`update_cache` で同期してください。

既存の database は開いたときに列を追加します。追加前からある node と、列を書かない古い writer が入れた node の `created_at` は、そのとき開いた時刻で埋めます。そのため既存の node は、移行した時点から期限を数え始めます。

## Minimal eval

```bash
python -m neuron_graph_rag eval
```

同じ 5 文書 corpus と 3 query に対して、次を比較します。

- `baseline_hybrid`: graph weight を 0 にした BM25 + dense retrieval
- `graph_rag`: 一つの入口 node から最大 2 hop 伝播する graph-integrated retrieval

指標は mean reciprocal rank、hit at 3、baseline より expected node の rank が改善した query 数です。この eval は品質ベンチマークではなく、graph 経路が baseline と異なる順位信号を生むことを検証する最小 smoke test です。

## D1 real-corpus fixture

`tests/fixtures/d1_liplus_wiki.json` は github-rag-mcp の本番 D1 から read-only で取得した決定論的な小 fixture です。`diff` と `wiki_doc` の時系列 metadata、Decision Structure の `mention` edge、検索、graph activation、success feedback を統合テストで再現します。

取得 tool は D1 の単一 `SELECT / WITH` だけを許可し、Wrangler が `rows_written=0` を返したことを query ごとに検証します。認証、再取得、schema fingerprint、coverage、既知 gap、完全性の限界は [docs/d1-corpus-fixture.md](docs/d1-corpus-fixture.md) を参照してください。

## Real-corpus benchmark

`tests/fixtures/d1_liplus_benchmark.json` と `.gold.json` は、12 wiki node の connected fixture と、結果を見る前に固定した 12 query（direct lookup / relation / negative control 各4件）です。

```bash
python -m neuron_graph_rag benchmark \
  --fixture tests/fixtures/d1_liplus_benchmark.json \
  --gold tests/fixtures/d1_liplus_benchmark.gold.json
```

同一 corpus・encoder・query 上の baseline / graph の MRR、Hit@3、rank delta に加え、one-hop / two-hop の説明経路と success feedback の局所性を検査します。品質結果は CI 合格条件にせず、固定 JSON と [観測記録](docs/real-corpus-benchmark.md) に支持・不支持・判定不能をそのまま残します。

初回観測では relation 改善と説明経路・feedback isolation は支持されましたが、negative control の2件が悪化したため非対象の過剰押し上げ仮説は不支持でした。詳細値と適用限界は観測記録を参照してください。

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

GitHub の最小 read-only adapter 候補は、coreへGitHub clientを持ち込まず、固定snapshotをlocal indexへ接続する形で検証しています。[github-rag-mcp replacement compatibility spike](docs/github-rag-mcp-replacement-compatibility.md) は、public repository一つの取得、保存済み `search` capture との比較、one-document update follow-upだけを扱います。committed observation は github-rag-mcp の最小 doc 検索 path の候補に限り、production serviceやMCP replacementを主張するものではありません。

既定 dense encoder は依存なしで再現可能な feature hashing です。実運用の埋め込みは callable を差し替えます。

```python
def my_encoder(text: str) -> list[float]:
    ...

rag = NeuronGraphRAG("knowledge.db", dense_encoder=my_encoder)
```

同じ encoder 呼び出し内で常に同じ次元数を返す必要があります。

## Optional MCP interface

local SQLite database を MCP 対応 AI へ stdio で接続する optional adapter を利用できます。MCP SDK は core の必須依存に含まれません。

```bash
pip install -e '.[mcp]'
neuron-graph-rag-mcp --database /absolute/path/to/knowledge.db
```

同じ adapter は SQLite 正本の判断を読む専用 tool `search_judgments`、`get_judgment`、`traverse_judgments` も公開します。これらは通常 `search` と異なり retrieval trace、activation、feedback を保存せず、read-only です。例えば `search_judgments` の `repository` には `liplus-language` や `neuron-graph-rag` の namespace を指定でき、archived judgment は `include_archived=true` を明示した時だけ検索・探索対象に入ります。接続先は workspace 固有 path を commit せず、上記の `/absolute/path/to/knowledge.db` を各 client の絶対 path へ置き換えます。

この接続例は local stdio の実装済み範囲だけを示します。Wiki 自動生成、共有フォルダ同期、remote / on-prem 配置は含みません。

To opt one local stdio server into the evaluated stabilization settings, pass both feedback options explicitly:

```bash
neuron-graph-rag-mcp \
  --database /absolute/path/to/knowledge.db \
  --relation-feedback-evidence-quorum 3 \
  --sibling-feedback-normalization 1.0
```

The CLI defaults remain quorum `1` and sibling normalization `0.0`. Invalid values are rejected before the database is opened. With positive sibling normalization, the MCP search tool returns the relation channel so feedback has the provenance required for local normalization; the default `0.0` path remains the existing hybrid search. The `3` / `1.0` command configures only that server process; it does not change library defaults, legacy engine/storage contracts, or establish production adoption.

confirmed outcome を唯一の正の trigger とする diminishing candidate も process 単位で明示 opt-in できます。flag と decay ratio は必ず組にし、既存の q3/s1 設定と default は変更しません。

```bash
neuron-graph-rag-mcp \
  --database /absolute/path/to/knowledge.db \
  --confirmed-outcome-reinforcement \
  --confirmation-decay-ratio 0.5 \
  --sibling-feedback-normalization 1.0
```

一般的な MCP client では次のように登録します。

```json
{
  "mcpServers": {
    "neuron-graph-rag": {
      "command": "neuron-graph-rag-mcp",
      "args": [
        "--database", "/absolute/path/to/knowledge.db",
        "--relation-feedback-evidence-quorum", "3",
        "--sibling-feedback-normalization", "1.0"
      ]
    }
  }
}
```

利用順序は `search` で得た `trace_id` と候補を保持し、実際の判断過程に合わせて `selected` → `validated` → `used` を `record_source_use` へ送ります。既定 policy では新規 `used` が credited edge の独立 evidence を記録し、設定 quorum 到達時に bounded reinforcement を発火します。confirmed-only candidate 有効時は `used` まで weight を変更せず、後から `record_outcome` へ送った独立 `confirmed` だけが初回 multiplier `1.0`、後続 geometric decay で保存済み relation path を強化します。`corrected`、`rolled_back`、`superseded` は既定 policy と confirmed-only policy のどちらでも weight を変更しません。

三つの tool の schema、model-facing description、trace retention、failure code の正本は [docs/optional-mcp-interface.md](docs/optional-mcp-interface.md) です。実装範囲は local stdio に限り、HTTP、認証、認可、remote deployment は含みません。

## Independent retrieval channels

`search_channels(query, limit=...)`は既存`search()`を変更せず、同一queryへ二つの独立した候補列を返します。

- `lexical`: BM25だけの順位。graph pathを保存しない
- `relation`: BM25+dense entryをseed選択にだけ使い、anchored edge-only graph activationだけで順位付けする
- 各laneは独立`trace_id`を持ち、`agreement_node_ids`は両方へ現れたnodeを示す
- cross-lane final score、combined rank、single winnerは返さない

```python
channels = rag.search_channels("How are these decisions related?", limit=5)
for hit in channels.lexical.hits:
    print("lexical", hit.explain())
for hit in channels.relation.hits:
    print("relation", hit.explain())

# lexical traceならedge不変、relation traceなら保存済みpathだけを強化する
rag.record_success(channels.relation.trace_id, ["related-node"])
```

callerは両laneを検査し、下流判断で実際に使用したlaneの`trace_id`と`node_id`をfeedbackへ渡します。channel文字列を後から自己申告せず、保存済みtrace provenanceがreinforcement有無を決めます。

Evidence-gated local feedback reinforcementは、credited edgeごとに異なるsuccess traceを永続evidenceとして数え、設定quorum到達後だけ既存bounded updateを一回ずつ適用するopt-in candidateです。`neuron_graph_rag.evidence_feedback` のclassから明示的に利用し、package rootと`.engine`のlegacy class identityは変更しません。既定quorumは`1`で現行動作を保ち、`2`以上では到達前のweightとsame-source siblingを変更しません。core / MCP receiptはcount、quorum、activationを返します。詳細は[Evidence-gated local feedback reinforcement](docs/evidence-gated-local-feedback-reinforcement.md)を参照してください。

Confirmed-outcome feedback reinforcementは、同じ candidate class の `confirmed_outcome_reinforcement=True` と明示 `confirmation_decay_ratio` で有効にします。`used` は履歴だけを保存し、relation trace 上で used となった node の一意な credited path を、独立 `confirmed` outcome ごとに減衰強化します。count、multiplier、actual delta、credited path は core / MCP receipt で同じ形に写され、SQLite restart 後も継続します。mechanics は default 採用や q3/s1 との優位性を主張しません。詳細は[Confirmed-outcome feedback reinforcement](docs/confirmed-outcome-feedback-reinforcement.md)を参照してください。

Soft-start feedback reinforcementは、`soft_start_feedback_reinforcement=True`、`soft_start_feedback_ratio`、`confirmation_decay_ratio` を明示する別の default-off candidate です。credited relation edge の最初の新規 `used` は通常 bounded update の一部だけを provisional に適用し、最初の独立 `confirmed` が残りを補完します。後続 confirmation は既存 geometric decay に従い、same-source sibling normalization は `used` ではなく各 confirmation の actual delta だけへ適用します。confirmed-only candidate、hard quorum と同時には有効化できず、既存 q1/s0、q3/s1、confirmed-only、凍結評価 artifact と default fingerprint は変更しません。詳細は[Confirmed-outcome feedback reinforcement](docs/confirmed-outcome-feedback-reinforcement.md#soft-start-successor-candidate)を参照してください。

Outcome-driven feedback deactivationは、soft-startへ `outcome_driven_feedback_deactivation=True` を追加した時だけ有効になるdefault-off candidateです。保存済みtrace / credited pathへ帰属できる`corrected`と`rolled_back`は、credited加算と同じcontributionのsame-source sibling normalizationを一体で正確に戻します。`superseded`はedgeと証拠を削除せずdormantにし、同じ保存済みpathの後続`confirmed`で再活性化します。lexical、zero-hop、別source、uncredited failure、基礎weight以下へのpunitive updateは変更しません。mechanicsとresult-free snapshot protocolだけを先にfreezeし、観測はmerge後のsuccessor Issueへ分離します。詳細は[Outcome-driven feedback deactivation](docs/outcome-driven-feedback-deactivation.md)を参照してください。

## Explanation model

各 `SearchHit` は次の情報を保持します。

- sparse score
- dense score
- entry score
- raw graph activation
- normalized graph activation
- raw / normalized BM25 score
- raw / normalized dense score
- competition前後のentry anchor
- final score
- seed node
- zero-hop / graph path種別
- entry / positive graph rank
- entry / graph fusion componentとfusion strategy
- path contribution
- path 上の edge type、weight、factuality

伝播は path 内の node 再訪を禁止し、`max_hops` と `max_propagation_expansions` で上限を設けます。同一 node に複数経路が到達した場合は活性を合算し、説明には寄与上位の経路を残します。

## Limits

- 既定 dense encoder は learned semantic embedding ではありません。意味検索品質が必要な環境では差し替えが必要です。
- SQLite を使う local MVP です。同一端末・同一ユーザーの bounded な複数 process contention は WAL と busy timeout で扱いますが、分散 ingestion や高並行 write は扱いません。
- edge は有向です。逆方向探索が必要なら逆 edge を明示的に登録します。
- 成功判定は自動化しません。呼び出し側が利用 node を明示します。
- reinforcement は credit assignment の最小実装として、成功結果の最大寄与経路を選びます。
- graph propagation は決定論的な重み付き探索であり、GNN 学習ではありません。
- eval corpus は機構確認用の小規模 fixture であり、一般的な retrieval 品質を保証しません。
- real-corpus benchmark も learned semantic embedding ではない現行 MVP 構成だけを評価し、Li+ D1 は損失あり snapshot であるため一般的な corpus / embedding 品質へ外挿できません。

受け入れ要件の詳細は [docs/requirements.md](docs/requirements.md) にあります。

### ノードの書き手と出典

既存のPython APIで、任意の `metadata.node_provenance` を指定できます。

```python
rag.add_document("decision-311", "共有ノードの根拠", metadata={
    "node_provenance": {
        "writer": "Luna",
        "source_refs": ["https://github.com/Liplus-Project/neuron-graph-rag/issues/311"],
    },
})
```

objectのキーは `writer` と `source_refs` の2つです。writerは非空文字列の自己申告で、認証済み身元ではありません。source_refsは1件以上のHTTPS GitHub issue/PR URLの配列で、`/issues/123` または `/pull/123` の形（末尾 `/` は任意、query/fragmentは不可）とし、URLの存在確認はしません。不正入力、null、空objectは保存前に拒否します。

再取り込みでキーを省略すると、既存のnode_provenanceだけを原子的に保持します。有効objectを明示すると全体を置換し、他のmetadataは従来どおり置換します。返却ノードもその書込みの保存内容を返します。created_at / last_used_atは保持し、既存judgmentのmetadata.provenanceの契約は変更せず、旧ノードへの推測補完は行いません。新しいDB列や取り込みMCP toolは追加せず、既存MCP通常/CUDA検索のmetadataに出典を返します。

## License

本プロジェクトは [Apache License 2.0](LICENSE) の下で提供されます。帰属情報は [NOTICE](NOTICE) を参照してください。
