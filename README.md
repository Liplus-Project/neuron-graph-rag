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

## Neural dynamics experiment

正方向加算、有限活性 budget、側方抑制、query-conditioned transmission、反復競合の13 variants を、PR #10 の development set と非重複9-node holdout で比較しました。manifest、holdout、gold、選択規則、停止規則は result 観測前の commit `4f240dd` で固定しています。

development では `budget-025` が relation MRR 0.3833 を維持し、negative-control MRR を 0.7083 から 1.0000 へ改善して選択されました。holdout を一度だけ開封した結果、direct / negative-control、path 3/3、feedback isolation は維持した一方、relation MRR が 0.3611 から 0.3254 へ退行しました。固定 stop rule に従って不採用とし、既定 strategy は `current_positive_additive` のままです。

全13 variants の gate 不合格と tradeoff、選択理由、holdout 判定は [Neural dynamics experiment](docs/neural-dynamics-experiment.md) と versioned result JSON に保存しています。現在の holdout は再選択や parameter 調整に再利用しません。

## Local recurrent competition experiment

PR #12のglobal recurrent tradeoffを受け、競合を同じsourceのsibling neighborへ局所化し、query relevanceとactive path identityを独立に比較する6-variant experimentを定義しています。

production D1から取得した新しいdevelopment / holdoutは、旧development、開封済み旧holdout、相互間でdoc pathとnode IDを分離しています。両provenance、contamination audit、二baseline gate、one-time holdout停止規則は[Local recurrent competition experiment](docs/neural-dynamics-local-competition-experiment.md)を参照してください。

freeze後のdevelopmentでは、queryなしのlocal variantsがrelationを改善した一方でdirect / negative-controlを退行させ、query variantsはnegative-controlを維持した一方でbest prior recurrentのrelationを上回れませんでした。候補gate通過は0件だったためholdoutは開かず、既定strategyは`current_positive_additive`のままです。

## Anchored BM25 and graph hybrid experiment

entry retrievalを競合外のzero-hop anchorとして保持し、graph scoreを1 edge以上通過したmessageだけに限定するanchored local strategyを追加しました。BM25-onlyはdense encoderとgraph traversalを呼ばない真のablationです。

production D1からread-only取得した新しい5-node development / holdoutは、過去4 fixturesの39 doc pathsおよび相互間から分離しています。固定した6 variants、raw / normalized score trace、contamination audit、候補gate、one-time holdout停止規則は[Anchored BM25 and graph hybrid experiment](docs/anchored-bm25-graph-hybrid-experiment.md)を参照してください。

freeze後のdevelopmentでは、anchored 3 variantsがrelation MRRを`current`の0.5000から0.7500–1.0000へ改善しましたが、direct lookupとnegative-controlがともに退行しました。候補gate通過は0件だったためholdoutは開かず、既定strategyは`current_positive_additive`のままです。

## Anchored fusion calibration experiment

Issue #15で分離したentry anchorとedge-only graph signalは維持したまま、graph尺度とfinal fusionだけを比較します。graph normalizationは`max`、rawの`none`、`l1_mass`を選択でき、final fusionはlinearとpositive graph nodeだけを順位付けするbottom-centered weighted RRFを選択できます。

production D1からread-only取得した新しい3-node development / holdoutは、既存7 fixturesの50 unique doc pathsおよび相互間から分離しています。固定6 variants、fusion formula、個別case non-regression gate、one-time holdout停止規則は[Anchored fusion calibration experiment](docs/anchored-fusion-calibration-experiment.md)を参照してください。

freeze後のdevelopmentでは、unscaled linearとbalanced RRFがrelation MRRを0.4167から0.6667へ改善しましたがdirect / negative-controlを1.0000から0.7500へ退行させました。conservative linear、L1 mass、conservative RRFはcontrolsを維持した一方relationを改善しませんでした。候補0件のためholdoutは未開封で、既定strategyは`current_positive_additive`のままです。

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

一般的な MCP client では次のように登録します。

```json
{
  "mcpServers": {
    "neuron-graph-rag": {
      "command": "neuron-graph-rag-mcp",
      "args": ["--database", "/absolute/path/to/knowledge.db"]
    }
  }
}
```

利用順序は `search` で得た `trace_id` と候補を保持し、実際の判断過程に合わせて `selected` → `validated` → `used` を `record_source_use` へ送ります。新規 `used` だけが credited edge の独立 evidence を記録し、設定 quorum 到達時に bounded reinforcement を発火します。既定 quorum は `1` です。後から結果が判明した場合は `record_outcome` で `confirmed`、`corrected`、`rolled_back`、`superseded` のいずれかを監査記録へ追加しますが、v1 の delayed outcome は weight を変更しません。

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

callerは両laneを検査し、下流判断で実際に使用したlaneの`trace_id`と`node_id`をfeedbackへ渡します。channel文字列を後から自己申告せず、保存済みtrace provenanceがreinforcement有無を決めます。固定D1 split、4-case hard gate、one-time holdout規則は[Independent retrieval channels experiment](docs/independent-retrieval-channels-experiment.md)を参照してください。

凍結後のdevelopmentではrelation MRR改善、lane parity、feedback帰属を含む10/12 gateが成立しましたが、rank-1 lexical controlとfrozen path-shape matcherの2 gateが不合格でした。停止規則に従ってholdoutは開かず、`search_channels()`へvalidated判定を付与していません。既存`search()`が引き続きdefaultです。

次段のblind selection protocolは、二つのlaneを変更せず、答えを示すfieldとlane scoreを除いたpacketをfresh judge 3体へ渡します。actual LLM callはparent orchestratorに限定し、repositoryはpacket生成、trace / node所属検証、raw response保存、majority集約、path射影、v1 byte hash監査、development失敗時のholdout停止だけを実装します。

実装・prompt・schema・gateをresult-free commit `062c131`としてpushした後、development packetを一度生成し、fresh judge 3体で観測しました。3件目が必須4 caseのうち1件を欠いたためresponse validationで停止し、retry、replacement、majority集約、accuracy計算を行っていません。development gateは不合格でholdout packetも生成せず、defaultとvalidated状態は変更していません。詳細は[Blind LLM channel selection experiment](docs/blind-llm-channel-selection-experiment.md)を参照してください。

Linux CIとfresh Windows worktreeの間でv1 / v2 frozen text artifactにLF / CRLF checkout差が生じるため、raw hashを優先し、完全なLF / CRLF相互変換だけをalternate verificationとして許可しました。本文差分やmixed newlineは拒否し、観測artifactと結果は変更していません。

Version 3のnode-first protocolは、一つのcase packetをfresh judge一体へ渡す12 independent invocationsとしてresponse completenessを構造化します。採否はchannelではなくnode IDの2/3 majorityで行い、同じnodeへlexical / relationの票が分かれてもcorrect evidenceとして扱います。channelは実traceのfeedback provenanceと補助分布として保持します。

result-free freeze `1fa2001` 後のdevelopment / holdoutは各12 fresh responsesで全12 gateを通過しました。両splitの4 caseはexpected nodeでunanimous majorityとなり、developmentのselected-node MRRは0.875、holdoutは1.0でした。これは frozen minimal holdout 上のblind node-first selection を支持する観測であり、default、production router、`search_channels()`全体のvalidated状態は変更しません。v1 / v2 artifactとinvalid resultは変更せず、v3も再生成・再集約しません。詳細は[Node-first blind selection experiment](docs/node-first-blind-selection-experiment.md)を参照してください。

Trace-credited feedback adaptationは、relation traceの`record_success`が後続relation retrievalをcontrolより改善するかを、同一frozen corpusとscheduleで比較します。controlもfeedback eventを記録しますがedgeを変更せず、treatmentだけがcredited pathを強化します。result-free manifestをpushした後はdevelopmentを一度だけ実行し、全gate通過時だけholdoutを一度開きます。既定API、router、production品質の主張は変更しません。詳細は[Trace-credited feedback adaptation experiment](docs/feedback-adaptation-experiment.md)を参照してください。

独立reproductionはprior feedback-adaptation resultを選択入力にせず、新規D1 development / holdout splitで同じtrace-credit claimを検証します。relation pathはruntime fieldを除き、`source_id`、`target_id`、`edge_type`だけへ射影してgoldと比較します。prior fixtureは識別子だけのcontamination auditに使い、prior goldとresultは読みません。詳細は[Trace-credited feedback adaptation reproduction experiment](docs/feedback-adaptation-reproduction-experiment.md)を参照してください。

Feedback rank elasticity runnerは、source SQLiteを変更せず、各累積feedback checkpointをfresh cloneから再生します。target rankだけでなくraw / normalized graph score、final-score margin、top-k rank delta、非対象churnを出力し、max-normalization ceiling、rank flip threshold、schedule全体のrank安定を区別します。これは診断専用であり、learning rate、fusion、normalization、既定値を変更しません。仕様と実行方法は[Feedback rank elasticity](docs/feedback-rank-elasticity.md)を参照してください。

Evidence-gated local feedback reinforcementは、credited edgeごとに異なるsuccess traceを永続evidenceとして数え、設定quorum到達後だけ既存bounded updateを一回ずつ適用するopt-in candidateです。`neuron_graph_rag.evidence_feedback` のclassから明示的に利用し、package rootと`.engine`のlegacy class identityは変更しません。既定quorumは`1`で現行動作を保ち、`2`以上では到達前のweightとsame-source siblingを変更しません。core / MCP receiptはcount、quorum、activationを返します。詳細は[Evidence-gated local feedback reinforcement](docs/evidence-gated-local-feedback-reinforcement.md)を参照してください。

Evidence-gated feedback controlled evaluationは、quorum `3`とsame-source sibling normalization `1.0`の組合せを、identity-disjointなresult-free development / conditional holdoutで比較します。4variantとfeedback count `[0, 1, 2, 3, 4, 10]`をcheckpointごとにfresh replayし、rank flip timing、top-k churn、control non-regression、mutation scope、atomic rollbackをhard gateにします。既存defaultと過去observed artifactは変更しません。詳細は[Evidence-gated feedback controlled evaluation](docs/evidence-gated-feedback-controlled-evaluation.md)を参照してください。

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
- SQLite を使う単一 process 向け MVP です。分散 ingestion や高並行 write は扱いません。
- edge は有向です。逆方向探索が必要なら逆 edge を明示的に登録します。
- 成功判定は自動化しません。呼び出し側が利用 node を明示します。
- reinforcement は credit assignment の最小実装として、成功結果の最大寄与経路を選びます。
- graph propagation は決定論的な重み付き探索であり、GNN 学習ではありません。
- eval corpus は機構確認用の小規模 fixture であり、一般的な retrieval 品質を保証しません。
- real-corpus benchmark も learned semantic embedding ではない現行 MVP 構成だけを評価し、Li+ D1 は損失あり snapshot であるため一般的な corpus / embedding 品質へ外挿できません。

受け入れ要件の詳細は [docs/requirements.md](docs/requirements.md) にあります。

## License

本プロジェクトは [Apache License 2.0](LICENSE) の下で提供されます。帰属情報は [NOTICE](NOTICE) を参照してください。
