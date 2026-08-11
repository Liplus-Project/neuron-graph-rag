# Optional MCP Feedback Interface

## 1. Status and purpose

この文書は、MCP 対応 AI が Neuron Graph RAG（NGR）を検索し、実際に利用した source と後から判明した結果を返すための、実装済み local stdio interface の契約を定義する。

この契約は次を意味しない。

- MCP SDK が NGR core の必須依存である
- 認証方式、transport、公開 endpoint、remote deployment が決定済みである
- delayed outcome が現在の edge weight を自動的に減算または巻き戻す

`src/neuron_graph_rag_mcp/` の optional adapter がこの契約を local stdio transport で実装する。`pip install -e '.[mcp]'` で追加依存を導入し、`neuron-graph-rag-mcp --database <path>` で起動する。NGR core は引き続き Python 標準ライブラリだけで動作する。

### Local stabilization opt-in

The stdio CLI accepts two process-local feedback settings:

```bash
neuron-graph-rag-mcp \
  --database /absolute/path/to/knowledge.db \
  --relation-feedback-evidence-quorum 3 \
  --sibling-feedback-normalization 1.0
```

`--relation-feedback-evidence-quorum` accepts positive integers and defaults to `1`. `--sibling-feedback-normalization` accepts finite values from `0.0` through `1.0` and defaults to `0.0`. The CLI validates both values before opening the SQLite database, then constructs the explicit `neuron_graph_rag.evidence_feedback.EngineConfig` used by that server process.

When sibling normalization is positive, the MCP `search` tool uses `search_channels(...).relation` and returns that relation trace through the existing MCP search result schema. This preserves the relation provenance required by candidate feedback normalization. At the default normalization `0.0`, the tool continues to use the existing hybrid `search()` path.

Omitting both options preserves the existing immediate-reinforcement and no-sibling-normalization behavior. The `3` / `1.0` combination is a reversible local opt-in supported by the frozen controlled evaluation; it is not an external-corpus generalization, a production-quality claim, or a project-wide default adoption. Library callers and legacy engine/storage identities remain unchanged. The normative process-local requirements are recorded in [MCP Feedback Stabilization Settings](mcp-feedback-stabilization-settings.md).

## 2. Protocol envelope

tool 名は `search`、`record_source_use`、`record_outcome` とする。すべての input と成功 output は JSON Schema で宣言し、未知 field を受け付けない。

成功時は MCP envelope の `resultType` を `complete` とし、機械処理用の `structuredContent` と、その同じ JSON を直列化した `TextContent` を返す。これは [MCP 2026-07-28 tools specification](https://modelcontextprotocol.io/specification/2026-07-28/server/tools) の tool result、structured content、後方互換性の指針に合わせる。

共通の version literal は次の値とする。

```json
{
  "contract_version": "ngr.mcp.feedback/v1"
}
```

別 version は同じ tool 名の silent reinterpretation ではなく、明示的な compatibility 判断を必要とする。

### 2.1 Model-facing description rule

tool の意味をこの文書だけに閉じ込めない。MCP client が `tools/list` から渡す `description` 自体に、consuming AI が feedback loop を正しく実行するための規則を含める。

各 tool の節にある英語 literal を model-facing `description` としてそのまま使う。英語にするのは、接続する model の対話言語に依存せず同じ行動契約を渡すためである。deployment 固有の trace retention だけは、`search` 節で定義する最終 sentence を具体的な実値へ置き換える。

推奨 annotation は次の通り。

| tool | `readOnlyHint` | `destructiveHint` | `idempotentHint` | `openWorldHint` |
| --- | --- | --- | --- | --- |
| `search` | `false` | `false` | `false` | `false` |
| `record_source_use` | `false` | `false` | `true` | `false` |
| `record_outcome` | `false` | `false` | `true` | `false` |

`search` は edge を強化しないが、retrieval trace と動的 activation を保存するため read-only ではない。annotation は表示上の hint であり、認証・認可の代替ではない。

## 3. Shared identifiers and validation

### 3.1 `trace_id`

- NGR core が `search` ごとに発行する opaque identifier
- v1 では小文字 hexadecimal 32 文字、pattern は `^[0-9a-f]{32}$`
- client は生成、短縮、case 変換、別 trace からの流用をしない
- feedback tool は identifier の形式に加え、保存済み retrieval trace の存在を確認する

### 3.2 `node_id`

- 1 文字以上 512 文字以下の文字列
- NUL と C0 control character を含めない
- path や URL として解釈せず、NGR node の opaque key として完全一致で扱う
- feedback tool では、指定した `trace_id` の検索結果に含まれることを確認する

### 3.3 `idempotency_key`

- client が書き込み call ごとに生成する 1 文字以上 128 文字以下の visible ASCII
- pattern は `^[A-Za-z0-9._:-]+$`
- 同じ key と同じ payload の再送は、初回と同じ receipt を返し、再記録・再強化しない
- 同じ key で payload が異なる場合は `idempotency_conflict` として拒否する

### 3.4 Enum and number rules

- enum はこの文書に列挙した小文字 literal のみを case-sensitive で受け付ける
- integer は JSON number の整数値でなければならず、boolean を整数として扱わない
- score は有限の JSON number とし、`NaN`、正負の infinity を拒否する
- input array は記載した上限を超えた時点で call 全体を拒否し、部分適用しない

### 3.5 Trace handle retention

永続 SQLite database を使う現在の NGR core は retrieval trace を時間で自動削除せず、TTL も持たない。したがって標準の persistent deployment では `trace_id` に自動 expiry はない。database の削除、置換、明示的な cleanup は保存状態そのものを失わせるため、その後の feedback は `unknown_trace` になる。

adapter または deployment が独自 retention を設ける場合は、次をすべて満たす。

- retention duration または expiry rule を `search` の model-facing description に具体的に書く
- `search` output の `trace_expires_at` に expiry の Unix timestamp seconds を返す
- expiry 後の `record_source_use` と `record_outcome` は、idempotency replay より先に expiry を判定して `unknown_trace` を返す
- expired trace を同じ identifier で復活または再利用しない

retention を hidden server configuration にしない。model は `search` の description と output だけで、feedback をいつまでに返す必要があるかを判断できなければならない。

## 4. Source-use stages

source-use は一つの順序付き状態として扱う。

| stage | 意味 | 発生主体 | reinforcement |
| --- | --- | --- | --- |
| `retrieved` | `search` の結果に候補として返った | NGR | なし |
| `selected` | consuming AI が詳細確認する source として選んだ | client | なし |
| `validated` | exact source を確認し、現在の判断材料として利用可能と判定した | client | なし |
| `used` | 最終回答、実装判断、レビュー判断などの根拠として実際に利用した | client | 新規遷移時に独立 evidence を記録し、設定 quorum 到達後だけ強化 |

`retrieved -> selected -> validated -> used` の順序を守る。既存状態と同じ stage の再送は idempotent no-op とする。後退、段階の飛び越し、`retrieved` の client 申告は拒否する。同じ `record_source_use` call 内では、同一 node の連続する複数段階を順に送ってよい。

`used` は「良さそう」「読んだ」という impression ではない。final artifact の根拠として使用した時点でのみ記録する。新しい `used` 遷移だけが credited edge の独立 evidence を記録でき、既定 quorum `1` では従来どおりその event が即時 reinforcement を発火する。quorum `2` 以上では到達前の serving weight を変更しない。`retrieved`、`selected`、`validated`、duplicate / retry、delayed outcome は evidence と reinforcement を発火しない。

## 5. Tool: `search`

### 5.1 Meaning

query に対して NGR core の hybrid retrieval と graph activation propagation を実行し、検索 trace と説明可能な hit を返す。候補が返ったことは `retrieved` の観測であり、source-use や成功を意味しない。

### 5.2 Normative model-facing description

自動 expiry を持たない標準の persistent core では、次を exact `description` とする。

```text
Search Neuron Graph RAG and return ranked source candidates with a trace_id. Returned hits are only retrieved candidates: retrieval does not mean a source was selected, validated, or used, and search alone never reinforces graph weights. Retain trace_id until feedback is complete. After a returned source is actually used in a downstream answer, implementation decision, or review, call record_source_use with that trace_id and node_id using stage used; record selected and validated when those earlier transitions occur. In the persistent NGR core, trace handles do not expire automatically.
```

deployment が retention を設ける場合は、上の最後の sentence だけを、具体値を含む次の形へ置き換える。`<retention policy>` を literal のまま公開してはならない。

```text
In this deployment, trace handles expire <retention policy>; feedback after expiry returns unknown_trace.
```

例えば 24 時間 retention なら、最後の sentence は `In this deployment, trace handles expire 24 hours after search; feedback after expiry returns unknown_trace.` となる。

### 5.3 Input

```json
{
  "contract_version": "ngr.mcp.feedback/v1",
  "query": "How was decision D17 implemented?",
  "limit": 5
}
```

| field | required | validation |
| --- | --- | --- |
| `contract_version` | yes | literal `ngr.mcp.feedback/v1` |
| `query` | yes | trim 後 1 文字以上 8192 文字以下 |
| `limit` | no | integer、1 以上 100 以下、default `5` |

### 5.4 Output

```json
{
  "contract_version": "ngr.mcp.feedback/v1",
  "trace_id": "0123456789abcdef0123456789abcdef",
  "query": "How was decision D17 implemented?",
  "created_at": 1785312000.0,
  "trace_expires_at": null,
  "hits": [
    {
      "node_id": "pr-42",
      "rank": 1,
      "text": "Pull request 42 implemented D17.",
      "metadata": {
        "kind": "pull_request"
      },
      "confidence": 1.0,
      "source_use_stage": "retrieved",
      "scores": {
        "sparse": 0.8,
        "dense": 0.9,
        "entry": 0.845,
        "graph_activation": 0.7,
        "final": 0.78
      },
      "paths": [
        {
          "seed_id": "decision-17",
          "contribution": 0.7,
          "steps": [
            {
              "source_id": "decision-17",
              "target_id": "pr-42",
              "edge_type": "implemented_by",
              "edge_weight": 0.7,
              "factuality": 1.0
            }
          ]
        }
      ]
    }
  ]
}
```

`hits` は `rank` 昇順で、最大 `limit` 件とする。hit がない場合は空 array を返してよい。空 corpus は正常な空検索とは区別し、`empty_corpus` error とする。

`trace_expires_at` は自動 expiry がなければ `null`、retention があれば Unix timestamp seconds とする。

### 5.5 Core mapping

| MCP field or behavior | Current NGR API |
| --- | --- |
| `query`, `limit` | `NeuronGraphRAG.search(query, limit=limit)` |
| `trace_id`, `query`, `created_at` | `SearchTrace` |
| `trace_expires_at` | persistent core は `null`、retention を持つ adapter が計算 |
| `node_id`, `text`, `metadata`, `confidence` | `SearchHit.node` |
| `scores`, `paths` | `SearchHit.explain()` と score field |
| `rank` | `SearchTrace.hits` の 1-based 順序 |
| `source_use_stage` | adapter が `retrieved` として表現 |

adapter は core の test 用 `now` parameter を MCP input に公開しない。

## 6. Tool: `record_source_use`

### 6.1 Meaning

consuming AI が、検索結果をどこまで判断材料として利用したかを記録する。`used` への新規遷移だけを NGR core の独立 evidence に接続し、credited edge の設定 quorum 到達時に bounded reinforcement を実行する。

### 6.2 Normative model-facing description

次を exact `description` とする。

```text
Record ordered source-use transitions for candidates from one Neuron Graph RAG search trace. Use selected only when a source is chosen for inspection, validated only after its exact source is checked and accepted as usable, and used only after it becomes an actual basis of a downstream answer, implementation decision, or review. Transitions must occur in order. A newly recorded used source can add one independent evidence item per credited edge; graph reinforcement occurs only when that edge's configured evidence quorum has been reached. The default quorum is one. Retrieved, selected, validated, retries, duplicate traces, and duplicate stages add no evidence and do not reinforce. If the trace handle has expired or does not exist, this tool returns unknown_trace.
```

### 6.3 Input

```json
{
  "contract_version": "ngr.mcp.feedback/v1",
  "idempotency_key": "answer-7-source-use-1",
  "trace_id": "0123456789abcdef0123456789abcdef",
  "events": [
    {
      "node_id": "pr-42",
      "stage": "selected"
    },
    {
      "node_id": "pr-42",
      "stage": "validated"
    },
    {
      "node_id": "pr-42",
      "stage": "used"
    }
  ]
}
```

| field | required | validation |
| --- | --- | --- |
| `contract_version` | yes | literal `ngr.mcp.feedback/v1` |
| `idempotency_key` | yes | shared identifier rule |
| `trace_id` | yes | shared identifier rule、trace が存在する |
| `events` | yes | 1 件以上 100 件以下 |
| `events[].node_id` | yes | shared identifier rule、指定 trace の hit に含まれる |
| `events[].stage` | yes | `selected`、`validated`、`used` のいずれか |

event は array 順に評価する。同一 call の途中で一件でも不正なら、stage ledger と reinforcement の両方を call 前の状態に保つ。

### 6.4 Output

```json
{
  "contract_version": "ngr.mcp.feedback/v1",
  "receipt_id": "fedcba9876543210fedcba9876543210",
  "trace_id": "0123456789abcdef0123456789abcdef",
  "events": [
    {
      "node_id": "pr-42",
      "stage": "selected",
      "changed": true
    },
    {
      "node_id": "pr-42",
      "stage": "validated",
      "changed": true
    },
    {
      "node_id": "pr-42",
      "stage": "used",
      "changed": true
    }
  ],
  "newly_used_node_ids": [
    "pr-42"
  ],
  "feedback": {
    "feedback_id": "00112233445566778899aabbccddeeff",
    "used_node_ids": [
      "pr-42"
    ],
    "reinforced_edges": [
      {
        "source_id": "decision-17",
        "target_id": "pr-42",
        "edge_type": "implemented_by",
        "old_weight": 0.7,
        "new_weight": 0.84
      }
    ],
    "evidence": [
      {
        "source_id": "decision-17",
        "target_id": "pr-42",
        "edge_type": "implemented_by",
        "count": 1,
        "quorum": 1,
        "activated": true
      }
    ]
  }
}
```

新規 `used` がない場合、`newly_used_node_ids` は空 array、`feedback` は `null` とする。

### 6.5 Core mapping

- adapter は transport-neutral な `FeedbackLedger.record_source_use` を呼び、stage ledger へ直接 SQL を発行しない。
- 一つの call で新しく `used` へ到達した node 群だけを `NeuronGraphRAG.record_success(trace_id, newly_used_node_ids)` へ一度渡す。`record_success` は credited-path 選択、contribution clamp、edge increment、channel、sibling normalization の唯一の計画元とする。
- source-use の outer transaction は `record_success` の inner commit を遅延させ、stage 遷移、idempotency receipt、reinforcement をまとめて commit または rollback する。
- `FeedbackReceipt` の `feedback_id`、`used_node_ids`、`reinforced_edges`、edge ごとの `evidence` を `feedback` に写す。quorum 前は `evidence` を返し、`reinforced_edges` は空とする。activation が maximum weight で cap された場合は、既存挙動どおり `old_weight == new_weight` の reinforced edge を返せるが、actual delta が `0` なので sibling は変更しない。
- 再送、同一 stage、すでに `used` の node は reinforcement 処理を再度適用しない。

core domain API は取得済みでない node を stage 更新前に拒否し、adapter は caller が修正可能な error code へ写す。

## 7. Tool: `record_outcome`

### 7.1 Meaning

source を利用した判断や artifact に後から判明した結果を、即時 source-use とは別軸で記録する。v1 では評価用の履歴であり、edge weight を自動変更しない。

### 7.2 Normative model-facing description

次を exact `description` とする。

```text
Record a delayed outcome for sources that were already marked used, such as confirmed, corrected, rolled_back, or superseded. In v1, delayed outcomes are audit and evaluation records only: they do not add, subtract, undo, or otherwise change graph weights. Do not use this tool instead of record_source_use for immediate source-use feedback. If the trace handle has expired or does not exist, this tool returns unknown_trace.
```

### 7.3 Outcome enum

| outcome | 意味 |
| --- | --- |
| `confirmed` | 後続の証拠または運用結果が、source を使った判断を支持した |
| `corrected` | 判断または artifact の一部が後から修正された |
| `rolled_back` | 判断または artifact が撤回、revert、rollback された |
| `superseded` | 誤りと断定せず、新しい前提または判断に置き換えられた |

`corrected` と `rolled_back` を即時の負の reinforcement に変換しない。query、index、source selection、source 自体、実装のどこに原因があるかを一件の outcome だけで判別できないためである。`confirmed` も `used` の reinforcement を重複加算しない。

### 7.4 Input

```json
{
  "contract_version": "ngr.mcp.feedback/v1",
  "idempotency_key": "pr-42-rollback-1",
  "trace_id": "0123456789abcdef0123456789abcdef",
  "node_ids": [
    "pr-42"
  ],
  "outcome": "rolled_back",
  "summary": "The implementation was reverted after a production regression.",
  "external_ref": "https://github.com/example/project/pull/42"
}
```

| field | required | validation |
| --- | --- | --- |
| `contract_version` | yes | literal `ngr.mcp.feedback/v1` |
| `idempotency_key` | yes | shared identifier rule |
| `trace_id` | yes | shared identifier rule、trace が存在する |
| `node_ids` | yes | 1 件以上 100 件以下、重複不可 |
| `node_ids[]` | yes | shared identifier rule、指定 trace ですでに `used` |
| `outcome` | yes | `confirmed`、`corrected`、`rolled_back`、`superseded` |
| `summary` | yes | trim 後 1 文字以上 2000 文字以下 |
| `external_ref` | no | absolute `https` URL、2048 文字以下 |

### 7.5 Output

```json
{
  "contract_version": "ngr.mcp.feedback/v1",
  "outcome_id": "ffeeddccbbaa99887766554433221100",
  "trace_id": "0123456789abcdef0123456789abcdef",
  "node_ids": [
    "pr-42"
  ],
  "outcome": "rolled_back",
  "recorded_at": 1785315600.0,
  "reinforcement_applied": false
}
```

`reinforcement_applied` は v1 では常に `false` とする。将来 delayed outcome を学習へ接続する場合は、原因帰属、weight rollback、再計算可能性を別の versioned policy として定義する。

### 7.6 Core mapping

transport-neutral な `FeedbackLedger.record_outcome` は outcome ledger にだけ保存し、`record_success`、edge update、activation update を呼ばない。

## 8. Failure contract

tool 名が不明、または `tools/call` 自体が MCP request schema を満たさない場合は protocol error とする。tool が受理した call の input validation、存在確認、状態遷移、core execution の失敗は、MCP の Tool Execution Error として `isError: true` で返す。

error の text content は、次の object を JSON 直列化する。

```json
{
  "code": "invalid_stage_transition",
  "message": "node pr-42 must reach validated before used",
  "retryable": false
}
```

| code | condition | retryable |
| --- | --- | --- |
| `unsupported_contract_version` | version literal が一致しない | false |
| `invalid_argument` | schema、長さ、pattern、enum、number が不正 | false |
| `empty_corpus` | 検索対象 node がない | false |
| `unknown_trace` | 保存済み trace がない、または retention により expiry 済み | false |
| `node_not_in_trace` | node が指定 trace の hit ではない | false |
| `source_not_used` | outcome 対象が `used` に到達していない | false |
| `invalid_stage_transition` | source-use の順序違反または後退 | false |
| `idempotency_conflict` | 同じ key に異なる payload が割り当てられた | false |
| `core_unavailable` | database lock など一時的な core failure | true |
| `internal_error` | caller が修正できない予期しない失敗 | false |

error message に query 本文、source 本文、credential、stack trace を含めない。server log との相関が必要なら、公開データを含まない correlation identifier を別 field として追加してよい。

## 9. Dependency boundary

同一 repository 内での初期配置は次の依存方向を守る。

```text
MCP host/client
      |
optional MCP adapter
      |
public neuron_graph_rag API
      |
SQLite core + retrieval engine
```

- `src/neuron_graph_rag/` は MCP SDK、transport、host 固有型を import しない
- adapter だけが MCP SDK に依存し、optional extra または別 install target とする
- core test、demo、eval は adapter を install せず実行できる
- adapter は `NeuronGraphRAG` の public API を呼び、storage 内部へ直接 SQL を発行しない
- stage ledger と outcome ledger に必要な core 拡張は、MCP 型ではなく transport-neutral な domain API として追加する
- stdio、HTTP、認証、認可、tenant isolation、remote deployment はこの契約の外側で決める

## 10. Repository separation criteria

MCP であることだけを理由に別 repository へ分離しない。次の圧力が具体化した時に分離を再判断する。

- core と独立した release cadence または互換性保証が必要になる
- MCP SDK や transport dependency が core の install、test、security update を拘束する
- 独立 deployment、scaling、credential boundary、incident response が必要になる
- NGR repository 外の複数 consumer が adapter 単体を versioned product として必要とする
- ownership または contribution policy が core と分かれる

分離する場合も、この文書の tool semantics と version literal を compatibility surface とし、core への依存方向を逆転させない。

## 11. Implementation acceptance

実装は最低限、次を検証する。

- core-only install に MCP dependency が混入しない
- three tools の input/output schema と error code が契約に一致する
- `search` だけでは edge weight が変わらない
- `selected` と `validated` では edge weight が変わらない
- 新規 `used` だけが一度だけ `record_success` を呼ぶ
- retry と duplicate stage が reinforcement を重複させない
- 同一 trace、idempotency replay、duplicate stage が evidence count を重複させず、quorum 前は serving weight を変更しない
- `corrected`、`rolled_back` を含む delayed outcome が weight を変更しない
- invalid trace、trace 外 node、enum、stage 順序、idempotency conflict を拒否する
- `tools/list` の description だけから feedback 順序、reinforcement 条件、delayed outcome 非変更規則を判断できる
- persistent core では `trace_expires_at` が `null`、retention deployment では具体的な description と timestamp が一致する
- retention expiry 後は両 feedback tool が `unknown_trace` を返す
- core test、demo、eval が adapter なしで引き続き通る
