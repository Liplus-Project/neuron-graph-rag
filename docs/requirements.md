# Requirements

## 1. Purpose

このプロジェクトは、ハイブリッド検索、型付き知識グラフ、決定論的な活性伝播、成功フィードバックを、一つの観測可能な RAG パイプラインとして提供する。

## 2. Premises

- 公開 API は特定のデータ源に依存しない。
- コアは Python 標準ライブラリだけで動作する。
- MCP 接続は同一 repository 内の任意 adapter とし、コアの必須依存にしない。
- sparse 検索は BM25、dense 検索は決定論的な feature-hashing encoder を既定実装とする。
- dense encoder は差し替え可能とし、既定実装を意味埋め込みモデルとは見なさない。
- グラフは型付き有向エッジであり、重みと事実性を別々に保持する。
- 活性は検索時刻に紐づく動的状態であり、知識の確信度やエッジの事実性とは別軸である。

## 3. Functional requirements

番号は詰めない。欠番は #313 で削除した研究用実験（評価 protocol、凍結 artifact、gate）の要件で、他文書や過去の issue / PR からの「要件 N」参照を壊さないため番号を残す。

1. 文書ノード、metadata、知識確信度を SQLite に永続化できる。
   任意の予約キー `metadata.node_provenance` は、非空の文字列 `writer` と、1件以上の HTTPS GitHub issue/PR URL を含む文字列配列 `source_refs` だけを持つ object とする。URL は `https://github.com/{owner}/{repo}/issues/{positive-number}` または `/pull/{positive-number}`（末尾 `/` は任意、query/fragment は不可）とし、存在確認は行わない。writer は自己申告で認証済み身元ではない。不正入力（null/空objectを含む）は保存前に拒否する。再取り込みでキー省略時は既存の同キーだけを原子的に保持し、明示した有効objectで全体を置換する。他のmetadataは従来どおり置換し、API返却値はその書込みの保存内容に一致する。旧ノードへの推測補完、judgmentの `metadata.provenance` 変更、日時の更新、schema移行は行わない。既存Python APIで取り込み、MCP通常/CUDA検索のmetadataで返す。
2. 型付きエッジ、結合重み、事実性を SQLite に永続化できる。
3. BM25 と dense cosine similarity を正規化し、入口スコアへ統合できる。
4. 上位入口ノードから、重み、事実性、hop decay を使って決定論的に活性を伝播できる。
5. 入口スコアとグラフ活性を統合して結果を順位付けできる。
6. 結果ごとに sparse、dense、入口、グラフ、最終スコアと伝播経路を説明できる。
7. 検索 trace と検索結果を、成功フィードバックとは別の記録として保持できる。
8. 成功時に明示された利用ノードへ至る経路だけを強化できる。
9. 検索しただけではエッジ重みを変更しない。
10. 活性値は半減期に従って時間減衰する。
11. 活性減衰はノード確信度とエッジ事実性を変更しない。
12. 同一コーパスで通常のハイブリッド検索とグラフ統合検索を比較できる。
13. 任意 MCP adapter の `search`、`record_source_use`、`record_outcome`、`write_judgment` と judgment 専用 read tool 契約を、実装と transport から独立して定義する。
14. source-use を `retrieved`、`selected`、`validated`、`used` に分け、新規 `used` への遷移だけを即時 reinforcement に接続する。
15. `corrected`、`rolled_back` などの delayed outcome を source-use と別に記録し、初期契約では edge weight を自動変更しない。
16. MCP adapter は trace、node、enum、stage 順序、idempotency を境界で検証する。
17. 各 MCP tool の model-facing description 自体が、feedback の呼び分けと reinforcement 条件を consuming AI へ伝える。
18. persistent core の trace は自動 expiry しない。retention を設ける deployment は `search` description と output に期限を明示し、expiry 後の feedback を `unknown_trace` とする。
19. github-rag-mcp の D1 `search_docs` を正本として、repo / type / per-type limit と固定順から決定論的な小 fixture を生成できる。
20. `search_docs.vector_id / content` を node ID / text へ、`doc_edges` の両端と `edge_kind` を typed edge へ変換し、欠損 endpoint は node を捏造せず除外理由を記録できる。
21. D1 取得は単一 SELECT / WITH query のみに制限し、各 query の `rows_written=0`、`changes=0`、`changed_db=false` を検証できる。
22. fixture と分離した provenance report に schema fingerprint、coverage、取得時刻、取得時点で未解消の既知 gap、redaction 件数を記録し、再取得前後の count / commit / 最新時刻を比較できる。未解消の既知 gap がない場合は空配列を記録する。
23. 実 D1 の明示的な wiki doc path 集合から、弱連結で決定論的な評価 fixture を生成できる。gold case と品質結果を見た後に選択集合を変更しない。
24. 12 件以上の gold query を direct lookup、relation、negative control に分け、query、期待 node、許容 rank、source URL、relation 時の期待 endpoint / edge type を保持できる。
25. baseline hybrid と graph-integrated retrieval を同一 corpus、encoder、query で実行し、全体・cohort 別の MRR、Hit@3、rank delta、改善・同値・悪化件数を機械可読に出力できる。
26. relation の説明は score だけでなく、固定した one-hop / two-hop の endpoint と edge type に対して照合する。
27. success feedback 前後で edge weight と全 gold case の rank を比較し、credited path 外の edge 変更と非対象 case の rank 変更を明示する。
28. 活性伝播は共通 interface の下で、現行正方向加算、有限活性 budget、側方抑制、query-conditioned transmission、反復競合を選択できる。
29. 各検索 trace は strategy、伝播 step 数、展開数、活性総量、収束有無、停止理由を決定論的な diagnostics として保持する。

要件 30〜33 は欠番。

34. recurrent activation は global inhibition に加え、同じ source の sibling neighbor だけを競合させる local strategy を選択できる。
35. local recurrent strategy は query relevance と active path identity を独立に有効化でき、競合集合ごとに source、path identity、neighbor 数、query relevance、配分前後の message 総量を記録する。

要件 36〜40 は欠番。

41. `EngineConfig` の活性伝播 strategy の既定値は `current_positive_additive` とする。
42. entry retrievalはgraph競合の外側にzero-hop anchorとして保持でき、競合前後で同一値であることをtrace diagnosticsで検証できる。
43. anchored graph signalは少なくとも1 edgeを通ったmessageだけで構成し、zero-hop seed residualとzero-hop pathを含めない。
44. dense retrievalとgraph propagationを独立に無効化でき、BM25-only ablationではdense encoderとgraph traversalを実行しない。
45. explanationはBM25 / denseのraw・normalized値、競合前後のentry anchor、graphのraw・normalized値、final score、zero-hop / graph path種別を区別して保持する。

要件 46〜49 は欠番。

50. graph activationは`max`、`none`、`l1_mass`を一般設定として選択でき、zero totalを決定論的に全0へ変換できる。
51. final fusionは既存linearに加え、entry rankとpositive graph nodeだけのgraph rankを使うbottom-centered weighted RRFを選択できる。
52. 各traceはentry / graph rank、entry / graph fusion component、normalization、fusion strategy、RRF k、positive graph node数を保持し、final orderingを機械的に再計算できる。

要件 53〜55 は欠番。

56. `search_channels`は同一queryからBM25 lexical laneとanchored edge-only relation laneを独立trace IDで同時返却し、cross-lane final score、combined rank、single winnerを生成しない。
57. lexical laneはBM25だけで順位付けし、dense retrievalとgraph propagationをlane順位へ使用せず、保存hitへgraph pathを持たせない。
58. relation laneはBM25+dense entryをseed選択だけに使い、`anchored_local_competition`で1 edge以上を通過したpositive graph nodeをraw activation降順・node ID昇順で順位付けする。
59. channel provenanceはcallerのchannel自己申告でなく独立trace IDに保存し、default config の`record_success`はlexical traceでedgeを変更せず、relation traceでは保存済みcredited pathだけを強化する。
60. 同一nodeが両laneに現れる場合も各rankと説明を保持し、片方のtraceに保存されていないnodeへのfeedbackをatomicに拒否する。

要件 61〜79 は欠番。

80. sibling relation feedback normalization は、明示的に有効化された candidate config でのみ、relation trace の credited edge を強化し、その edge と同じ source から出る未 credit sibling だけを局所的に正規化できる。lexical trace、zero-hop、未関係 source、credited sibling は変更しない。candidate は synthetic isolation test と result-free development / holdout 相当の relation、direct、lexical、negative-control gate を通過するまで default にしない。

要件 81〜82 は欠番。

83. soft-start feedback reinforcement は明示 opt-in の relation-only candidate とし、最初の新規 `used` で通常 bounded increment の固定 ratio 分だけを credited path へ適用する。最初の独立 `confirmed` は同じ schedule の残量を一回分の通常 increment まで補い、後続 confirmation は固定 decay ratio で加算する。used 時は sibling normalization を行わず、confirmation の actual delta だけを同一 source の uncredited sibling へ配分する。duplicate、lexical、zero-hop、別 source、uncredited edge、negative outcome は変更せず、candidate mechanics の合格だけで default や local serving policy を変更しない。

要件 84〜85 は欠番。

86. outcome-driven feedback deactivation はsoft-startと同時にだけ有効化できるdefault-off candidateとする。provisional / confirmationごとにcredited加算と同時発生したsame-source sibling normalization減算を一つのsigned mutation journalへ永続化し、因果帰属できる`corrected` / `rolled_back`だけが未反転contributionを基礎weight未満へ下げずexact reversalする。`superseded`はedge、evidence、trace、outcomeを削除せずrelation edgeをdormantにして通常activationから除外し、同じ保存済みcredited pathの後続`confirmed`で再活性化する。duplicate、retry、restart、transaction failure、lexical、zero-hop、別source、uncredited edge、因果帰属不能outcomeは二重減算または局所外mutationを行わない。

要件 87 は欠番。

88. judgment store の新規 judgment graph は SQLite の stable identity、revision、lifecycle、provenance、typed relation を machine-readable 正本とする。NGR 自身の Decision Structure の正本は GitHub Wiki であり、store へ取り込んだ NGR 自身の判断は Wiki の写しとして扱う。add / update / supersede / archive / restore / hard-delete candidate は raw SQL でなく atomic domain API を通し、stale revision、dangling relation、部分更新、二重 successor を fail closed にする。archive は通常 retrieval から外す論理的忘却、hard delete は履歴参照のない archived candidate だけに許す物理削除として分離する。current graph の deterministic export / atomic import と SQLite backup / integrity-checked restore を維持し、既存 Wiki entry の本番移行は fixture 検証後に分離する。
89. Li+ / NGR Decision Structure Wiki pilot は各 repository の index が列挙する entry だけを専用の新規 SQLite へ取り込み、repository namespace 付き identity、page 本文、Wiki URL、repository、取得 commit、source state、typed relation を保持する。duplicate identity、unknown relation target、parser ambiguity、partial publication、既存出力の上書きを fail closed にし、SQLite / supersession integrity、deterministic export、backup を検証する。Wiki、既存検索 DB、凍結済み feedback 実験 DB は変更せず、本 pilot だけで正本を切り替えない。
90. judgment 専用 read API は `search_judgments`、`get_judgment`、`traverse_judgments` を提供し、current revision、lifecycle、statement、rationale、provenance、typed relation を返す。search は judgment の既存 node projection に同じ lexical / dense scorer と有効 weight を適用し、既定で active のみ、明示指定時だけ archived を含め、repository namespace で絞り込める。traversal は relation type、incoming / outgoing / both、1 以上の有限 hop を受け、cycle-safe な hop 優先・stable identity 順を維持する。三操作と対応 MCP tool は成功時・失敗時とも judgment、revision、relation、retrieval trace、feedback、node、edge、activation を永続変更せず、MCP は read-only annotation、未知 field を拒否する schema、既存 error envelope を持つ。既存 `search`、feedback、`write_judgment`、library default は変更しない。
91. optional MCP server の永続 database path は明示 `--database`、`NGR_DATABASE`、`~/.ngr/config.json` の `database`、既定 `~/.ngr/db/knowledge.db` の順で解決し（未移行の旧 `~/.ngrdb/knowledge.db` があるときは既定解決を拒否し `--migrate-home --confirm-stopped` を求める）、同一端末・同一ユーザーの複数 AI client が client-neutral な user-owned 正本を共有できる。明示 path と core library の `:memory:` 既定を維持し、file-backed connection だけに WAL と 5 秒の bounded busy timeout を適用する。既存 database は自動移動・統合せず、source / destination / backup をすべて明示し、source integrity、SQLite backup、destination integrity、既存 output 上書き拒否を備えた一回性 migration だけを提供する。network drive、cloud sync、remote deployment は保証範囲外とする。
92. judgment relation semantics は共有 SQLite の relation type registry、definition revision、namespace、provenance、active / deprecated lifecycle を正本とする。`depends_on`、`refines`、`supersedes`、`conflicts_with`、`informs` は `docs/Decision-Structure.md` と一致する revision 1 として決定的かつ冪等に登録し、既存 relation は内容を失わず `assertion_kind=explicit` と revision 1 binding へ移行する。新規 assertion は登録済み type の current revision を束縛し、未知 type は revision 未束縛の explicit assertion として保存する。active type は warning なし、未知 / deprecated type は structured advisory warning を返すが write を拒否しない。registry write は atomic domain API と `expected_revision` を要求し stale update を fail closed にする。read / validate は成功・失敗とも全 persistent table を変更しない。canonical `judgment_relations` は explicit assertion 専用とし、reasoner、推論結果、taxonomy、OWL / RDF / JSON-LD、query expansion、retrieval score / feedback / activation への意味接続を含めない。export / import、backup / restore、integrity check は registry revision と assertion binding を保持する。

要件 93 は欠番。

94. Li+ Knowledge Tier 2 の node expiry は明示 opt-in の操作とし、自動では削除しない。各 node は nullable な `created_at`（初回取り込み時刻、再取り込みで変えない）と `last_used_at`（最後に成功した利用の時刻）を持つ。`last_used_at` は success feedback と source-use の `used` 記録でだけ進み、検索、`selected`、`validated` では進まず、古い時刻では巻き戻らない。expiry は `last_used_at`、未利用なら `created_at` が N 日より古い judgment 以外の node を一つの `BEGIN IMMEDIATE` transaction で削除し、edge、activation、retrieval result、source-use、success / outcome node、edge 依存の feedback 記録を cascade で除き、失敗時は全体を戻す。BM25 / dense の入口は検索ごとに `nodes` から計算し、attach 済み shortlist cache は commit 後に同じ node を除き、未同期の cache は stale として fail closed にする。既存 database は開いたときに列を追加し、列のない行の `created_at` をその時刻で埋める。CLI `expire-nodes` は `--dry-run` を持ち、存在しない database を作らない。

## 4. Constraints

- GNN 学習、自動正誤判定、分散実行、GitHub 専用 UI は対象外とする。
- reinforcement は成功の申告でのみ発火し、単なる retrieval impression を学習信号にしない。
- MCP SDK、transport、認証、remote deployment はコア要件に含めない。
- 経路は循環を避け、最大 hop 数と結果ごとの最大説明経路数で計算量を制限する。
- edge weight の強化には上限を設ける。ただし、既存 weight が上限を超えている場合も強化処理で現在値を引き下げない。

## 5. Acceptance verification

- `python -m unittest discover -s tests -v` が単体テストと統合テストを通過する。
- `python -m neuron_graph_rag demo` が取り込み、検索、成功フィードバック、再検索を実演する。
- `python -m neuron_graph_rag eval` が baseline hybrid と graph retrieval の比較指標を出力する。
- `python -m neuron_graph_rag benchmark --fixture ... --gold ...` が固定実コーパス上の比較、説明経路、feedback isolation、仮説判定を出力する。
- CI が editable install、test、eval を新規環境で実行する。
- [Optional MCP Feedback Interface](optional-mcp-interface.md) が tool semantics、input、output、failure、core mapping、依存境界、repository 分離条件を定義する。
- `tests/fixtures/d1_liplus_wiki.json` が実 D1 形状から ingest、検索、時系列 metadata、graph activation、success feedback を再現する。
- [D1 corpus fixture](d1-corpus-fixture.md) が read-only 取得、認証境界、provenance、coverage 比較、再取得手順を定義する。
- [Real-corpus benchmark](real-corpus-benchmark.md) が gold freeze、判定規則、観測結果、外挿限界を定義する。
- [Sibling relation feedback normalization](sibling-relation-feedback-normalization.md) が opt-in candidate の局所 sibling 正規化、trace isolation、default 変更前の検証境界を定義する。
- [Confirmed-outcome feedback reinforcement](confirmed-outcome-feedback-reinforcement.md) が confirmed-only と soft-start の明示 policy、永続 schedule、transaction、receipt、default-preserving boundary を定義する。
- [Outcome-driven feedback deactivation](outcome-driven-feedback-deactivation.md) がsigned contribution journal、exact reversal、dormancy / reactivation、default-off境界、result-free freezeを定義する。
- [Canonical SQLite judgment graph](canonical-sqlite-judgment-graph.md) が judgment source-of-truth、domain write API、logical forgetting、hard-delete boundary、deterministic portability、backup / restore を定義する。
- [Optional MCP Feedback Interface](optional-mcp-interface.md#shared-database-home) が user-owned database home、path precedence、file-backed concurrency policy、fail-closed migration、local-only 保証境界を定義する。
- [Decision Wiki pilot migration](decision-wiki-pilot-migration.md) が Li+ / NGR Wiki import の対象境界、identity namespace、provenance、fail-closed 条件、再現手順を定義する。
