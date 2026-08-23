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

1. 文書ノード、metadata、知識確信度を SQLite に永続化できる。
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
30. neural dynamics experiment は development と doc path が重ならない connected holdout、gold、探索空間、最大 variant 数、選択規則、停止規則を結果観測前に固定する。
31. 候補選択は development result だけで行い、relation MRR と negative-control MRR の Pareto gate、worst-cohort MRR、展開数、構造複雑度、variant ID の順で一意に決める。
32. development gate を通る候補がない場合は holdout を開かず既定を変更しない。候補がある場合だけ holdout を一度評価し、cohort 退行、path 不一致、feedback 汚染のいずれかがあれば採用しない。
33. experiment result は gate 不合格、Pareto 支配、holdout 不採用を含む全 variant を上書きせず versioned artifact として保存する。
34. recurrent activation は global inhibition に加え、同じ source の sibling neighbor だけを競合させる local strategy を選択できる。
35. local recurrent strategy は query relevance と active path identity を独立に有効化でき、競合集合ごとに source、path identity、neighbor 数、query relevance、配分前後の message 総量を記録する。
36. local recurrent experiment は production D1 から read-only 取得した新しい development / holdout を旧 development / 開封済み holdout および相互間で分離し、両 provenance と contamination audit を結果観測前に固定する。
37. 旧 development result は family と baseline の探索的根拠だけに使用し、旧 holdout は fixture identifier の重複拒否以外では読み込まない。
38. local recurrent experiment は `current`、`recurrent-balanced`、neighbor / query / path の4 ablationを合わせた6 variantsに固定し、parameter gridを追加しない。
39. development候補はrelation MRRで両baselineを厳密に上回り、direct / negative-control MRRがcurrentから退行せず、全relation pathとfeedback isolationを満たす場合だけ選択する。
40. development候補がある場合だけ、未観測holdoutで`current`、`recurrent-balanced`、選択候補を一度評価する。resultの再実行と上書きを拒否する。
41. local recurrent strategyは未観測holdoutで同じgateを通過した場合だけ既定候補になり、それ以外では`current_positive_additive`を維持する。
42. entry retrievalはgraph競合の外側にzero-hop anchorとして保持でき、競合前後で同一値であることをtrace diagnosticsで検証できる。
43. anchored graph signalは少なくとも1 edgeを通ったmessageだけで構成し、zero-hop seed residualとzero-hop pathを含めない。
44. dense retrievalとgraph propagationを独立に無効化でき、BM25-only ablationではdense encoderとgraph traversalを実行しない。
45. explanationはBM25 / denseのraw・normalized値、競合前後のentry anchor、graphのraw・normalized値、final score、zero-hop / graph path種別を区別して保持する。
46. anchored hybrid experimentはproduction D1からread-only取得した新development / holdoutを、過去39 doc pathsを含む4 fixturesおよび相互間で分離し、provenanceとcontamination auditを結果観測前に固定する。
47. anchored hybrid experimentは`current`、`bm25-only`、BM25+現行graph、BM25+dense anchorのlocal/query local、BM25 anchorのlocalを合わせた6 variantsだけを比較する。
48. development候補はrelation MRRがcurrentを厳密に上回り、direct / negative-controlが退行せず、relation path、feedback isolation、anchor invariant、edge-only graph signalをすべて満たす場合だけ選択する。
49. 候補がある場合だけ未観測holdoutで`current`、`bm25-only`、選択候補を一度評価し、同じgateを通過した場合だけdefault変更候補とする。
50. graph activationは`max`、`none`、`l1_mass`を一般設定として選択でき、zero totalを決定論的に全0へ変換できる。
51. final fusionは既存linearに加え、entry rankとpositive graph nodeだけのgraph rankを使うbottom-centered weighted RRFを選択できる。
52. 各traceはentry / graph rank、entry / graph fusion component、normalization、fusion strategy、RRF k、positive graph node数を保持し、final orderingを機械的に再計算できる。
53. fusion calibration experimentはproduction D1の新しい3-node development / holdoutを、既存7 fixturesの50 unique doc pathsおよび相互間から分離し、provenance、balanced gold、contamination audit、6 variantsを結果観測前に固定する。
54. development候補はrelation MRRをcurrentから厳密に改善し、少なくとも1 relation caseを個別改善し、direct / negative-controlのcohort MRRと全個別rankを退行させず、path、feedback、anchor、edge-only graph、formula auditを満たす場合だけ選択する。
55. development候補がある場合だけ未観測holdoutでcurrentと選択候補を一度評価し、同じgateを通過した場合だけdefault変更候補とする。
56. `search_channels`は同一queryからBM25 lexical laneとanchored edge-only relation laneを独立trace IDで同時返却し、cross-lane final score、combined rank、single winnerを生成しない。
57. lexical laneはBM25だけで順位付けし、dense retrievalとgraph propagationをlane順位へ使用せず、保存hitへgraph pathを持たせない。
58. relation laneはBM25+dense entryをseed選択だけに使い、`anchored_local_competition`で1 edge以上を通過したpositive graph nodeをraw activation降順・node ID昇順で順位付けする。
59. channel provenanceはcallerのchannel自己申告でなく独立trace IDに保存し、default config の`record_success`はlexical traceでedgeを変更せず、relation traceでは保存済みcredited pathだけを強化する。
60. 同一nodeが両laneに現れる場合も各rankと説明を保持し、片方のtraceに保存されていないnodeへのfeedbackをatomicに拒否する。
61. independent-channel experimentはproduction D1からread-only取得した相互disjointな2-node / 1-edge developmentとholdoutを、既存9 fixturesの全node pathから分離し、各splitの4-case hard gate、provenance、contamination audit、lane規則、feedback規則、停止規則を結果観測前に固定する。
62. developmentでlane parity、relation個別改善、edge-only path、独立trace、edge不変、feedback帰属、cross-lane拒否、決定性の全hard gateを通過した場合だけholdoutを一度開き、同じgateを全通過した場合だけ`search_channels`をvalidatedと記録する。
63. blind channel-selection experimentはv1 fixture / gold / manifest / evaluator / runner / development resultをbyte hashで固定し、既存`search()`、`search_channels()`、default、feedback、v1 artifactを変更しない。
64. blind packetはopaque case ID、query、lane semantics、独立trace、lane内rank、node本文とsource metadata、raw / projected relation path、agreementだけを含み、cohort、intended channel、期待node / rank / path、gate、過去結果、lane scoreを含めない。
65. judge responseはpacketだけを受けるfresh agent 3体が独立生成し、trace / node所属を検証した後に`(channel, node)`のmajorityで集約する。actual LLM callはparent orchestratorだけが行い、core、runner、CIはjudgeを呼ばない。
66. relation pathはraw stepを保存したままendpoint / edge typeへ射影して照合し、zero-hopを拒否する。judge raw response、parse結果、model、agent type、実行時刻をimmutable artifactへ保存する。
67. development packet / response / resultはresult-free commitのpush後に各一度だけ生成し、全12 gate通過時だけholdout packetを一度生成して異なるfresh judge 3体で判定する。全観測artifactは上書きと再生成を拒否する。
68. development gate通過前のholdoutではpacket生成、`search_channels()`実行、judge提示、v2 gold照合を禁止する。byte hash、schema、既存contract testの非表示process readはselectionに使わず、holdout open countへ含めない。
69. v1 / v2 frozen text artifactのbyte hash監査はraw checkout bytesを最初に照合する。raw不一致時は全改行がLFまたはCRLFの一種類で、もう一方へのexact変換だけが固定hashと一致する場合に限ってcheckout変換を許可し、本文差分、mixed newline、bare CRを拒否する。
70. node-first blind experimentはv1 / v2 manifest、prompt、packet、judge artifact、result、source、runner、tests、experiment docsをraw-firstかつLF / CRLFだけを許可するhash監査で固定し、v2 invalid responseをretry、補完、再集約しない。
71. v3はv1のdirect / directional-negative queryを維持し、relation queryだけedge先targetを自然文で明示する固定overrideを使う。stage packetは4 caseを保持するが、judge用packetとresponseは一つのopaque caseだけを含む。
72. developmentは4 case x fresh 3 judgesの12 independent invocationsとし、各agent contextを再利用せず、repo、web、tool、gold、他case、他judge、prior responseを渡さない。actual LLM invocationはparent orchestratorだけが行う。
73. v3集約は`node_id`の2/3 majorityを採否の正本とし、同じnodeへ複数laneから投票した場合もnode correctnessを満たす。channel voteはtrace provenanceと分布だけに使い、gold channel gateを持たない。
74. relation traceを選んだresponseはselected nodeのraw pathを保存し、endpoint / edge typeへの射影が固定relation pathと一致してzero-hopでないことを監査する。feedbackは実行せず、lexical / relation traceの強化provenanceだけを記録する。
74a. node-first captureの前にmanifest、stage packet、およびstageが参照する全case packetのcase ID、path、実byte hashを照合する。いずれかが不一致ならjudge起動、raw response read、capture artifact write、aggregationを開始せず、unassessedのまま停止する。
75. trace-credited feedback adaptation experimentは、同一のfrozen corpus、config、query schedule、limit、時刻規則でcontrolとtreatmentを比較し、controlはfeedbackを記録してedge mutationを適用せず、treatmentだけがrelation traceのcredited pathを強化する。
76. feedback adaptationのdevelopment / holdout、feedback event、score query、expected path、gate、registered run count、hash、contamination audit、exclusive outputを結果観測前に固定し、development全gate通過時だけholdoutを一度開く。
77. feedback adaptation reproductionはprior resultを選択入力にせず、prior fixtureとの識別子だけのcontamination auditを行う。raw relation stepは比較前に`source_id`、`target_id`、`edge_type`だけへ射影し、runtime fieldを含むsynthetic testでpath identityをfreeze前に検証する。
78. 新規 feedback-adaptation experiment の primary relation gate は、baseline relation MRR が 1.0 未満ならtreatmentのstrict improvement、baseline relation MRR が 1.0 ならtreatmentのnon-regressionを要求する。いずれも endpoint/type projected path、direct lexical / directional-negative controls、credited-only mutation、deterministic replay、contamination、immutable output を含む全 safety gate を必須とする。ceiling case のnon-regression pass は追加の順位改善を示せないことを記録するだけで、generalization、default変更、production採用を許可しない。
79. repository-native controlled corpus v3 の engine-backed trajectory experiment は、source commit、split / cluster identity、explicit-link edge、0 / 1 / 3 / 10 feedback schedule、query、used node、credited path、control / treatment、gate、manifest hash、exclusive output を観測前に固定する。control は relation trace と used node を記録して edge を変更せず、treatment だけが同じ schedule の relation trace ID を `record_success` に渡す。headroom は 0 から 10 で厳密改善し途中 checkpoint で退行せず、control case と ceiling case も退行せず、credited edge 以外が変化しない場合だけ development gate を通過する。development 全 gate 通過時だけ holdout を一度開き、観測後は evaluator、fixture、gold、schedule、manifest、gate、docs を変更しない。
80. sibling relation feedback normalization は、明示的に有効化された candidate config でのみ、relation trace の credited edge を強化し、その edge と同じ source から出る未 credit sibling だけを局所的に正規化できる。lexical trace、zero-hop、未関係 source、credited sibling は変更しない。candidate は synthetic isolation test と result-free development / holdout 相当の relation、direct、lexical、negative-control gate を通過するまで default にしない。
81. sibling normalization controlled evaluation は、評価対象 source commit / hash、相互に identity-disjoint な development / holdout cluster、明示 edge、baseline `0.0` / treatment `1.0`、query、used node、credited path、mutation scope、rollback、係数 / 時刻 schedule、hard gate、exclusive output を観測前に固定する。実 `NeuronGraphRAG` の `search_channels` relation trace ID を `record_success` に渡し、headroom strict improvement、ceiling・direct・lexical・directional-negative non-regression、path・mutation・atomicity・determinism の全 development gate 通過時だけ holdout を一度開く。観測後は protocol artifact と docs を変更せず、既定値と external D1 claim を変更しない。
82. frozen evaluation の historical source hash は、manifest path の初回追加 commit または manifest が明示する lowercase full 40-hex source / baseline / prior commit の exact blob bytes に対して検証する。後続の committed manifest rewrite、mutable ref / revision expression、未知 commit、非 ancestor commit、manifest bytes差、欠落 path、hash 不一致を fail closed にし、同名 path の current working tree を過去の evidence として扱わない。既存 protocol が明記する raw-first LF / CRLF whole-file alternate だけを維持し、本文差、mixed newline、bare CR、その他の byte 差を拒否する。
83. soft-start feedback reinforcement は明示 opt-in の relation-only candidate とし、最初の新規 `used` で通常 bounded increment の固定 ratio 分だけを credited path へ適用する。最初の独立 `confirmed` は同じ schedule の残量を一回分の通常 increment まで補い、後続 confirmation は固定 decay ratio で加算する。used 時は sibling normalization を行わず、confirmation の actual delta だけを同一 source の uncredited sibling へ配分する。duplicate、lexical、zero-hop、別 source、uncredited edge、negative outcome は変更せず、candidate mechanics の合格だけで default や local serving policy を変更しない。
84. soft-start snapshot evaluation は、local source database を read-only URI と SQLite backup API で一度だけ transaction-consistent snapshot へ複製し、同じ snapshot のfresh clone上で `control`、`used_q3_s1`、`confirmed_r05_s1`、`soft_start_r025_r05_s1` の固定4 armを比較する。query、public node / edge identifier、outcome、3回のfresh trace schedule、checkpoint、metric、hard gate、exclusive output、snapshot / protocol hashを登録result生成前に固定する。source database、live config、snapshot本体、private本文、absolute private pathをpublic artifactへ含めず、developmentを一度だけ実行し、全hard gate通過時だけholdoutを一度開く。不支持または判定不能を保存してもprotocolを調整せず、local cutover、library default、external corpus、production qualityへ自動で一般化しない。
85. baseline-aware soft-start snapshot evaluation は、v1のprotocol、gate、observed result、private snapshotを変更、再実行、再集計、入力再利用せず、fresh snapshot、新規namespace、新規output、v1 observed developmentと異なるcredited edge identityを使う。各relation caseのinitial weight、reinforced count、evidence count、confirmation countを結果前に登録し、q3/s1のfirst mutationを`max(1, quorum - initial evidence count)`で導出する。導出不能、baseline不一致、event budget内のquorum capacity不足はregistered resultを作らずfailure reportで停止する。v2 protocolはregistered output不在のfreeze-only PRで固定し、そのsquash merge後の別Issueでdevelopmentを一度だけ実行する。全8 hard gate通過時だけholdoutを一度開き、支持結果もlocal cutover候補に限定してsource database、live config、library defaultを変更しない。
86. outcome-driven feedback deactivation はsoft-startと同時にだけ有効化できるdefault-off candidateとする。provisional / confirmationごとにcredited加算と同時発生したsame-source sibling normalization減算を一つのsigned mutation journalへ永続化し、因果帰属できる`corrected` / `rolled_back`だけが未反転contributionを基礎weight未満へ下げずexact reversalする。`superseded`はedge、evidence、trace、outcomeを削除せずrelation edgeをdormantにして通常activationから除外し、同じ保存済みcredited pathの後続`confirmed`で再活性化する。duplicate、retry、restart、transaction failure、lexical、zero-hop、別source、uncredited edge、因果帰属不能outcomeは二重減算または局所外mutationを行わない。
87. outcome-driven deactivation evaluation はcontrol / candidate、`corrected` / `rolled_back` / `superseded`、exact credited / sibling inverse、baseline floor、dormancy / reactivation、rank / locality、source isolation、exclusive outputを結果観測前に固定する。protocolはregistered output不在のfreeze-only PRで固定し、そのsquash merge後のsuccessor Issueでdevelopmentを一度だけ実行する。全hard gate通過時だけholdoutを一度開き、観測前後にquery、case、schedule、metric、gate、default、live configを変更しない。
88. NGR 自身の新規 judgment graph は SQLite の stable identity、revision、lifecycle、provenance、typed relation を machine-readable 正本とする。add / update / supersede / archive / restore / hard-delete candidate は raw SQL でなく atomic domain API を通し、stale revision、dangling relation、部分更新、二重 successor を fail closed にする。archive は通常 retrieval から外す論理的忘却、hard delete は履歴参照のない archived candidate だけに許す物理削除として分離する。current graph の deterministic export / atomic import と SQLite backup / integrity-checked restore を維持し、既存 Wiki entry の本番移行は fixture 検証後に分離する。
89. Li+ / NGR Decision Structure Wiki pilot は各 repository の index が列挙する entry だけを専用の新規 SQLite へ取り込み、repository namespace 付き identity、page 本文、Wiki URL、repository、取得 commit、source state、typed relation を保持する。duplicate identity、unknown relation target、parser ambiguity、partial publication、既存出力の上書きを fail closed にし、SQLite / supersession integrity、deterministic export、backup を検証する。Wiki、既存検索 DB、凍結済み feedback 実験 DB は変更せず、本 pilot だけで正本を切り替えない。
90. judgment 専用 read API は `search_judgments`、`get_judgment`、`traverse_judgments` を提供し、current revision、lifecycle、statement、rationale、provenance、typed relation を返す。search は judgment の既存 node projection に同じ lexical / dense scorer と有効 weight を適用し、既定で active のみ、明示指定時だけ archived を含め、repository namespace で絞り込める。traversal は relation type、incoming / outgoing / both、1 以上の有限 hop を受け、cycle-safe な hop 優先・stable identity 順を維持する。三操作と対応 MCP tool は成功時・失敗時とも judgment、revision、relation、retrieval trace、feedback、node、edge、activation を永続変更せず、MCP は read-only annotation、未知 field を拒否する schema、既存 error envelope を持つ。既存 `search`、feedback、`write_judgment`、library default は変更しない。
75. v3 implementation、prompt、manifest、query override、schema、集約、path audit、hash規則、gate、stop rule、testsをresult-free commitでpushした後、development stage / 4 case packet / 12 responses / resultを各一度だけ生成する。
76. development全12 gate通過時だけholdout stageを一度生成し、異なるfresh 12 judgesで同じgateを評価する。packet、response、resultの上書き、観測後の規則変更、実LLM品質値のCI再生成を拒否する。

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
- [Neural dynamics experiment](neural-dynamics-experiment.md) が development / holdout 分離、固定探索空間、候補選択、単一 holdout 開封、停止規則を定義する。
- [Local recurrent competition experiment](neural-dynamics-local-competition-experiment.md) が新規D1 subgraph、contamination audit、query / path ablation、二baseline gateを定義する。
- [Anchored BM25 and graph hybrid experiment](anchored-bm25-graph-hybrid-experiment.md) がzero-hop anchor、edge-only graph signal、BM25 ablation、新規D1 split、単一holdout gateを定義する。
- [Anchored fusion calibration experiment](anchored-fusion-calibration-experiment.md) がgraph normalization、bottom-centered RRF、新規D1 split、個別case gateを定義する。
- [Independent retrieval channels experiment](independent-retrieval-channels-experiment.md) が非融合lane、独立trace provenance、feedback帰属、4-case hard gate、単一holdout開封を定義する。
- [Blind LLM channel selection experiment](blind-llm-channel-selection-experiment.md) がanswer-free packet、fresh judge分離、majority集約、path射影、result-free freeze、conditional holdoutを定義する。
- [Node-first blind selection experiment](node-first-blind-selection-experiment.md) がsingle-case invocation、node-majority、channel provenance分離、v1 / v2不変監査、conditional holdoutを定義する。
- [Trace-credited feedback adaptation experiment](feedback-adaptation-experiment.md) がcontrol / treatmentの因果比較、result-free freeze、conditional holdoutを定義する。
- [Trace-credited feedback adaptation reproduction experiment](feedback-adaptation-reproduction-experiment.md) が新規D1 split、prior-result非参照、endpoint/type-only path projection、conditional holdoutを定義する。
- [Engine-backed feedback trajectory experiment](engine-backed-feedback-trajectory-experiment.md) が repository-native controlled corpus v3 上の 0 / 1 / 3 / 10 feedback trajectory、実 relation trace、credited-only mutation、result-free freeze、conditional holdout を定義する。
- [Sibling relation feedback normalization](sibling-relation-feedback-normalization.md) が opt-in candidate の局所 sibling 正規化、trace isolation、default 変更前の検証境界を定義する。
- [Sibling normalization controlled evaluation](sibling-normalization-controlled-evaluation.md) が repository-native corpus、result-free hash freeze、実 relation trace feedback、mutation / rollback gate、conditional holdout を定義する。
- [Historical source verification](historical-source-verification.md) が frozen manifest path の初回追加 commit、明示 full source commit ID、exact blob、ancestor、path、newline portability、fail-closed 境界を定義する。
- [Confirmed-outcome feedback reinforcement](confirmed-outcome-feedback-reinforcement.md) が confirmed-only と soft-start の明示 policy、永続 schedule、transaction、receipt、default-preserving boundary を定義する。
- [Soft-start snapshot evaluation](soft-start-snapshot-evaluation.md) が transaction-consistent private snapshot、固定4 arm、result-free freeze、privacy、one-time development、conditional holdout、local cutover境界を定義する。
- [Outcome-driven feedback deactivation](outcome-driven-feedback-deactivation.md) がsigned contribution journal、exact reversal、dormancy / reactivation、default-off境界、result-free freezeを定義する。
- [Baseline-aware soft-start snapshot evaluation](baseline-aware-soft-start-snapshot-evaluation.md) がfresh baseline stateからのq3 boundary導出、v1 evidence isolation、capacity preflight、新規one-time result境界を定義する。
- [Canonical SQLite judgment graph](canonical-sqlite-judgment-graph.md) が judgment source-of-truth、domain write API、logical forgetting、hard-delete boundary、deterministic portability、backup / restore を定義する。
- [Decision Wiki pilot migration](decision-wiki-pilot-migration.md) が Li+ / NGR Wiki import の対象境界、identity namespace、provenance、fail-closed 条件、再現手順を定義する。
