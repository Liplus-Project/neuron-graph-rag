# Decision Structure

## 目的と範囲

本書は NGR の Decision Structure における、判断ノード一覧、entry format、edge vocabulary、所有境界、lifecycle の正本である。設計・評価判断を時系列の作業ログではなく、現在の状態と明示的な関係として記録する。

本書は要求仕様、実験プロトコル、実装文書を置き換えない。挙動と検証の正本はそれぞれの文書に残る。Decision Structure entry は判断の現在の結論と境界を記録し、linked requirements と実験文書が実行可能な契約を担う。

## 現在の索引

| Node | State | Current resolution |
| --- | --- | --- |
| [observation-lifecycle-test-exception](https://github.com/Liplus-Project/neuron-graph-rag/wiki/observation-lifecycle-test-exception) | active | #31 の observed-state lifecycle test exception は hash、exclusive-write、no-recompute assertion に限定する。 |
| [ceiling-aware-feedback-adaptation-gate](https://github.com/Liplus-Project/neuron-graph-rag/wiki/ceiling-aware-feedback-adaptation-gate) | active | 新規 feedback-adaptation experiment は baseline relation MRR が 1.0 未満なら strict improvement、1.0 なら全 safety gate を満たす non-regression を要求する。ceiling pass は default や一般化を意味しない。 |
| [evidence-gated-local-feedback-reinforcement](https://github.com/Liplus-Project/neuron-graph-rag/wiki/evidence-gated-local-feedback-reinforcement) | active | relation edge ごとの独立 success trace が固定 quorum に達するまで serving weight を変えず、到達後は既存 bounded reinforcement と same-source sibling normalization を一回ずつ適用する。candidate は default-preserving であり、採用値と一般化を主張しない。 |
| [confirmed-outcome-feedback-reinforcement](https://github.com/Liplus-Project/neuron-graph-rag/wiki/confirmed-outcome-feedback-reinforcement) | superseded | confirmed-only candidate の mechanics と証拠は保持するが、利用直後の小さな適応を残す [soft-start-feedback-reinforcement](https://github.com/Liplus-Project/neuron-graph-rag/wiki/soft-start-feedback-reinforcement) が後続比較の現在候補としてこの判断を supersede する。 |
| [outcome-driven-feedback-deactivation](https://github.com/Liplus-Project/neuron-graph-rag/wiki/outcome-driven-feedback-deactivation) | evaluating | 因果帰属できるnegative outcomeだけがsoft-start contributionと同一単位のsibling normalizationを正確に戻し、superseded edgeは削除せずdormant / reactivationで扱う。mechanicsとresult-free freezeを先に固定し、観測と採用判断はsuccessor Issueへ分離する。 |
| [frozen-output-round-trip-integrity](https://github.com/Liplus-Project/neuron-graph-rag/wiki/frozen-output-round-trip-integrity) | active | result-free evaluation は canonical gate ID array を唯一の順序正本とし、freeze 前に登録外 placeholder と temporary output で実 writer から実 verifier への非アルファベット順 round-trip を証明する。observed result は exclusive creation し、array の完全性、順序、重複なし、全 gate pass を検証する。 |
| [historical-freeze-verification-boundary](https://github.com/Liplus-Project/neuron-graph-rag/wiki/historical-freeze-verification-boundary) | active | historical source hash は current working tree ではなく、frozen manifest が登録した source commit の exact bytes に対して検証し、repository の正当な後続変更と過去証拠の不変性を時間軸で分離する。 |
| [longitudinal-feedback-adaptation](https://github.com/Liplus-Project/neuron-graph-rag/wiki/longitudinal-feedback-adaptation) | active | longitudinal feedback-adaptation は、repository-native controlled corpus v3 の相互に独立した cluster と、その source 文書に明記した 0、1、3、10 credit ceiling を用いる。corpus phase は query、gold、schedule、runner、gate、manifest、result、既定値を定義・変更しない。 |
| [single-corpus-real-feedback-validation](https://github.com/Liplus-Project/neuron-graph-rag/wiki/single-corpus-real-feedback-validation) | superseded | [repository-native-controlled-corpus](https://github.com/Liplus-Project/neuron-graph-rag/wiki/repository-native-controlled-corpus) がこの node を supersede する。以後の evaluation は、NGR repository に公開する固定 SHA の controlled corpus を source とし、D1 single-corpus experiment は capacity が増えるまで waiting とする。 |
| [repository-native-controlled-corpus](https://github.com/Liplus-Project/neuron-graph-rag/wiki/repository-native-controlled-corpus) | active | repository-native controlled corpus v2 は、固定 SHA の公開 documentation と本文中の明示的な同一 directory 相対 link だけから、node、doc path、source URL、credited edge identity が相互に分離した development / holdout の各 3-edge path を導出する。v1 は provenance として保持する。これは controlled benchmark であり、外部 corpus への一般化、評価 query、gold、result、既定値変更を含まない。 |
| [soft-start-feedback-reinforcement](https://github.com/Liplus-Project/neuron-graph-rag/wiki/soft-start-feedback-reinforcement) | active | 最初の credited `used` に通常 bounded updateの小さなprovisional fractionを適用し、最初の独立`confirmed`がremainder、後続confirmationがgeometric decayを適用する。v1 snapshot評価の不支持を保持し、baseline-aware successorはfresh initial evidenceからq3 first mutationを導出する。v2 freeze-only PRとsquash後のobserved registrationを分離し、development全gate通過時だけholdoutを一度開く。source database、live config、defaultを変更しない。 |
| [real-task-shadow-feedback-validation](https://github.com/Liplus-Project/neuron-graph-rag/wiki/real-task-shadow-feedback-validation) | active | confirmed-triggered reinforcement の次の実証は、客観的 outcome を事前登録した Codex 実タスクの non-serving shadow A/B とし、検索品質、誤強化率、Agent end-to-end 効果を default 採用前に測る。 |
| [github-rag-mcp-replacement-compatibility](https://github.com/Liplus-Project/neuron-graph-rag/wiki/github-rag-mcp-replacement-compatibility) | active | public GitHub repository一つのread-only snapshotをNGR local indexへ接続する。github-rag-mcp `search` の保存済み raw capture と source URL、根拠を比較する。共有 source identity を確認しても最小 doc 検索 path の候補に限り、production github-rag-mcp、MCP authentication / transport、remote deployment、default変更は含まない。 |
| [sqlite-canonical-judgment-graph](https://github.com/Liplus-Project/neuron-graph-rag/wiki/sqlite-canonical-judgment-graph) | active | NGR 自身の判断構造は SQLite の stable identity、revision、lifecycle、typed relation を machine-native 正本とし、raw SQL でなく atomic domain API で変更する。Wiki は移行 fixture の検証後に optional generated view へ下げる。 |
| `neuron-graph-rag:user-owned-shared-database-home` | active | optional MCP server は明示 path を保ったまま `~/.ngrdb/knowledge.db` を client-neutral な user-owned 既定正本とし、file-backed SQLite の WAL / bounded busy timeout と fail-closed migration で同一端末・同一ユーザー内の共有を支える。 |

## Entry format

各 Decision Structure entry は lowercase kebab-case の Wiki-only page とし、次の section を持つ。

1. `Question` - 解く判断対象。
2. `Current resolution` - 現在の状態を直接かつ境界付きで記す。
3. `Edges` - 他の Decision Structure node との型付き関係。関係がなければ `none` とする。
4. `Background` - 判断が必要になった根拠と背景。
5. `Constraints` - 不変条件と明示的な除外。
6. `Conclusion` - 適用範囲を含む運用上の結論。
7. `Related` - issue、pull request、document、code などの支持根拠。Decision Structure edge にはしない。

前提が変わり得るときは、短い再評価条件を追加してよい。Git history が過去の本文を保持するため、entry を時系列ログにしない。

## Edge vocabulary

`Edges` では次の有向 label だけを使う。

- `depends on` - この結論は target が真であり続けることを前提にする。
- `refines` - target を置き換えずに、その範囲を狭めるか運用可能にする。
- `supersedes` - この結論が target を現在の状態として置き換える。
- `conflicts with` - 二つの現在の結論は同時に適用できない。
- `informs` - target に関連する判断材料を与えるが、target を制約しない。

edge target は Decision Structure node slug とする。外部資料は `Edges` でなく `Related` に置く。

## Source-of-truth boundary

- 新規に domain API から登録された judgment graph の machine-readable 正本は SQLite である。詳細契約は [Canonical SQLite judgment graph](canonical-sqlite-judgment-graph.md) に置く。
- SQLite domain API で直接登録し Wiki page を持たない judgment は、stable identity を索引の Node として記す。`neuron-graph-rag:user-owned-shared-database-home` は Issue #123 の判断を SQLite 正本へ登録した node である。
- 既存 Wiki entry は検証済み import が行われるまで従来の正本境界を維持する。本変更だけで本番 Wiki entry を自動移行または削除しない。

- `docs/Decision-Structure.md` は main repository における索引、format、vocabulary、所有境界、lifecycle の正本である。これは docs-owned であり、GitHub Wiki の `Decision-Structure.md` へ mirror する。
- lowercase kebab-case の Decision Structure entry と `_Sidebar.md` は Wiki-only である。docs-to-Wiki synchronization は、`docs/` に対応物がないことを理由にこれらを create、overwrite、delete しない。
- 個別 Wiki entry は、その判断の current state の正本である。GitHub issue、pull request、commit、test output、fixture、gold、manifest、gate、result artifact は、それぞれの所有境界に従う根拠または契約であり、リンクしただけで Decision Structure entry にはならない。
- `requirements.md`、実験文書、fixture、gold、manifest、gate、result artifact は、既存の source-of-truth boundary を維持する。
- frozen source / artifact hash の source of truth は、manifest path の初回追加 commit または manifest が明示する lowercase full 40-hex source commit の exact blob bytes である。後続の committed manifest rewrite や mutable ref は信頼境界を動かさず、current working tree の同名 path は後続 evolution の現在状態であって historical evidence ではない。

## Lifecycle

### Add

必要な format を持つ lowercase kebab-case の Wiki entry を新規作成する。current state を本索引に追加し、`_Sidebar.md` に到達可能な Wiki link を追加する。entry に依拠する前に applicable constraints を明記する。

### Update

新しい根拠が underlying decision を置き換えず current resolution を変えるときは、同じ Wiki entry を更新する。`Question` を保ち、state、edge、constraint、related evidence を必要に応じて更新し、本索引も更新する。timeline を追記せず、履歴は Git history に委ねる。

### Supersede

判断そのものが置き換わるときは、新しい entry を作る。新 entry は prior node への `supersedes` edge を持つ。prior entry は replacement を current state として示し、索引の prior row を superseded に更新し、両 page と Sidebar からの到達可能性を残す。prior entry を delete したり、その evidence を successor へ書き換えたりしない。
