# Decision Structure

## 目的と範囲

本書は NGR の Decision Structure における、判断ノード一覧、entry format、edge vocabulary、所有境界、lifecycle の正本である。設計・評価判断を時系列の作業ログではなく、現在の状態と明示的な関係として記録する。

本書は要求仕様、実験プロトコル、実装文書を置き換えない。挙動と検証の正本はそれぞれの文書に残る。Decision Structure entry は判断の現在の結論と境界を記録し、linked requirements と実験文書が実行可能な契約を担う。

## 現在の索引

| Node | State | Current resolution |
| --- | --- | --- |
| [observation-lifecycle-test-exception](https://github.com/Liplus-Project/neuron-graph-rag/wiki/observation-lifecycle-test-exception) | active | #31 の observed-state lifecycle test exception は hash、exclusive-write、no-recompute assertion に限定する。 |
| [ceiling-aware-feedback-adaptation-gate](https://github.com/Liplus-Project/neuron-graph-rag/wiki/ceiling-aware-feedback-adaptation-gate) | active | 新規 feedback-adaptation experiment は baseline relation MRR が 1.0 未満なら strict improvement、1.0 なら全 safety gate を満たす non-regression を要求する。ceiling pass は default や一般化を意味しない。 |
| [single-corpus-real-feedback-validation](https://github.com/Liplus-Project/neuron-graph-rag/wiki/single-corpus-real-feedback-validation) | superseded | [repository-native-controlled-corpus](https://github.com/Liplus-Project/neuron-graph-rag/wiki/repository-native-controlled-corpus) がこの node を supersede する。以後の evaluation は、NGR repository に公開する固定 SHA の controlled corpus を source とし、D1 single-corpus experiment は capacity が増えるまで waiting とする。 |
| [repository-native-controlled-corpus](https://github.com/Liplus-Project/neuron-graph-rag/wiki/repository-native-controlled-corpus) | active | repository-native controlled corpus v2 は、固定 SHA の公開 documentation と本文中の明示的な同一 directory 相対 link だけから、node、doc path、source URL、credited edge identity が相互に分離した development / holdout の各 3-edge path を導出する。v1 は provenance として保持する。これは controlled benchmark であり、外部 corpus への一般化、評価 query、gold、result、既定値変更を含まない。 |

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

- `docs/Decision-Structure.md` は main repository における索引、format、vocabulary、所有境界、lifecycle の正本である。これは docs-owned であり、GitHub Wiki の `Decision-Structure.md` へ mirror する。
- lowercase kebab-case の Decision Structure entry と `_Sidebar.md` は Wiki-only である。docs-to-Wiki synchronization は、`docs/` に対応物がないことを理由にこれらを create、overwrite、delete しない。
- 個別 Wiki entry は、その判断の current state の正本である。GitHub issue、pull request、commit、test output、fixture、gold、manifest、gate、result artifact は、それぞれの所有境界に従う根拠または契約であり、リンクしただけで Decision Structure entry にはならない。
- `requirements.md`、実験文書、fixture、gold、manifest、gate、result artifact は、既存の source-of-truth boundary を維持する。

## Lifecycle

### Add

必要な format を持つ lowercase kebab-case の Wiki entry を新規作成する。current state を本索引に追加し、`_Sidebar.md` に到達可能な Wiki link を追加する。entry に依拠する前に applicable constraints を明記する。

### Update

新しい根拠が underlying decision を置き換えず current resolution を変えるときは、同じ Wiki entry を更新する。`Question` を保ち、state、edge、constraint、related evidence を必要に応じて更新し、本索引も更新する。timeline を追記せず、履歴は Git history に委ねる。

### Supersede

判断そのものが置き換わるときは、新しい entry を作る。新 entry は prior node への `supersedes` edge を持つ。prior entry は replacement を current state として示し、索引の prior row を superseded に更新し、両 page と Sidebar からの到達可能性を残す。prior entry を delete したり、その evidence を successor へ書き換えたりしない。
