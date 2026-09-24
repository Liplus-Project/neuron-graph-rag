# 通常 wheel と checkout 実験コードの境界

通常の `neuron-graph-rag` wheel には、公開ライブラリ、既存 CLI、optional MCP adapter、および利用ガイドから直接呼ぶ opt-in 検索 API を含める。実験専用 Python module は checkout に残し、wheel からは除外する。検索と feedback の既定設定は変わらない。

## 配布対象

`neuron_graph_rag` package では次の22 moduleを明示的に配布する。

| 用途 | module |
| --- | --- |
| 公開 API と内部依存 | `__init__`, `dynamics`, `engine`, `exclusion_intent`, `feedback`, `judgments`, `models`, `ontology`, `precision_control`, `retrieval`, `storage` |
| CLI と既存3コマンド | `__main__`, `cli`, `benchmark`, `d1_fixture`, `evaluation`, `sample` |
| optional MCP adapter の依存 | `config_provenance`, `database_home`, `evidence_feedback` |
| 利用ガイドで直接案内する opt-in API | `semantic_retrieval`, `cpu_shortlist_retrieval` |

`neuron_graph_rag_mcp` の `__init__`, `__main__`, `server` も配布する。MCP SDK は従来どおり `.[mcp]` の任意依存であり、core の必須依存にはならない。CPU shortlist と semantic retrieval のモデル依存も、各 API を明示的に使うときだけ別途導入する。

`benchmark`, `d1_fixture`, `evaluation` は実験に見える名前だが、通常 CLI の `benchmark` / `eval` が直接 import するため保持する。`semantic_retrieval` と `cpu_shortlist_retrieval` は静的な runtime import root からは到達しないが、[意味検索](semantic-retrieval.md)と[CPU shortlist](cpu-shortlist-retrieval.md)の利用ガイドが直接 import する API なので保持する。package root、CLI、MCP からの静的 import と、それらの動的 import を監査した。残す module から実験専用 module への動的 import はない。Python の任意の外部利用コードによる import までは静的監査で証明できない。

## checkout 専用コード

上表以外の `src/neuron_graph_rag/*.py` は通常 wheel から除外する。対象は `cross_encoder_precision_*`, `github_retrieval_parity*`, `real_task_shadow*`, `source_grounded_relation_observation*` などの凍結済み実験・観測・再現コードと、`decision_wiki_import`, `github_source`, `intent_aware_*`, `rank_observation_*` など checkout の tool / 実験から使う補助 module を含む。除外判定は名前の接尾辞ではなく、上記 runtime import と利用導線から作った明示的な配布リストで行う。将来追加した module も、配布リストへ明示的に加えるまでは wheel に入らない。

既存の実験 source file は移動・改名・編集しない。frozen manifest が source path と hash を証拠として登録しており、移動や内容変更は過去の再現契約を変えるためである。実験ランナーとテストは引き続き checkout 上の legacy import path を使う。新規実験専用コードは repository 直下の `experiments/` に置き、通常 wheel に必要な API への昇格は別途判断して配布リストと隔離 wheel テストを更新する。

## 配布差分と検証

2026-09-24 の `src` 内 `.py` source 比較では、変更前は107 module / 2,866,424 bytes、変更後の通常 wheel 対象は25 module / 408,817 bytes。82 module / 2,457,607 bytes（source bytes の約85.7%）を除外する。これは source file の合計であり、圧縮後 wheel サイズや実行時メモリ量ではない。

`tests/test_runtime_wheel.py` は隔離ディレクトリで通常 wheel をビルド・インストールし、file list、wheel からの import、CLI `demo` / `eval` / `benchmark`、opt-in API を検証する。MCP SDK のある環境では、同じ wheel の MCP tool 一覧、検索、source-use feedback、outcome も検証する。checkout の既存 test suite は実験 module の legacy import path を継続して検証する。
