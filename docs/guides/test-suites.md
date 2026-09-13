# テスト選択の要件と実行方法

## 範囲と制約

Issue #210 のローカル開発用実行経路。凍結済み protocol、manifest、既存テスト、fixtures、実験証拠の bytes を維持する。歴代版の物理的共通化、実クエリ観測、共有 DB 変更は行わない。

`.github/workflows/ci.yml` は凍結 manifest の hash 対象で、source-grounded v3 のテストは現在のファイルを直接検証する。したがって既存 CI の全件 discover と whole-module probe を維持する。**今回、通常 CI の高速化は未実施**。追加 workflow や環境変数で既存 discover を迂回しない。成果は明示的な局所選択と分類漏れ検出であり、CI 実行時間削減ではない。

## 実行

repository root で CI と同じ Python 3.11 を使用し、先に `python -m pip install -e .` を実行する。パッケージ自体の対応範囲と凍結結果の再現条件は異なり、他版での float 比較は #193 の既知制約がある。optional MCP の検証には `python -m pip install -e '.[mcp]'` が必要。未インストール時の skip は合格した MCP 検証件数ではない。

```bash
python tools/run_tests.py normal
python tools/run_tests.py experiments
python tools/run_tests.py all
```

`-v` は各テスト名を表示する。`--list` は import/実行せず分類検証と選択モジュール一覧のみを表示する。実行時の最終 JSON は選択数、実行数、skip、失敗、エラーと import を含む経過秒数を示す。件数は AST のメソッド数ではなく unittest がロードした件数。

`all` は従来の `python -m unittest discover -s tests -v` と同じfull discoverを使う。v5 retrieval parity観測後は、alphabetical-firstの専用routerが元v5 test moduleの正規import名だけを対象に、元loaderで11件を確認してからfreeze commitのtemporary checkoutへroutingする。通常のfull discoveryと選択runnerはいずれもこのrouterを先に読み、元11件の全件成功を必須にする。その他のmoduleは元loaderへ無変更で委譲し、現rootでは観測専用testを実行する。他のproduct / shared / historical testはdiscover結果から除外しない。手動全件監査にはさらに CI と同じ `python tools/probe_source_grounded_relation_observation_v3.py --root .` と `python -m neuron_graph_rag eval` を実行する。現行 CI が PR ごとにこの全件経路を実行し続ける。

## 分類と変更時の選択

分類の単一の定義は `tools/test_suites.py`。各モジュールを一度だけ登録し、分類理由を併記する。名前のパターンで追加テストを自動除外しない。

| 役割 | 内容と理由 | 経路 |
| --- | --- | --- |
| product | 公開API、永続化、検索、feedback、CLI/MCPの直接回帰 | normal、all |
| shared | 製品engine、選択、監査、共通runtimeを含む混在モジュール。実験名でも通常実行に残す | normal、experiments、all |
| historical | 凍結cross-encoder各版の再現・監査 | experiments、all |

通常の局所確認は `normal`。cross-encoder 実験関連変更は `experiments` で歴代全版と shared を検証する。版間依存を自動推定して一版だけを選ぶ機能は提供しない。engine、検索、共有runtime、共通監査、fixture、依存関係、runner/分類自身の変更や影響範囲が不明な変更では `all` を使う。**normal は共有runtime変更時の all の代替ではない**。製品変更にも歴代実験との依存があり、製品だけで十分とは主張しない。CI の全件検証は引き続き必要。

起動時、`tests/` 以下の `test*.py` を再帰列挙し、未分類・削除後の古い登録・重複登録を検出したら終了コード 2 で停止する。新規テストは分類理由を確認して明示登録する。検証自体を `tests/test_suite_selection.py` に置くため、既存 CI の discover も分類漏れを失敗として扱う。分類によるファイル移動や既存テストの編集は不要。

## 計測

ローカル単発の経過時間は環境・キャッシュで変動する。CI 高速化や関数集約による高速化の証拠にはしない。

2026-09-06、Windows、Python 3.11.15、既存 `.venv` (MCP extra 有効)、逐次実行。起動時の import を含む runner の経過秒数。

| 経路 | モジュール | ロード件数 | 実行件数 | skip | 経過秒 | 結果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| normal | 50 | 371 | 371 | 0 | 117.402 | 成功 |
| all | 81 | 767 | 767 | 0 | 304.200 | 成功 |
| experiments | 68 | 677 | 未実行 | — | 未計測 | 一覧・ロードのみ。all 内で対象テストを検証 |

この単発比較では normal は all より 186.798 秒短い。選択しない歴代テスト 396 件分の局所実行時間差であり、テスト自体や CI が高速化したという意味ではない。既存 tracked ファイルの差分はなく、追加は runner・分類・検証・本書の 4 ファイルのみ。

先行する Python 3.14.5 (MCP extra なし) の normal 実行は 371 件、skip 21、86.495 秒、`test_real_corpus_benchmark.test_checked_result_matches_frozen_inputs` が 1 件失敗した。これは成功値から除外し、再現条件の証拠として残す。同テストを Python 3.11 で単独実行すると成功し、上記 normal 全体も成功した。凍結期待値を変更して解消していない。
