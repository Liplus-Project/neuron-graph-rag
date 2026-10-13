# 過去評価の runtime バイト分離

## 要件

`src/neuron_graph_rag/storage.py` と `src/neuron_graph_rag/judgments.py` は、1 つの successor が複数の active な predecessor を置き換えられるように変更する（#303）。この 2 ファイルは `tests/fixtures/github_retrieval_parity_v{2,3,4,5}.manifest.json` の `runtime_sha256` に登録されている。登録済みの evaluator、tests、manifest、証跡、ハッシュ値は変更しない。検証も省略しない。v5 の development 観測は済んでいる（#228）ので、登録済みの評価は再実行しない。

登録ハッシュは次の 2 つで、どちらも commit `b78dbcaf0167cc6975428ccce662b8228a88cfe2` の LF バイトと一致する。

| path | SHA-256 | byte |
|---|---|---|
| `src/neuron_graph_rag/storage.py` | `3aaf1fb2283ac97486fc9cab34a44676650aec5b6e1ccb8e3100132efc8459d0` | 106,345 |
| `src/neuron_graph_rag/judgments.py` | `cfc21058f63fcd665e8fa233cdcd01a8dfe44f4241e867886155eddcc4fac3f7` | 40,847 |

## 方式

[過去評価の README バイト分離](historical-readme-verification.md)（#295 / PR #296）の bridge を一般化する。`tools/historical_readme_bridge.py` の `ARCHIVED_SLOTS` は、既知の path と既知の SHA-256 の組（slot）だけを保存済みの原バイトへ解決する。slot は README と上の 2 ファイルの 3 つで、原バイトは `tools/historical_readme/` に置く（runtime の 2 つは import されないよう `.py.orig` とする）。原バイトは長さと SHA-256 の両方で照合する。

slot の解決は group ごとに分ける。

- `readme` group: 従来どおり。CE v1–v16 と parity v1 の 21 module が README slot だけを解決する。registry は `registered-assets.json`。
- `judgment-runtime` group: parity v2、v3、v4、v5 observation の 4 module（原テスト 8 + 9 + 9 + 6 = 32 件）と、v5 の凍結テスト（11 件）が runtime の 2 slot だけを解決する。registry は `judgment-runtime-assets.json`。

`judgment-runtime-assets.json` は 169 件の identity を固定する。v2–v5 manifest の SHA-256 登録の union（155 件）、manifest 自身、`tests/` 以下で `github_retrieval_parity_v2`–`v5` を名前に含む tracked ファイル（テスト、fixture、証跡、v5 routing）、`github_retrieval_parity_v5_observation.py` である。registry 自身の SHA-256 は bridge に固定する。

検証の規則は README と同じにする。group の slot だけを原バイトで照合し、それ以外の登録資産はすべて current bytes で照合する。他の group の slot は解決しない（runtime group で README は current として照合する）。不明な slot、group、mapping、registry の commit、原バイトの欠落・改竄、registry の欠落・書換え、登録資産の欠落・改竄、path escape、symlink は失敗とする。slot の live ファイルが無い場合も失敗とする。

## 実行経路

`tests/test_000_historical_readme_routing.py` は `ROUTED_MODULE_COUNTS`（README 21 module と runtime 4 module）を bridge へ渡す。親 loader で原テスト数を確かめ、隔離した temporary root を作り、group の slot だけを原バイトに置き換えて子プロセスで実行する。子でも原テスト数、実行件数、成否を確かめる。対象外の module は従来の loader へ渡す。

v5 の凍結テストは既存の v5 routing（`tests/test_000_github_retrieval_parity_v5_routing.py`）がそのまま 11 件を 1 件の wrapper にまとめる。変えたのは wrapper の実行場所だけで、wrapper は bridge の `run_module("test_github_retrieval_parity_v5")` を呼ぶ。子は原テスト数 11 を確かめ、隔離 root で `verify_observation()` と `run_frozen_v5_tests()` を実行する。凍結テストの出力が `Ran 11 tests` と `OK` を含むときだけ成功とする。v5 module は loader router の対象には入れず、二重 routing を作らない。

対象 module を直接実行するときは `python tools/historical_readme_bridge.py --module <bare module name>` を使う。bridge を通さない直接実行は、現在の runtime と登録バイトの不一致を引き続き検出する。

## 検証境界

登録済みの実評価、holdout、real query、model inference は再実行しない。保存済み観測の照合と synthetic テストだけを実行する。`tests/test_historical_runtime_bridge.py` は、原バイトが登録 commit の `git show` と一致すること、registry が v2–v5 manifest の union と関連 identity を覆うこと、各種の改竄・欠落・不明 mapping・他 group の slot・path escape、routing の閉包と原テスト数、追加テスト、v5 wrapper の件数不足と失敗、子プロセスの失敗を検出することを確かめる。

`docs/requirements.md` は CE 系の凍結 corpus と README registry に登録されているため、この変更では書き換えない。仕様は [Canonical SQLite judgment graph](canonical-sqlite-judgment-graph.md) と [Decision Wiki 差分取り込み](specifications/decision-wiki-incremental-import.md) に書く。
