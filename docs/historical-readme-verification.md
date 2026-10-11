# 過去評価の README バイト分離

## 要件

現在の README の画像・下線は表示用文書として維持する。21 個の過去評価テスト module（原テスト合計 276 件）は、登録済み evaluator、tests、manifest、証跡を変更せず、隔離 temporary root の current bytes を検証する。README.md の登録 SHA-256 `5a1438ea5c8eecbed015661ad1decd7451e0c07aa848efc110557334f7de3638` だけを、commit `b3cc03a15b81f0e395ae564387a46fe57d320f31` の保存済み 31,856 byte LF ファイルへ解決する。

## 開発検証 bridge

`tools/historical_readme_bridge.py` は wheel に含めない。bridge registry は CE v1–v16 の関連 manifest の登録資産の union と、manifest 自身・対象 tests・保存済み関連証跡・materialized corpus 文書の identity を固定する。registry 自身の SHA-256 を bridge に固定し、自己整合する manifest/hash 書換えでも通過させない。既知 README の exact path/hash 以外は current root で照合する。missing、改竄、不明 mapping、path escape、symlink は失敗とする。

実行時は全 current tracked ファイルを temporary root へコピーし、README slot のみ保存済み原バイトへ置換する。コピー後も同じ preflight を行う。独立した temporary Git metadata と refs を作り、object alternates に current repository の immutable objects の読取先を設定し、既存 evaluator の historical `git show` が使えるようにする。`GIT_DIR` / `GIT_WORK_TREE` を子プロセスへ設定しない。source repository の index / refs は共有しない。合成テストが作る別 repository の git identity を奪わず、git-free verifier と root alias 契約は変更しない。corpus 文書は current bytes からコピーし、欠落・不一致時に archive fallback を行わない。

`tests/test_000_historical_readme_routing.py` は明示した 21 module の全テストを bridge へ委譲する。各 module の original count を親 loader と隔離子 loader の両方で確認し、子の実行件数・失敗を確認する。対象外 module は従来 loader へ渡す。子は明示 module のみを読み、router を再導入しない。既存 v5 routing は変更しない。

通常の full discovery と `tools/run_tests.py` の suite に router を含める。対象 module の直接実行で bridge を使う場合は `python tools/historical_readme_bridge.py --module <bare module name>` を使う。bridge を通さない凍結テストの直接実行は、現在 README と過去登録バイトの不一致を引き続き検出する。

## 検証境界

登録実評価、holdout、real query、model inference を再実行しない。保存済み観測の照合と synthetic テストだけを実行する。bridge の負例は temporary root で archive/登録資産/manifest/identity の改竄と欠落、routing closure、追加原テスト、対象外委譲、子プロセス失敗を検出する。全 core CI と runtime wheel の既存 allowlist を維持する。
