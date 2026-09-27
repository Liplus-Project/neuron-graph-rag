# Windows 配布版の検証付き更新

## 配布契約

インストール先の `NGR.exe` と同じディレクトリに `package-manifest.json` を置く。インストール内の値は `schema: "ngr.windows-package/v1"`、`version`（`MAJOR.MINOR.PATCH`）、`flavor`（`cpu` または `cuda`）。ビルド artifact 側の manifest はさらに `setup_file`、`sha256`、`size`、`bundle_size_bytes`（bytes）を持つ。manifest が欠落したソース版では現在版を表示できるが、CPU/CUDA 配布種別を推定せず更新候補を出さない。

候補は `Liplus-Project/neuron-graph-rag` の公開済み Latest GitHub Release 一件から得る。draft、prerelease、公開日時がない release は対象外。現在版より新しい安定した `MAJOR.MINOR.PATCH` と同じ flavor の `NGR-<version>-windows-x64-<flavor>-setup.exe` を要求する。別 flavor や PR artifact は候補にしない。release の公式ページ URL、asset の公式 URL、正の size（最大 16 GiB）、GitHub Releases API の `sha256:` digest がそろわなければ候補を拒否する。

## 操作と安全境界

トレイは現在版・flavor・候補を表示し、24 時間以上の間隔で控えめに確認する。自動確認はユーザー領域の `~/.ngrdb/update-settings.json` で無効にでき、手動確認は残す。確認と取得はトレイとは別のスレッドで行い、ネットワーク障害やオフライン状態は MCP 本体の接続・検索・停止に影響しない。

利用者が `Download verified installer` を選び、確認ダイアログで承諾した時だけ取得する。`~/.ngrdb/updates/` の一時ファイルへ逐次書き込み、宣言 size と SHA-256 が完全一致した場合だけ正式ファイル名に移す。キャンセル、通信失敗、size・hash 不一致は一時ファイルを削除し、現行版を維持する。既存ファイルは上書きしない。検証後に release ページと保存先を表示する。

現行の配布にはコード署名の信頼連鎖と署名検証手順がない。digest は取得物の整合性を示すが、発行者署名の代わりにはならない。そのため NGR はインストーラーを起動せず、サービス停止・再起動・設定移行も自動では行わない。適用する場合は利用者が release 内容を確認し、トレイの `Exit` で本体と旧トレイを終了してから手動でセットアップを実行する。再接続と DB・MCP 設定の確認も利用者の操作とする。自動適用と署名検証は別 issue で扱う。

更新経路は DB、token、モデル、cache、MCP クライアント設定を読み書きしない。インストーラー側もユーザーデータを削除しない契約である。複数クライアントが同時に更新適用を競合する経路は提供しない。release 作成や Latest 変更も行わない。
