# Windows 版 NGR の更新を確認する

Windows 配布版で共有 MCP に接続すると、通知領域の NGR アイコンに現在版と更新状態が表示される。`Check for updates` で今すぐ確認できる。通常は 24 時間以上の間隔で自動確認する。オフラインでも MCP はそのまま使える。`Disable automatic update checks` を選ぶと自動確認を止められ、再度選ぶと再開する。

新しい同じ CPU/CUDA 種別の正式版が見つかった場合、`Download verified installer` を選べる。確認後に公開 release のインストーラーを取得し、size と SHA-256 を照合して `~/.ngrdb/updates/` に保存する。取得中は `Cancel download` を選べる。失敗や検証不一致なら保存しない。`Open release page` から公開内容を確認できる。

**インストーラーは未署名です。** NGR は取得・検証まで行い、インストーラーを自動実行しない。適用する場合は release ページを確認し、トレイの `Exit` で共有 MCP と旧トレイを終了する。続いて保存済みセットアップを手動実行し、MCP の再接続で起動を確認する。停止前から接続していた AI クライアントは接続し直す。DB、token、モデル、cache、MCP 設定は更新確認・取得で変更しない。インストール後に同じ DB と MCP 設定を確認する。

セットアップの署名確認、正常停止と再起動を含む自動適用は未提供。詳細な安全境界は[検証付き更新の仕様](../specifications/windows-verified-updates.md)を参照。
