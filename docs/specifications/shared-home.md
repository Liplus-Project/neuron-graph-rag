# NGR 共通ホーム仕様

NGR のユーザー設定の正本は `~/.ngr/config.json`。新しい既定 DB は
`~/.ngr/db/knowledge.db`、共有本体・トレイの lock、paused、marker、ログ、
更新設定・取得物、登録前の退避は `~/.ngr` に置く。
旧版 v0.2.1 の `.ngrdb` と凍結済み実験仕様は旧版の記録として維持する。

CLI > 環境変数 > 中央 JSON > 既定値の順で DB、port、CUDA を解決する。
JSON の相対 path は `.ngr` 基準。CUDA は三つの path が揃ったときだけ有効。
不正 JSON、不明な項目、型・範囲の誤りは起動を拒否する。
token はユーザー環境変数のみから読む。JSON、起動引数、ログへ保存しない。

旧 `.ngrdb/knowledge.db` が存在する未移行環境では既定起動を拒否する。
`--migrate-home --confirm-stopped` は利用者が全クライアントと旧トレイを終了した後に
実行する。実行中の NGR 本体・トレイ・プロキシおよび SQLite 書込みロックを検出した場合は拒否する。
SQLite backup API で整合性確認済みの退避と新 DB を非上書きで作成し、
再開用記録を保存してから設定を切り替える。旧 DB は復旧用に保持する。
成功記録により旧 DB と運用 DB を区別し、旧 DB は以後の運用に使わない。
設定保存失敗は旧 DB、退避、新 DB と再開記録を保持し、同じコマンドで再試行する。
未記録の移行先競合やコピー後の DB 変更は拒否する。明示 DB は自動移行しない。

Windows 設定コマンドは Codex 共通、Claude Code user scope、Claude Desktop、
従来の Claude Code project scope を個別に選べる。
各 native 登録には実行ファイルと `--shared` の入口だけを置く。
他サーバーを保持し、同名登録の置換は安全な要約と明示選択、`.ngr/backups` の退避を伴う。
CLI の出力と登録済み引数・秘密値を表示しない。
Claude Desktop は公式の `%APPDATA%/Claude` と実在する Store パッケージの
`LocalCache/Roaming/Claude` を調べ、複数候補・候補なしの場合は利用者に選択を求める。

公開リリース・版変更・実データ切替はこの実装の検証後に親側で判断する。
