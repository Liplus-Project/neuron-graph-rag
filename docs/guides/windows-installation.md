# Windows 版の導入・更新・削除

## 選ぶ版と導入

v0.3.0 は共通ホーム対応の配布候補。通常は `NGR-<version>-windows-x64-cpu-setup.exe` を選ぶ。
NVIDIA GPU、対応 driver、固定モデルを配置して CUDA 検索を使う場合は CUDA 版を選ぶ。
通常検索と共有 MCP は CPU 版で動く。モデル weight はどちらにも含まれない。

EXE と同名の `.sha256` を取得し、PowerShell の `Get-FileHash <EXE> -Algorithm SHA256` で比較する。
EXE は未署名。SHA-256 は配布元の身元を証明しない。インストールに管理者権限、Python、checkout、
Visual C++ Redistributable の別途導入は不要。同梱の x64 VC runtime 4 DLL を使う。

セットアップでは NGR と Microsoft Visual C++ runtime の条項を確認して進める。
無人導入は `/ACCEPTVCRUNTIME=yes` で同意を明示する。既定の導入先は
`%LOCALAPPDATA%\Programs\Neuron Graph RAG`。ライセンスは **Licenses and notices** または
`licenses/README.txt` にあり、`licenses/manifest.json` に文書・DLL の SHA-256 と出自を記録する。
Windows PowerShell 5.1 と WMI を初回共有本体・トレイ起動に使う。
ログイン時自動起動や Windows Service 登録は追加しない。

## 共通 MCP 登録

**Configure MCP clients** (`NGR.exe --configure-clients`) で次を個別に選ぶ。

- `codex`: ユーザーの Codex `config.toml`。同じローカル Codex ホストの CLI と Codex / ChatGPT Desktop が共有する。
- `claude-user`: ホームの `.claude.json` の user scope。各プロジェクトで同じ入口を利用する。
- `claude-desktop`: 通常版または Store 版の実在設定。複数候補なら選ぶ。
- `claude-project`: 従来のフォルダー選択と `.mcp.json` 登録。既存 project scope の互換入口。

新しい共通登録は導入 EXE と `--shared` のみを設定する。Codex は token 等を
`env_vars` の変数名で透過指定し、値を TOML へ保存しない。他サーバーを保持し、同名登録は
安全な要約を確認して置換するか選ぶ。変更前の native ファイルは
`%USERPROFILE%\.ngr\backups\clients` に退避する。CLI の出力や秘密値は画面に転記しない。
project 互換登録は `${NGR_MCP_EXE} --shared` を使い、旧 user scope 削除も個別選択する。
共有する `.mcp.json` を使うマシンには NGR とそのマシンの `NGR_MCP_EXE` が必要。

Claude Desktop 通常版は公式の `%APPDATA%\Claude\claude_desktop_config.json`。
Store 版は `%LOCALAPPDATA%\Packages\Claude_*\LocalCache\Roaming\Claude` の実在ファイルを探す。
候補がない・複数ある場合は **Settings > Developer > Edit Config** で開くファイルを選ぶ。
パッケージ ID は固定しない。設定と再起動条件は [MCP 公式ガイド](https://modelcontextprotocol.io/docs/develop/connect-local-servers)、
user scope は [Claude Code 公式仕様](https://code.claude.com/docs/en/mcp#user-scope)、
同じ Codex ホストの共有は [OpenAI 公式仕様](https://learn.chatgpt.com/docs/extend/mcp) に基づく。
クラウドの ChatGPT チャットは今回のローカル登録対象に含まれない。

初回登録は `~/.ngr/config.json` を `{}` で非上書き作成する。
未設定なら DB は `~/.ngr/db/knowledge.db`、port は 8765、CUDA は無効。
CUDA を有効にする例は次のとおり。CUDA を使わない場合は CUDA 項目を省く。

```json
{
  "port": 8765,
  "cuda_cache": "cache/shortlist.db",
  "cuda_e5_snapshot": "D:/models/e5",
  "cuda_v2_m3_snapshot": "D:/models/v2-m3",
  "cuda_device": 0
}
```

JSON の相対 path は `.ngr` 基準。DB、port、CUDA は CLI > `NGR_DATABASE` / `NGR_PORT` /
`NGR_CUDA_*` > JSON > 既定値。起動・停止・トレイ再開が同じ resolver を使う。
中央設定を変更する前に全クライアントとトレイを終了する。
固定モデルの配置は [CUDA ガイド](cuda-shortlist-retrieval.md) に従う。モデルはインストーラー更新対象に含まれない。

秘密 token はユーザー環境変数 `NGR_MCP_HTTP_BEARER_TOKEN` にだけ生成・保持する。
設定時に Explorer へ環境変更を通知する。既に起動している全クライアントと起動元の端末を終了し、
スタートメニューから開き直す。環境通知が失敗した場合はサインアウト後に開き直す。
token は JSON、起動引数、診断ログに保存しない。

## v0.2.1 以前からの移行

旧 DB は `~/.ngrdb/knowledge.db`。未移行の旧 DB がある状態では、新しい空 DB を作る既定起動を拒否する。
運用先は移行後の `.ngr/db`。残る `.ngrdb` は復旧用旧データであり、新版はそこへ接続しない。
旧版の EXE を再起動しない。

1. 全 MCP クライアントを終了し、旧トレイの **Exit** で本体とトレイを終了する。
2. v0.3.0 のセットアップでプログラムを更新する。DB をインストーラーで移動しない。
3. 新 EXE を `NGR.exe --migrate-home --confirm-stopped` で実行する。
4. 整合性確認済み `.ngr/db/knowledge.db` と `.ngr/backups/legacy-knowledge.db`、完了記録を確認する。
5. **Configure MCP clients** で既存入口を退避して共通登録へ切り替え、クライアントを開き直す。

稼働中 NGR、SQLite writer、既存移行先、破損、移行中の再起動は拒否する。
コピーや設定保存が失敗した場合は旧 DB とコピーを保持し、`home-migration.pending` で新規起動を止める。
原因を解消して同じコマンドを再実行する。`home-migration.json` が再開位置を持つ。
移行先や退避を手で編集した場合は上書きせず停止するため、全ファイルを保持して内容を確認する。
移行成功後の DB 編集は通常利用として許可する。
`--database`、`NGR_DATABASE`、JSON `database` の利用者管理 DB は自動移行しない。
その DB は `python tools/migrate_database.py --source <旧DB> --destination <新DB> --backup <退避>`
で別途退避・確認し、中央 JSON の `database` に選択した絶対 path を設定する。

## 運用・更新・削除

各 stdio proxy は `127.0.0.1` の一つの本体と DB を使う。
**Stop and release GPU** で停止、**Resume** で同じ設定を再開、**Exit** で終了して次回接続の自動起動へ戻す。
管理ログ、paused、lock、トレイ marker、更新設定・取得物は `.ngr` 内。
中央ファイルなしの旧明示 DB 登録では、旧監視入口に対応する一時的な `.ngrdb` marker hard-link を作り、トレイ終了時に消す。

更新の size / SHA-256 検証付き取得は [更新ガイド](windows-verified-updates.md) を参照する。
全クライアントとトレイを終了して同じ導入先へ更新する。DB、token、cache、モデルを削除しない。
`NGR.exe --version` または `package-manifest.json` で版を確認する。取得物を自動実行しない。

削除は全クライアントとトレイ終了後、Windows の **インストールされているアプリ** から実行する。
`.ngr`、復旧用 `.ngrdb`、明示 DB、モデル、token / `NGR_MCP_EXE` 環境変数、native MCP 登録は残る。
不要な登録とデータはバックアップを確認して利用者が個別に削除する。
