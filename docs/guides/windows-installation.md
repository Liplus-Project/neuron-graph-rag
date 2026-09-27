# Windows 版の導入・更新・削除

## 選ぶ版

通常は `NGR-<version>-windows-x64-cpu-setup.exe` を選ぶ。NVIDIA GPU と対応 driver があり、固定モデルを自分で配置して CUDA 検索を使う場合は `...-cuda-setup.exe` を選ぶ。CUDA 版は PyTorch 等を含むため容量が大きく、GPU とモデルがない環境では CUDA 検索が使えない。通常検索と共有 MCP は CPU 版で動く。

ダウンロードした EXE と同名の `.sha256` を同じ場所に置き、PowerShell で `Get-FileHash .\NGR-<version>-windows-x64-cpu-setup.exe -Algorithm SHA256` の値を比較する。これらは未署名の EXE であり、SHA-256 は配布元の身元を証明しない。公開 release asset は公開前に別途確認される。インストールに管理者権限、Python、checkout は不要。

## 導入と MCP 登録

セットアップを起動すると、既定では現在ユーザーの `%LOCALAPPDATA%\Programs\Neuron Graph RAG` に onedir が入る。初回接続時に共有 MCP とトレイが起動する。Windows PowerShell 5.1 と WMI を子プロセス起動に使うため、Windows の標準機能として有効にしておく。Service とログイン時自動起動は追加されない。

セットアップ完了画面、またはスタートメニューの **Configure MCP clients** から `NGR.exe --configure-clients` を実行する。Codex と Claude Code をそれぞれ選び、表示された `ngr-shared` のコマンドと設定先を確認して登録を承認する。既存の同名登録があれば安全な要約と設定ファイル path を示し、利用者がエディターで内容を開いて確認できる。CLI の任意の設定出力は転記せず、上書きせずに止まる。設定ファイルの退避を選べる。新規登録時は設定ファイルを自動退避する。既存名を変える場合は利用者自身が内容と退避を確認してからクライアントの管理機能で削除し、登録コマンドを再実行する。

初回登録時、秘密 token がなければ現在ユーザーの環境変数に生成する。token はコマンド引数、MCP 設定、診断ログに書かない。既に Codex / Claude Code が起動していれば再起動して環境変数を読み直す。MCP を使うときは `NGR.exe --shared` が stdio proxy となり、`127.0.0.1:8765` の一つの本体と DB を共有する。トレイの **Stop and release GPU** で停止、**Resume** で再開、**Exit** で終了して次回接続時の自動起動に戻す。

CUDA 検索は [CUDA ガイド](cuda-shortlist-retrieval.md)に従い、固定 revision の E5 ONNX と v2-m3 snapshot を利用者管理の model directory に別途配置する。クライアント登録に `--cuda-cache`、`--cuda-e5-snapshot`、`--cuda-v2-m3-snapshot` の同じ絶対 path を設定したい場合、登録後にクライアントの MCP 設定で `NGR.exe --shared` に追加する。model はインストーラーと更新対象に含まれない。

## 更新

トレイから公開済みの互換版を確認し、size と SHA-256 を検証して取得できる。操作は[検証付き更新のガイド](windows-verified-updates.md)を参照する。手動取得した場合も配布 EXE と `.sha256` を検証する。共有本体とトレイを **Exit** で終了してから新しいセットアップを同じユーザーで実行する。インストール先のプログラムを入れ替える。DB、token、cache、別置きモデルは更新対象に含まれない。`NGR.exe --version` または同じディレクトリの `package-manifest.json` で導入版を確認できる。MCP 登録 path を変えた場合は **Configure MCP clients** で現状を確認して自分で更新する。署名検証付きの自動適用は別 issue の範囲。

## アンインストール

トレイの **Exit** を選び、Windows 設定の「インストールされているアプリ」から **Neuron Graph RAG** を削除する。インストーラーが配置したプログラムが削除される。ユーザー DB (`%USERPROFILE%\.ngrdb` または指定先)、token 用ユーザー環境変数、cache、別置きモデル、Codex / Claude Code の MCP 登録は維持する。これらを消す場合はバックアップを確認した上で利用者が個別に削除する。MCP 登録を残すとクライアントが存在しない `NGR.exe` を呼ぶため、不要なら各クライアントの MCP 管理機能で `ngr-shared` を削除する。
