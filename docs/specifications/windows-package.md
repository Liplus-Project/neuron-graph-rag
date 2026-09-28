# Windows インストーラーの配布契約

## 目的

Windows x64 の利用者が checkout と開発用 Python を用意せず、共有 MCP と通知領域の操作を使えるようにする。通常 wheel の境界は [runtime-wheel.md](runtime-wheel.md)、共有 MCP の意味は [shared-local-mcp.md](shared-local-mcp.md) に従う。この配布は検索と feedback の既定動作を変更しない。

## 成果物とバージョン

- `NGR-<version>-windows-x64-cpu-setup.exe` と `NGR-<version>-windows-x64-cuda-setup.exe`。`version` は `pyproject.toml` の `[project].version` をビルド時に読み取る。各 EXE に `*.exe.sha256` を添える。ハッシュ行は SHA-256 とファイル名を記す。
- CI artifact の `package-manifest.json` は `schema: ngr.windows-package/v1`, `version`, `flavor`, `setup_file`, `sha256`, `size`（bytes）、`bundle_size_bytes`（onedir の総 bytes）を持つ。インストール先の `NGR.exe` と同じディレクトリには `schema`, `version`, `flavor` を持つ manifest を置く。セットアップ自身の digest をセットアップ内 manifest に含める循環はしない。
- `NGR.exe --version` は実行形式に含めた配布 metadata の version を表示する。公開 GitHub Release への asset 添付と Latest の変更はこの workflow の範囲外。

## ビルドと実行境界

GitHub Actions Windows runner が PyInstaller onedir をビルドし、Inno Setup が各 onedir を一つの per-user セットアップ EXE にまとめる。Inno Setup の `PrivilegesRequired=lowest` と `{localappdata}\Programs\Neuron Graph RAG` により昇格を要求しない。Service 登録、ログイン時自動起動、更新時のデータ削除は行わない。初回 MCP stdio 接続が共有 HTTP 本体と通知領域のコントローラーを起動する。frozen 版の子プロセスは同じ `NGR.exe` に `--http` / `--tray-controller` を渡す。

CPU 版は `mcp`, `httpx2`, `uvicorn`, `starlette`, `numpy`, `onnxruntime`, `tokenizers` を含み、PyTorch と CUDA 実行依存を含めない。CUDA 版はこれに CUDA 対応 PyTorch、`transformers`, `safetensors` を追加する。固定 revision の E5 ONNX と v2-m3 model snapshot はサイズとライセンス確認のため両版とも含めず、利用者が明示的に別途配置する。CUDA 版でもモデル path の指定がない通常検索は CPU の既定経路に従う。

CUDA 版では Transformers の遅延読み込みがモデル種別の一覧からモジュールを動的に import するため、PyInstaller に `transformers.models` の全サブモジュールを収集させる。インストール済み EXE の archive に、ビルド環境の固定 Transformers 版にあるモデルモジュールがすべて含まれることを package smoke で照合する。これは GPU のない CI で確認できる配布内容の検査であり、実際の v2-m3 推論は GPU 実機で別途確認する。

ビルド依存の version と取得元は [`packaging/windows/build.ps1`](../../packaging/windows/build.ps1) に固定する。Python package は PyPI、CUDA PyTorch は PyTorch の `cu128` wheel index、Inno Setup 6.7.3 は開発元の GitHub release から取得して SHA-256 を照合する。ユーザー配布の EXE にはコード署名を施していない。出力 digest は破損・取り違えの検査用で、発行元認証にはならない。

## 状態と登録

DB、token、cache、モデルはインストール先でなくユーザー領域または利用者指定 path に置く。更新とアンインストールはそれらを削除しない。設定用コマンド `NGR.exe --configure-clients` は利用者が Codex / Claude Code を個別に選択した場合だけ MCP 登録を行う。Claude Code は利用者がフォルダー選択画面で指定した既存プロジェクトのルートに限り、`claude mcp add --scope project` により `.mcp.json` を更新する。別プロジェクトには自動登録しない。同名 `ngr-shared` が選択先にあれば安全な要約と設定ファイル path を示し、利用者がエディターで内容を確認できるようにする。選択先の JSON が不正なら変更を止める。CLI の任意出力や設定本文はコンソールやログに転記せず、自動上書きしない。新規登録前には既存設定ファイルを timestamp 付きで退避し、他のサーバーを保持する。既存 user scope 登録は project 登録成功後も維持し、利用者が他プロジェクトへの影響を確認して選んだ場合だけホーム設定を退避して削除する。複数プロジェクトに個別登録でき、登録内容は `${NGR_MCP_EXE} --shared` を参照する。ユーザー環境変数 `NGR_MCP_EXE` はこのマシンに導入した EXE の絶対 path とし、インストール先を各プロジェクトの設定に複製せず、一つの共有本体に接続する。新規 token はユーザー環境変数 `NGR_MCP_HTTP_BEARER_TOKEN` に保存し、コマンド引数や MCP 設定には書かない。既存アプリは環境変数を読み直すため再起動する。

Claude Code の project 設定のバックアップは、選択したプロジェクトの外にあるユーザー専用の `%LOCALAPPDATA%\Neuron Graph RAG\mcp-backups\` 以下へ保存する。プロジェクトの Git 管理対象に設定本文の退避コピーを作らない。

## CI 受け入れ

PR と main の両方で CPU / CUDA を別 job でビルドし、インストールした `NGR.exe` に Python のない PATH を渡して version、初回 stdio MCP 接続、共有 identity、tool 一覧、トレイ状態、停止・再開、アンインストール後の DB 残存を検査する。CUDA 版では固定 Transformers 版のモデルモジュール一覧と EXE の収録内容も照合する。GPU を持たない runner では CUDA wheel の同梱と起動入口・モジュール収録まで確認し、実機 NVIDIA GPU の推論成功は検証範囲に含まれない。
