# Windows インストーラーの配布契約

## 目的

Windows x64 の利用者が checkout と開発用 Python を用意せず、共有 MCP と通知領域の操作を使えるようにする。通常 wheel の境界は [runtime-wheel.md](runtime-wheel.md)、共有 MCP の意味は [shared-local-mcp.md](shared-local-mcp.md) に従う。この配布は検索と feedback の既定動作を変更しない。

## 成果物とバージョン

- `NGR-<version>-windows-x64-cpu-setup.exe` と `NGR-<version>-windows-x64-cuda-setup.exe`。`version` は `packaging/windows/release-version.txt` から読み取り、隔離した配布用 source の package metadata に反映する。ハッシュ登録済みの原本 `pyproject.toml` は変更しない。各 EXE に `*.exe.sha256` を添える。ハッシュ行は SHA-256 とファイル名を記す。
- CI artifact の `package-manifest.json` は `schema: ngr.windows-package/v1`, `version`, `flavor`, `setup_file`, `sha256`, `size`（bytes）、`bundle_size_bytes`（onedir の総 bytes）を持つ。インストール先の `NGR.exe` と同じディレクトリには `schema`, `version`, `flavor` を持つ manifest を置く。セットアップ自身の digest をセットアップ内 manifest に含める循環はしない。
- `NGR.exe --version` は実行形式に含めた配布 metadata の version を表示する。公開 GitHub Release への asset 添付と Latest の変更はこの workflow の範囲外。

## ビルドと実行境界

GitHub Actions Windows runner が PyInstaller onedir をビルドし、Inno Setup が各 onedir を一つの per-user セットアップ EXE にまとめる。Inno Setup の `PrivilegesRequired=lowest` と `{localappdata}\Programs\Neuron Graph RAG` により昇格を要求しない。セットアップの LicenseFile は NGR と Microsoft runtime の両条件を一つの画面で提示し、通常の画面操作では同意を要求する。無人セットアップは `/ACCEPTVCRUNTIME=yes` を明示しなければ開始しない。Service 登録、ログイン時自動起動、更新時のデータ削除は行わない。初回 MCP stdio 接続が共有 HTTP 本体と通知領域のコントローラーを起動する。frozen 版の子プロセスは同じ `NGR.exe` に `--http` / `--tray-controller` を渡す。

CPU 版は `mcp`, `httpx2`, `uvicorn`, `starlette`, `numpy`, `onnxruntime`, `tokenizers` を含み、PyTorch と CUDA 実行依存を含めない。CUDA 版はこれに CUDA 対応 PyTorch、`transformers`, `safetensors` を追加する。固定 revision の E5 ONNX と v2-m3 model snapshot はサイズとライセンス確認のため両版とも含めず、利用者が明示的に別途配置する。CUDA 版でもモデル path の指定がない通常検索は CPU の既定経路に従う。

CUDA 版では Transformers の遅延読み込みがモデル種別の一覧からモジュールを動的に import するため、PyInstaller に `transformers.models` の全サブモジュールを収集させる。インストール済み EXE の archive に、ビルド環境の固定 Transformers 版にあるモデルモジュールがすべて含まれることを package smoke で照合する。これは GPU のない CI で確認できる配布内容の検査であり、実際の v2-m3 推論は GPU 実機で別途確認する。

## ライセンス表示

インストール先の `licenses/` に NGR の `LICENSE` と `NOTICE`、ビルドに使った CPython 配布物の `LICENSE.txt`、Inno Setup のライセンス、PyInstaller が実際に収録した第三者 Python 配布物のライセンス・NOTICE 本文を置く。スタートメニューの「Licenses and notices」から一覧を開ける。Inno Setup のセットアップ画面にも NGR ライセンスを表示する。`licenses/manifest.json` は各文書の SHA-256 と、PyInstaller の Analysis TOC とインストール済み wheel の RECORD から特定した配布物名・version を記録する。TOC に現れた site-packages のファイルを配布物へ帰属できない場合、または帰属した配布物のライセンス本文を wheel または固定 tag の upstream 原文から得られない場合はビルドを失敗させる。tokenizers 0.22.1 の Windows wheel はライセンス本文を含まないため固定 tag の `LICENSE` を補い、同じ tag の Cargo.lock にある Windows 向け通常依存をたどって crates.io のチェックサム付き archive から集めた Rust ライセンス・NOTICE も加える。これは保守的な source 依存範囲で、実際にリンクされた crate の厳密な証明ではない。ONNX Runtime と PyTorch の upstream NOTICE も固定 tag の原文を wheel の文書に追加する。

PyInstaller の DLL 探索 PATH は選択した Python と Windows の標準ディレクトリに限定する。Windows 配布専用の NumPy は公式 `1.26.4` cp311-win_amd64 wheel に固定し、通常の `pyproject.toml` は変更しない。PyInstaller が自動収集した `msvcp140.dll` / `msvcp140_1.dll` / `vcruntime140.dll` / `vcruntime140_1.dll` の出自は採用しない。ライセンス文書と対になる正式版 Visual Studio 2022 の `VC\Redist\MSVC\<version>\x64\Microsoft.VC143.CRT` にある 4 ファイルで上書きする。`preview` と `debug_nonredist` は除外し、4 ファイルがそろわない、x64 PE でない、コピー後の SHA-256 が違う場合はビルドを失敗させる。`licenses/manifest.json` の `vc_runtime` に VS 年版、REDIST list、元のディレクトリ、各 DLL の元ファイル path・file version・SHA-256 を記録する。最終 onedir の DLL を再計算し、この 4 ファイル以外の Visual C++ runtime DLL があれば失敗させる。セットアップは System32 の VC++ ランタイムを要求しない。それ以外の TOC ファイルが NGR source、CPython 配布物、wheel RECORD のどれにも帰属しない場合もビルドを失敗させる。

[Visual Studio 2022 REDIST list](https://learn.microsoft.com/en-us/visualstudio/releases/2022/redistribution) は、対象ファイルを未改変でアプリと配布できる範囲を示す。`Setup-LICENSE.txt` は NGR と [Microsoft の runtime terms](https://visualstudio.microsoft.com/license-terms/vs2022-cruntime/) をセットアップ内で提示する。Visual Studio 2022 の Distributable Code 条項は、外部利用者との契約が Microsoft を保護する条件も課すため、提示した文面がその水準を満たすかという契約判断は公開前に別途必要である。NGR の機能が主であること、配布者の Visual Studio ライセンス、外部利用者への条項の十分性を確認する。VC DLL の更新は NGR の次のリリースで行い、manifest の出自と条件を再監査する。Visual Studio 2026 へ切り替える場合はその年版の条項と同意文面を先に更新する。

NumPy の `numpy.libs/libopenblas*.dll` はインストール済み wheel RECORD の SHA-256 と実 onedir の SHA-256 を照合し、NumPy wheel の `LICENSE.txt` が OpenBLAS、LAPACK、静的リンクされた GCC runtime（libgfortran）と libquadmath の各条項を含むことを確認する。監査結果は `licenses/manifest.json` に実ファイルの相対 path、digest、RECORD path と対応するライセンス path を記録する。これは wheel の記述と実バンドルの一致検査であり、将来 wheel の構成が変わった場合は再監査してから更新する。

CPython の Windows EXE / DLL / PYD にリンクされた Microsoft Distributable Code の条件は、その配布物の `LICENSE.txt` に従う。standalone runtime DLL の除外と文書の同梱は、配布物全体の包括的な法的確認に代わるものではない。モデル weight は両版に含めない。別途取得する [E5 の固定 revision](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3) の model card は MIT、[v2-m3 の固定 revision](https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e) は Apache-2.0 と表示する。これはモデルをインストーラーで再配布しているという意味ではない。

ビルド依存の version と取得元は [`packaging/windows/build.ps1`](../../packaging/windows/build.ps1) に固定する。Python package は PyPI、CUDA PyTorch は PyTorch の `cu128` wheel index、Inno Setup 6.7.3 は開発元の GitHub release から取得して SHA-256 を照合する。ユーザー配布の EXE にはコード署名を施していない。出力 digest は破損・取り違えの検査用で、発行元認証にはならない。

## 状態と登録

DB、token、cache、モデルはインストール先でなくユーザー領域または利用者指定 path に置く。更新とアンインストールはそれらを削除しない。設定用コマンド `NGR.exe --configure-clients` は利用者が Codex / Claude Code を個別に選択した場合だけ MCP 登録を行う。Claude Code は利用者がフォルダー選択画面で指定した既存プロジェクトのルートに限り、`claude mcp add --scope project` により `.mcp.json` を更新する。別プロジェクトには自動登録しない。同名 `ngr-shared` が選択先にあれば安全な要約と設定ファイル path を示し、利用者がエディターで内容を確認できるようにする。選択先の JSON が不正なら変更を止める。CLI の任意出力や設定本文はコンソールやログに転記せず、自動上書きしない。新規登録前には既存設定ファイルを timestamp 付きで退避し、他のサーバーを保持する。既存 user scope 登録は project 登録成功後も維持し、利用者が他プロジェクトへの影響を確認して選んだ場合だけホーム設定を退避して削除する。複数プロジェクトに個別登録でき、登録内容は `${NGR_MCP_EXE} --shared` を参照する。ユーザー環境変数 `NGR_MCP_EXE` はこのマシンに導入した EXE の絶対 path とし、インストール先を各プロジェクトの設定に複製せず、一つの共有本体に接続する。新規 token はユーザー環境変数 `NGR_MCP_HTTP_BEARER_TOKEN` に保存し、コマンド引数や MCP 設定には書かない。既存アプリは環境変数を読み直すため再起動する。

Claude Code の project 設定と、移行時に削除する旧 user scope 設定のバックアップは、ユーザー専用の `%LOCALAPPDATA%\Neuron Graph RAG\mcp-backups\` 以下へ保存する。プロジェクトの Git 管理対象やホーム直下に設定本文の退避コピーを作らない。アンインストール後も `NGR_MCP_HTTP_BEARER_TOKEN` と `NGR_MCP_EXE` のユーザー環境変数、MCP 登録、退避ファイルは残る。

## CI 受け入れ

PR と main の両方で CPU / CUDA を別 job でビルドし、インストールした `NGR.exe` に Python のない PATH を渡して version、初回 stdio MCP 接続、共有 identity、tool 一覧、トレイ状態、停止・再開、アンインストール後の DB 残存を検査する。CUDA 版では固定 Transformers 版のモデルモジュール一覧と EXE の収録内容も照合する。GPU を持たない runner では CUDA wheel の同梱と起動入口・モジュール収録まで確認し、実機 NVIDIA GPU の推論成功は検証範囲に含まれない。

同じ smoke test は、インストール先の全ライセンス文書の存在・ハッシュ・非空本文、NumPy OpenBLAS DLL と wheel RECORD 由来の digest、VC\Redist 由来の 4 DLL の実 digest、主要 runtime 配布物の掲載、CPU/CUDA の区別、モデル weight の非同梱を検査する。CPU shortlist と CUDA shortlist の既存 unit test も NumPy 1.26.4 を導入した環境で実行する。
