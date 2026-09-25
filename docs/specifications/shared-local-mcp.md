# 共有ローカル MCP サービス

## 接続時起動（Issue #259）

`neuron-graph-rag-mcp --shared` は stdio MCP の入口であり、接続開始時に認証済みのローカル HTTP サービスを確認する。サービスが停止中なら一つのクライアントが起動し、同時接続の後続クライアントは同じプロセスに接続する。プロセス間の起動ロックは port 単位で、lock file に token を保存しない。認証済み identity は NGR のサービス種別、絶対 DB path、CUDA 設定と PID を返す。不一致、認証失敗、他サービスによる port 占有は中継を開始せず診断可能なエラーにする。

stdio 入口は HTTP に tool 一覧と tool 呼び出しを中継し、既存 tool の公開内容と戻り値を変えない。接続ごとに stdio プロキシと HTTP MCP session は作るが、HTTP サービスの `FeedbackMCPAdapter`、SQLite connection と opt-in CUDA retriever は一つのまま。stdio プロキシの終了は共有サービスを停止しない。共有サービスの寿命は明示的な `--stop` または OS ログアウト／終了までである。`--stop` は token と DB path の一致を確認した後にサービスへ認証済み停止要求を送り、プロセス終了まで待つ。停止後、次の `--shared` 接続は再起動する。

token 未設定・不正値ならサービス起動前に拒否する。token は環境変数 `NGR_MCP_HTTP_BEARER_TOKEN` のみから渡し、設定本文、起動引数、ログには書かない。手動 `--http` と従来の token 不要 stdio は継続する。`--shared` の CUDA path 一式は opt-in で、稼働中のサービスと不一致なら接続を拒否する。

## 手動 HTTP（Issue #257）

Issue #257 の実装契約。`neuron-graph-rag-mcp --http` は、利用者端末で一つの NGR プロセスを起動し、固定ループバック URL `http://127.0.0.1:8765/mcp/` で Streamable HTTP を公開する。ポートは明示変更できるが bind 先は `127.0.0.1` に固定する。MCP SDK の Host / Origin 検査はそのポートの `127.0.0.1` だけを許可する。

HTTP 起動には専用環境変数 `NGR_MCP_HTTP_BEARER_TOKEN` に 32 文字以上の非公開 base64url token を要求し、未設定・不正な値なら DB を開く前に起動を拒否する。すべての `/mcp/` HTTP 要求は MCP session manager より前に `Authorization: Bearer <token>` の一致を検査し、欠落・不一致・重複した認証 header は `401` を返す。token は URL、ログ、起動メッセージに出さない。認証済み要求にも Host / Origin 制限を適用する。従来の直接 stdio 入口に token 要件は追加しない。

同一 OS ユーザーのクライアントが秘密を共有する運用を想定するが、HTTP は OS アカウント自体を照合しない。端末内の別ユーザーやプロセスでも token を入手すれば書き込み tool を呼べるため、token の配布・保管・更新は利用者が管理する。ループバック bind だけを同一ユーザーの認証とみなさない。

HTTP 接続ごとに MCP session を作るが、server、`FeedbackMCPAdapter`、NGR engine、SQLite connection はプロセス内で一つを共有する。同期的な DB と CUDA 操作は一つのイベントループ上で逐次実行する。プロセスを止めると session manager を終了し、CUDA retriever と DB を閉じる。複数 worker・別スレッドで同一 connection を共有する運用はサポートしない。

既存 `neuron-graph-rag-mcp` の stdio 起動と九つの既存 tool、通常の `search` と `engine.search()` の意味は維持する。HTTP ではこの九つに `update_cuda_shortlist_cache` と `search_cuda_shortlist` を追加する。後者は明示的な CUDA shortlist 順位を返す読み取り操作であり、feedback trace を発行せずグラフを強化しない。返す hit は node ID、本文、metadata、score、stage 1 順位、選択 chunk、diagnostics を含む。feedback を必要とする利用者は既存 `search` を使う。

CUDA の三つの path は cache、固定 E5 snapshot、固定 v2-m3 snapshot を一緒に指定する。モデルの自動取得と CPU fallback はしない。CUDA が未設定なら GPU tool は `cuda_unavailable` を返し、利用不能・モデル欠落・実行時失敗も明示的に失敗する。E5 cache は corpus に対して明示的に更新し、古い cache での検索は失敗する。一つの retriever がモデルを保持し、連続する tool 呼び出しで再利用する。

通常 wheel の必須依存は増やさない。HTTP は既存の `mcp` extra に含まれる SDK、Starlette、Uvicorn を使う。ONNX E5、transformers、CUDA 対応 PyTorch、モデル weight は別途利用者が用意する。外部ネットワーク公開、認証付き remote 配置、GUI、トレイ、インストーラー、複数 worker は対象外。
