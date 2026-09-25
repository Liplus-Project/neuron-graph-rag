# 共有ローカル MCP サービス

Issue #257 の実装契約。`neuron-graph-rag-mcp-http` は、同一 OS ユーザーの端末で一つの NGR プロセスを起動し、固定ループバック URL `http://127.0.0.1:8765/mcp/` で Streamable HTTP を公開する。ポートは明示変更できるが bind 先は `127.0.0.1` に固定する。MCP SDK の Host / Origin 検査はそのポートの `127.0.0.1` だけを許可する。

HTTP 接続ごとに MCP session を作るが、server、`FeedbackMCPAdapter`、NGR engine、SQLite connection はプロセス内で一つを共有する。同期的な DB と CUDA 操作は一つのイベントループ上で逐次実行する。プロセスを止めると session manager を終了し、CUDA retriever と DB を閉じる。複数 worker・別スレッドで同一 connection を共有する運用はサポートしない。

既存 `neuron-graph-rag-mcp` の stdio 起動と九つの既存 tool、通常の `search` と `engine.search()` の意味は維持する。HTTP ではこの九つに `update_cuda_shortlist_cache` と `search_cuda_shortlist` を追加する。後者は明示的な CUDA shortlist 順位を返す読み取り操作であり、feedback trace を発行せずグラフを強化しない。返す hit は node ID、本文、metadata、score、stage 1 順位、選択 chunk、diagnostics を含む。feedback を必要とする利用者は既存 `search` を使う。

CUDA の三つの path は cache、固定 E5 snapshot、固定 v2-m3 snapshot を一緒に指定する。モデルの自動取得と CPU fallback はしない。CUDA が未設定なら GPU tool は `cuda_unavailable` を返し、利用不能・モデル欠落・実行時失敗も明示的に失敗する。E5 cache は corpus に対して明示的に更新し、古い cache での検索は失敗する。一つの retriever がモデルを保持し、連続する tool 呼び出しで再利用する。

通常 wheel の必須依存は増やさない。HTTP は既存の `mcp` extra に含まれる SDK、Starlette、Uvicorn を使う。ONNX E5、transformers、CUDA 対応 PyTorch、モデル weight は別途利用者が用意する。外部ネットワーク公開、認証付き remote 配置、GUI、トレイ、インストーラー、複数 worker は対象外。
