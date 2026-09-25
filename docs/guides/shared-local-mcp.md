# 共有ローカル MCP を起動する

## 準備と起動

Python 環境に `pip install '.[mcp]'` を入れる。HTTP 用の秘密 token を生成して同じ OS ユーザーの環境変数に保存し、サービスを起動する。以下は PowerShell の例で、token 値を画面に表示しない。

```powershell
$token = (& python -c 'import secrets; print(secrets.token_urlsafe(32))').Trim()
[Environment]::SetEnvironmentVariable('NGR_MCP_HTTP_BEARER_TOKEN', $token, 'User')
$env:NGR_MCP_HTTP_BEARER_TOKEN = $token
Remove-Variable token
neuron-graph-rag-mcp --http --database "$HOME/.ngrdb/knowledge.db"
```

既定の接続先は `http://127.0.0.1:8765/mcp/`。別ポートを使う場合は `--port 8766` を指定し、クライアントの URL も一致させる。端末で Ctrl+C を押して停止する。DB path を省略したときは `NGR_DATABASE`、次に `~/.ngrdb/knowledge.db` を使う。token がない場合は DB を開く前に起動を拒否する。二つの AI クライアントは同じ URL と token に個別接続する。Codex App を既に起動している場合、ユーザー環境変数を読み直すため再起動してから次を設定する。

```powershell
codex mcp add ngr-local --url http://127.0.0.1:8765/mcp/ --bearer-token-env-var NGR_MCP_HTTP_BEARER_TOKEN
```

他の MCP クライアントでも同じ URL に `Authorization: Bearer <token>` を付ける。token を URL や設定ファイル本文に直接書かず、クライアントの秘密環境変数から渡す。既存 stdio コマンド `neuron-graph-rag-mcp` は token なしで引き続き利用できる。HTTP サービスは一つの Python プロセスと DB 操作列を共有し、接続中のクライアントごとにモデルや DB を開かない。

## CUDA shortlist を明示的に使う

[CUDA API ガイド](cuda-shortlist-retrieval.md)に従い、固定 revision の E5 ONNX と v2-m3 モデルをローカルに配置する。CUDA 対応 PyTorch、`numpy`、`onnxruntime`、`tokenizers`、`transformers` は別途導入する。通常 wheel / MCP extra は CUDA やモデル weight を同梱しない。

```powershell
neuron-graph-rag-mcp --http `
  --database "$HOME/.ngrdb/knowledge.db" `
  --cuda-cache "$HOME/.ngrdb/shortlist.db" `
  --cuda-e5-snapshot "C:\models\e5\614241f622f53c4eeff9890bdc4f31cfecc418b3" `
  --cuda-v2-m3-snapshot "C:\models\v2-m3\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
```

モデル path は例であり、利用者が実際に配置した絶対 path を渡す。最初に MCP tool `update_cuda_shortlist_cache` を呼び、corpus の更新後も呼び直す。続いて `search_cuda_shortlist` に `contract_version: "ngr.mcp.feedback/v1"` と `query` を渡す。モデルは最初の検索時に読み込み、プロセス終了まで同じ retriever で再利用する。未設定、CUDA 不可、モデル欠落、古い cache は検索失敗として返る。通常の `search` は CUDA を使わず従来の feedback trace を返す。

現段階では token を共有する同一ユーザーのローカル利用を想定する。HTTP は OS ユーザーの身元を検査しないため、token を知る端末内の別プロセスも書き込み tool を使える。token が漏れたら新しい値を生成し、サービスとクライアントを再起動する。他端末への公開、remote deployment、GUI / トレイ、インストーラーは提供しない。CUDA 検索中は同期処理が HTTP の他の tool 呼び出しを待たせる。モデルの VRAM と E5 cache のディスク容量が別途必要になる。
