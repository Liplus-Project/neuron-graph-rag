# 共有ローカル MCP を起動する

## 準備と起動

Python 環境に `pip install '.[mcp]'` を入れ、端末を一つ開いて次を実行する。

```powershell
neuron-graph-rag-mcp --http --database "$HOME/.ngrdb/knowledge.db"
```

既定の接続先は `http://127.0.0.1:8765/mcp/`。別ポートを使う場合は `--port 8766` を指定し、クライアントの URL も一致させる。端末で Ctrl+C を押して停止する。DB path を省略したときは `NGR_DATABASE`、次に `~/.ngrdb/knowledge.db` を使う。二つの AI クライアントは同じ URL に個別接続する。例:

```powershell
codex mcp add ngr-local --url http://127.0.0.1:8765/mcp/
```

既存 stdio コマンド `neuron-graph-rag-mcp` も引き続き利用できる。HTTP サービスは一つの Python プロセスと DB 操作列を共有する。接続中のクライアントごとにモデルや DB を開かない。

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

現段階ではローカル同一ユーザー用であり、他端末への公開、認証付き remote deployment、GUI / トレイ、インストーラーは提供しない。CUDA 検索中は同期処理が HTTP の他の tool 呼び出しを待たせる。モデルの VRAM と E5 cache のディスク容量が別途必要になる。
