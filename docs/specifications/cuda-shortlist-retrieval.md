# 任意の CUDA shortlist 検索

#255 は既存の E5 cache と shortlist / NLME 順位計算を再利用し、NVIDIA CUDA 上の v2-m3 再順位付けを明示的に選ぶ API を追加する。`CudaShortlistRetriever` と `attach_cuda_shortlist_retriever()` が公開入口であり、engine instance には `update_cuda_shortlist_cache()` と `search_cuda_shortlist()` を追加する。既定の `search()`、CPU 用 API とその順位計算は変更しない。

`LocalPinnedCudaV2M3` は呼び出し元が渡したローカル snapshot のみを使い、FP32、`max_length=512`、batch size 8 で CUDA model と CUDA tensor を使う。import や adapter の構築では PyTorch、モデル、CUDA context をロードせず、最初の明示検索でロードする。backend instance はモデルを保持して次の検索で再利用する。`close()` でモデル参照を外し、CUDA cache を解放する。PyTorch や他の CUDA 利用者が保持する割当、CUDA context、OS による VRAM 利用はこの操作では解放されない。完全な回収にはプロセス終了が必要になり得る。

CUDA 不可、必須モデルファイル欠落、モデルロード失敗、OOM は例外として伝える。検索の timeout / cancel は既存 retriever と同じ batch 境界で検査し、実行中の CUDA kernel を強制停止するものではない。CPU または既定順位へ自動的には切り替えない。結果の `diagnostics` は既存 cache fingerprint / pair 数 / 時間に加え CUDA device 名、search 区間の peak allocated / reserved bytes を含む。GPU peak は当該プロセスの PyTorch allocator の値であり、他プロセスの VRAM は含まない。

通常 wheel は軽量 Python module のみを含む。CUDA 版 PyTorch、transformers、ONNX E5 依存と model weights は必須依存に追加せず、利用者が別途準備する。revision 定数と cache identity は保持するが、API はモデルファイル hash を検証しない。固定 v3 の再現には corpus と E5 / v2-m3 ファイル SHA-256 を固定 manifest と手動照合する。実測の範囲と値は実験記録に分ける。
