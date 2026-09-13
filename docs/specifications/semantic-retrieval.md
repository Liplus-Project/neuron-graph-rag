# 任意の多言語意味検索

#230 の要件は、学習済みの小型多言語埋め込みを CPU で選択できる接続面と実測を提供すること。既存の FeatureHashingEncoder、既定の検索設定、凍結済み実験コードと証拠を変更しない。

`semantic_retrieval` は標準ライブラリだけで import / 構築できる。FastEmbed とモデルのロードは最初の非空検索で行う。モデルは multilingual-e5-small、ONNX CPUExecutionProvider を明示し、query と passage の接頭辞を区別する。モデル依存は `fastembed>=0.8,<0.9` の任意インストールとする。

文書は1000文字の窓、850文字の間隔で全体を覆い、接頭辞・特殊トークン込み512トークンを超える窓は再帰的に二分する。トークン検査用 tokenizer はモデル tokenizer の複製で、切り捨てと padding を無効にする。長すぎる質問は黙って切り捨てず ValueError にする。文書スコアは各窓の cosine の最大値（下限0）。長文ほど候補窓が増える偏り、窓境界で文脈が分かれる限界がある。

キャッシュはインスタンス内の本文 SHA-256 をキーにする LRU。既定256文書分で、バイト数の上限ではない。IDが同じでも本文が変われば再計算する。削除文書を検索候補へ戻すことはなく、古い内容の cache は eviction または clear_cache まで残る。モデル構成ごとに新しい backend/retriever を作成する。逐次実行用であり、並列検索の同期は提供しない。

`attach_semantic_retriever` は検索開始前に document engine と judgment graph の dense retriever を同時に差し替える。疎検索・グラフ・除外処理・既定重みは既存のまま。クエリ別・文書別の加点はない。否定の論理判定器ではなく、スコアは確率でもない。

検証は新規一時DB・独立モデルcacheを使用する。既知 v5 development の既保存結果から質問と正解IDのみを読み、新規証拠へ保存する。mixed holdout fixture、旧モデルvolume、共有DB、GitHub RAG の再検索は使わない。結果は探索的診断であり、未見性能や確認的再実験と呼ばない。効果が不足しても既定方式へ昇格しない。
