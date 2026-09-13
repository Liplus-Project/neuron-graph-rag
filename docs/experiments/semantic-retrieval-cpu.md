# multilingual-e5-small CPU 実測（2026-09-13）

CPUだけで多言語検索は動作した。ただし、既知 v5 development の問題をこの候補だけで解消できなかった。opt-in の試行用として提供し、標準検索は変えない。

証拠: [development-cpu.json](../../tests/evidence/semantic_retrieval/development-cpu.json)。これは既知の開発用診断であり、未見・確認的評価ではない。旧証拠を書き換えず、holdoutを開かず、GitHub RAGへ再送していない。

| 診断 | 既存hybrid top10内順位 | E5単独全件順位 | E5接続hybrid top10内順位 |
|---|---:|---:|---:|
| direct lexical | 1 | 1 | 1 |
| semantic paraphrase | 未到達 | 62 | 未到達 |
| relation | 1 | 1 | 1 |
| negative control | 2 | 2 | 2 |
| overexclusion control | 1 | 2 | 1 |

E5単独では overexclusion の正解順位が1→2へ下がった。hybridのtop10正解到達数は4/5のまま。日本語から英語へ一般的な意味の対応ができても、この抽象的な軸の区別を問うクエリの正解には届いていない。クエリ別補正・重み調整は行っていない。

独立した小規模probeでは、日本語→英語の返金期限、スペイン語→英語の配送日数、英語のパスワード言い換えが各1位。反復文の後ろに置いた潜水艦の救難信号を日本語で問う長文probeも12窓から1位となった。同じIDのバックアップ文書をパンの文書へ更新するとcache missが1増え、元の質問とのcosineは0.874→0.718へ低下した。

否定probeでは許可文書が禁止文書より上だったが、cosineは0.865対0.860と僅差で、禁止文書もdense/hybrid両方の結果に残る。論理的な否定理解・除外成功の証拠とはしない。probeは動作確認用の小規模な例で、一般性能推定には不十分。

Intel Core i7-14700K、RAM 63.7 GiB、Windows 11、threads=4、CPUExecutionProviderのみ。Python 3.12.14 / FastEmbed 0.8.0 / ONNX Runtime 1.30.0 / tokenizers 0.23.2 / NumPy 2.5.3。モデルrevisionは `614241f622f53c4eeff9890bdc4f31cfecc418b3`、ファイルごとのSHA-256は証拠JSONに記録した。

| 計測 | 秒 |
|---|---:|
| 初回download + load + query（事前smokeの端末計測） | 16.61 |
| download済cacheからload + 最初のquery（診断） | 1.74 |
| 93文書の初回埋め込み索引 + warmup query | 66.31 |
| 文書cache利用 E5単独query（5件範囲） | 0.039–0.041 |
| 文書cache利用 E5接続hybrid（5件範囲） | 0.086–0.210 |
| 既存hybrid（5件範囲） | 1.037–1.200 |

速度はこのCPUと93文書のcache利用条件に限る。既存denseには同じ文書cacheがなく、差をモデルそのものの速度優位と解釈できない。大量文書のスケーラビリティ試験ではない。

隔離環境はworkspace直下の `.semantic230-env`、モデルcacheは `.semantic230-model`、HF_HOMEは `.semantic230-hf`。DBはrunner内の新規 TemporaryDirectory へ作成し終了時に閉じて削除した。共有DBと旧モデルvolumeは使用していない。
