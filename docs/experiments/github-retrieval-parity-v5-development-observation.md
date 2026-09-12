# GitHub RAG / NGR retrieval parity v5 development観測

## 結論

freeze commit `ef1c14d573fde04913b86b9ef2f8ad8b8621497c`に対するdevelopment観測は、固定5 caseを各1回だけ登録して完了した。10 hard gateは6件pass、4件failであり、developmentの範囲でもNGR優位または同等を支持しない。holdoutは閉鎖したままで、default adoption、一般化、production置換を主張しない。

失敗gateは`negative-safety`、`over-exclusion-safety`、`cohort-non-regression`、`expected-source-completeness`である。NGRはrelation-linkedのpath completenessとover-exclusion controlでgithub-rag-mcpを上回った一方、negative controlのMRR / nDCG@10は下回り、semantic paraphraseは両者ともexpected sourceを取得できなかった。github-rag-mcpはnegative controlでforbidden sourceを返し、over-exclusion controlでprotected-safe sourceを保持できなかった。

## 一回性とpreflight

- clean-env import smoke: GitHub RAG request、NGR search、SQLite open、artifact createはすべて0
- result-free audit: source 93、development 5、holdout 5、registered observation 0
- queryless preflight: exact path 93、stored content / public source SHA-256一致93、duplicate / missing / out-of-prefix / not-found / driftはすべて0
- registered development query: 5件を各1回、合計5回
- holdout query: 0回
- request: `repo=Liplus-Project/neuron-graph-rag`、`type=doc`、`path_prefix=corpora/github-retrieval-parity-v4/`、`top_k=10`、`fusion=rrf`、`rerank=true`、`graph_expand=true`、`graph_hops=2`

固定10 queryはdevelopment 5件とholdout 5件の事前分割である。本観測はdevelopmentだけを開き、holdout 5件を送信していない。

## Metrics

| cohort | github-rag-mcp MRR / nDCG@10 / Recall@10 | NGR MRR / nDCG@10 / Recall@10 | relation path completeness (GitHub / NGR) |
|---|---:|---:|---:|
| direct lexical | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 |
| semantic paraphrase | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 1 |
| relation linked | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 1 |
| negative control | 1 / 1 / 1 | 0.5 / 0.6309297535714575 / 1 | 1 / 1 |
| over-exclusion control | 0 / 0 / 0 | 1 / 1 / 1 | 1 / 1 |

NGR replayは同じ93文書と固定`EngineConfig()`をfresh temporary SQLiteへ適用し、2回のnormalized result SHA-256が一致した。elapsedは7.507563700004539秒と7.232735099998536秒、database sizeは各1,531,904 bytesである。latencyは記録のみでhard gateではない。各databaseはreplay後に削除され、feedbackは接続していない。shared `C:\Users\smile\.ngrdb\knowledge.db`は開いていない。

## Evidenceと検証

| artifact | SHA-256 |
|---|---|
| `development.preflight.json` | `fbc74c135127ad972142e67a521cfd450050e1ea1569b8ae77e657596ee65f19` |
| `development.capture.json` | `92b93069fb3434d9f89e8c3dd7f444f503c77b9ca1a44a92ac08a32d97e56600` |
| `development.claim.json` | `9739ccb18d6eb2d33a6fdfcf31820ec663fe7c5b479609dccd4aaae65d51b7f8` |
| `development.observed.json` | `9fdb094f246dcd2293481c46635cedd068fc0790c986a20163519e7933632e7c` |

v5 manifestが固定するprotocol moduleと元testのbytesは変更しない。観測後verificationはfreeze commitをtemporary checkoutへ展開し、そこへ4 evidenceをcopyして凍結validatorを実行する。

```console
python -I tools/verify_github_retrieval_parity_v5_observation.py
python tools/run_tests.py normal
python tools/run_tests.py all
```

alphabetical-firstの専用test routerは、通常のfull discoveryと選択runnerの双方で元v5 test moduleの正規import名だけを対象にする。元loaderで11件を確認してからfreeze temporary checkoutへroutingし、11件すべての成功を必須にする。その他のmoduleは元loaderへそのまま委譲し、現rootの観測専用testは4 evidence、preflight、result、lifecycle、metrics、holdout不在を検証する。他のproduct / shared / historical testはdiscover結果から除外しない。
