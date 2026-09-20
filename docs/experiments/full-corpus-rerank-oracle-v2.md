# Full-corpus rerank oracle v2

## 位置づけ

この実験は、#234 で公開済みの development 1件について、93文書すべてを固定cross-encoderで採点し、候補生成と意味識別のどちらが主なボトルネックかを探索的に診断する。未見性能の評価や確認的評価ではない。v1のrankと分類はholdout-bearing mixed-stage fixtureを登録実行がparseしたため無効であり、v2の判断材料には使わない。

## holdout不在契約

queryとexpected sourceは別のdevelopment-only JSONへ固定する。fresh WSLC volumeにはallowlistしたrunner、projection helper、v1から継承する固定計算helper、query、schema、manifest、93文書corpus、pinned model registryだけを個別mountからcopyする。mixed v5 query/gold、holdout fixture、v1 runtime/evidence、共有DBはcopyしない。

preflightと2つのworkerにはgold JSONをmountしない。workerはqueryだけを読み、93文書を採点してraw rankingを保存する。両workerの終了後、finalizer containerだけにdevelopment-only goldをread-only mountし、expected source rankとcutoff 20による分類を算出する。goldはvolumeへcopyされない。

## 固定条件

- model: `BAAI/bge-reranker-base@2cfc18c9415c912f9d8155881c133215df768a70` と `BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`
- projection: Unicode codepoint 480、overlap 80、先頭から末尾まで
- aggregation: 文書内chunkのmaximum raw logit
- cutoff: 20
- runtime: Linux x86_64、CPU 4 threads、network disabled、fresh volume
- corpus: `github_retrieval_parity_v4.corpus.json` の93文書

## 実行境界

`tools/run_full_corpus_rerank_oracle_v2_wslc.ps1 preflight` はfresh volumeを作り、forbidden path不在、exact allowlist、93文書、model revision/file hash、network disabledを確認し、各modelでsynthetic forwardを1回だけ行う。登録queryは実行しない。preflight成功後の `run` はappend-only claimを書き、baseとv2-m3をそれぞれ一度だけ実行する。失敗時はerror evidenceを残し、同じv2 one-shotを再試行しない。

CIは `audit`、`probe`、unit testだけを実行し、model downloadまたは登録query推論を行わない。結果は `tests/evidence/full_corpus_rerank_oracle_v2/` にappend-onlyで保存する。

## 観測結果

結果はまだ登録されていない。result-free protocolをcommit/pushした後にpreflightを通し、一度だけ登録実行する。
