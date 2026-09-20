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

source commit `3ca2a3c53fe9911ce64f38a50b2ff0338be7bf74` に対し、preflight後のsingle one-shotを実行した。preflightはgold不在、holdout-bearing input 0件、登録query 0回、93文書、network disabled、共有DB 0件、両modelのrevision/file hash一致、synthetic forward 2回を記録した。

観測結果は次のとおりである。

| model | expected source rank | cutoff内 | runtime | peak RSS |
| --- | ---: | --- | ---: | ---: |
| `BAAI/bge-reranker-base` | 64 | いいえ | 182.877秒 | 2,008,219,648 bytes |
| `BAAI/bge-reranker-v2-m3` | 45 | いいえ | 602.301秒 | 3,133,009,920 bytes |

両modelともrank 20圏外だったため、固定classification policyによる有効な分類は `semantic_discrimination_bottleneck` である。これは既知development 1件に対する探索的診断であり、未見性能またはproduction統合の根拠ではない。result payload SHA-256は `1a7493a81e999df2047c8595a09748823c87ae2c04405de9f99f126beac55932`、preflight attestation file SHA-256は `405663d2e3388e079a76043d1e8370983982f8bff565801a898d7f340f79e9e1`、claim file SHA-256は `416ff3e48305df1f1619a71c6c665fecbf07b18cc6bce8e72295688033b0ed17`、result file SHA-256は `52f2289de5a2da675fd4f980c9752bda387101c2601184fbd01fdd59642dccd4` である。
