# GitHub RAG / NGR retrieval parity benchmark

## 目的

本 protocol は、NGR の document retrieval が github-rag-mcp の hybrid retrieval と同等以上かを、結果観測前に固定した共通 surface で判定する。Issue #63 / #64 の compatibility spike を置き換える successor だが、既存 fixture、capture、result は入力へ再利用しない。本 freeze は protocol、public source snapshot、query、gold、gate、writer、verifier だけを固定し、github-rag-mcp の live search、NGR の登録 query、development、holdout、採用判断を実行しない。

## 固定 source surface

source は public repository `Liplus-Project/github-rag-mcp` の commit `b26086bd91f9a8f87281b685b9ee73b2efaa50f5` にある Markdown 12 file 全体である。対象 path、Git blob SHA、UTF-8 content SHA-256、commit 固定 URL、本文を `github_retrieval_parity_v1.corpus.json` に保存する。GitHub 取得は `tools/acquire_github_snapshot.py` に閉じ、NGR core は固定 snapshot だけを受け取る。

更新追随は直前 commit `b4f1b44e324a4e8b189e63b8d82220c7486d6765` の同一 path surface から固定 commit への差分で確認する。両 snapshot は public source であり retrieval result ではない。NGR node は repository、path、commit、source URL、blob SHA、content SHA-256 を保持する。

## Query / gold

development と holdout は、expected、forbidden、relation seed を含む gold identity 集合が互いに disjoint である。各 split は direct lexical、semantic paraphrase、relation / linked context、directional negative control を一件ずつこの順で持つ。

両 retriever へ同じ query、`repo=Liplus-Project/github-rag-mcp`、`type=doc`、`top_k=5`、RRF、rerank、graph expansion 条件を渡す。github-rag-mcp は keyword axis と relationship axis を raw のまま保存し、NGR は final rank、entry / graph rank、source provenance、`SearchHit.explain()` を保存する。relation case の rank は keyword / graph のうち到達した小さい rank、他 cohort は通常 ranked result の rank を使う。

## Capture と一回性

development / holdout はそれぞれ `capture -> claim -> result` の三 artifact を持ち、すべて exclusive-create で上書きを拒否する。

- capture は frozen merge commit、stage、全 query の request、未加工 `search` response、全 keyword result の `vector_id` に対する未加工 stored-content response を保持する。
- stored content は `content_source=index`、`content_max_chars=8000`、`not_found=[]` を要求し、`path + "\n\n" + source body` の固定 prefix と照合する。これにより main URL の path 一致だけで古い index content を同一 source と扱わない。
- claim は capture SHA-256 を固定し、stage 開始後の再実行を拒否する。実行失敗も immutable failure result として保存する。
- holdout capture の登録は development result が全 hard gate を通るまで拒否する。

観測 Issue は freeze PR の squash merge commit を `--protocol-commit` に渡す。その commit が `origin/main` に含まれ、実行中の全 artifact byte が manifest hash と一致するときだけ capture 登録と stage 実行を許可する。

```powershell
$env:PYTHONPATH = "src"
git fetch origin main
python tools/run_github_retrieval_parity.py --audit
python tools/run_github_retrieval_parity.py --probe
python tools/run_github_retrieval_parity.py --register-capture development --input local-development-capture.json --protocol-commit <freeze-merge-sha>
python tools/run_github_retrieval_parity.py --stage development --protocol-commit <freeze-merge-sha>
python tools/run_github_retrieval_parity.py --verify development
```

`--probe` は登録 query / gold を実行せず、登録外 synthetic placeholder だけで writer から verifier までを round-trip する。

## Hard gate

primary gate は protocol integrity、source / provenance integrity、fresh isolated DB 二回の deterministic replay、update following、direct case 非退行、negative control 非退行と forbidden source 不在、cohort ごとの MRR / Hit@k 非退行、両 retriever の expected source top-k 完全性、NGR source / path explanation と relation graph path を個別に保持する。平均だけで direct / negative regression を隠さない。

latency、database size、replay elapsed time は記録するが、local NGR と remote github-rag-mcp の deployment 差を混同しないため hard gate にしない。

## 安全境界

- NGR は各 replay ごとに fresh temporary SQLite を作り、終了時に削除する。共有 `~/.ngrdb/knowledge.db` と既存 experiment DB を開かない。
- search trace は temporary DB 内だけに残し、source-use、success feedback、delayed outcome へ接続しない。
- development failure、hard gate failure、capture、claim、result を変更・再生成しない。失敗時は holdout を開かない。
- supporting result でも NGR default、local MCP config、github-rag-mcp service、repository integration を変更しない。
- issue、PR、review、comment、release、commit diff の取得 parity は本 document corpus benchmark の外である。
