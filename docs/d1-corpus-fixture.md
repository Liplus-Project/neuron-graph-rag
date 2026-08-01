# D1 corpus fixture

## 目的と境界

`tests/fixtures/d1_liplus_wiki.json` は github-rag-mcp の本番 D1 から読み取り専用で取得した、小規模な NGR 統合 fixture である。`search_docs` を文書の正本とし、FTS5 virtual table と shadow table は取得しない。`doc_edges` は fixture 内に両端 node がある edge だけを変換する。

D1 は検索用の損失ありスナップショットである。content は切り詰められることがあり、binary / patchless file は diff index に存在しない場合がある。byte-exact な履歴復元の正本は GitHub とする。

## 変換

- `search_docs.vector_id` → `DocumentNode.node_id`
- `search_docs.content` → `DocumentNode.text`
- `content` と派生 `content_fts` 以外の source 列 → node metadata
- `doc_edges.src_vector_id / dst_vector_id` → edge endpoint
- `doc_edges.edge_kind` → `TypedEdge.edge_type`
- node confidence、edge weight、edge factuality → `1.0`

fixture edge の `metadata.source_record` は現在の `mention` と将来の typed edge を source record として区別する。NGR core の `TypedEdge` 自体へ metadata は渡さない。

## 読み取り専用取得

Wrangler の OAuth 認証は host 側で行う。token を引数、環境変数の出力、fixture、provenance へ含めない。取得 tool は任意 SQL を受け取らず、内部 query も単一の `SELECT` / `WITH` 以外を拒否する。各 Wrangler 応答について `rows_written=0`、`changes=0`、`changed_db=false` を検証し、一つでも崩れたら出力しない。

workspace root から次を実行する。

```powershell
python tools/acquire_d1_fixture.py `
  --repo Liplus-Project/liplus-language `
  --type diff `
  --type wiki_doc `
  --per-type-limit 3 `
  --output tests/fixtures/d1_liplus_wiki.json `
  --provenance-output tests/fixtures/d1_liplus_wiki.provenance.json `
  --known-gap "github-rag-mcp#178: forward-gap backfill status at capture time" `
  --wrangler-project C:\path\to\github-rag-mcp
```

選択順は `(repo, type, updated_at, vector_id)` で固定する。同一 snapshot と引数から fixture JSON は byte-identical になる。取得時刻は provenance report だけに置く。credential らしい文字列は、node / edge だけでなく source 引数と既知 gap を含む最終 fixture / provenance 全体で `[REDACTED_SECRET]` へ決定論的に置換する。fixture、provenance、合計の置換件数を report に残す。

完全 export や作業中の raw JSON は commit しない。`.gitignore` は `*.d1-export.json`、`artifacts/d1/`、`tests/fixtures/.full-*` を除外する。

## Provenance と coverage 監査

`tests/fixtures/d1_liplus_wiki.provenance.json` は次を保持する。

- source database、repo、type、source schema fingerprint
- type 別の source count、最古 / 最新 `updated_at`、空文字 sentinel を除外した distinct commit count
- fixture の node / edge 数と欠損 endpoint 除外数
- query ごとの zero-write evidence
- 取得日時、既知 gap、完全性の限界、redaction 件数

`known_gaps` は取得時点で未解消の gap だけを保持し、確認済みの gap がない場合は空配列にする。

backfill 完了後は同じ引数で別 path へ再取得し、次で coverage を比較する。

```powershell
python tools/compare_d1_provenance.py previous.provenance.json current.provenance.json
```

`diff` の `source_count_delta`、`distinct_commit_count_delta`、`newest_extended` を監査する。比較結果を確認してから管理対象 fixture と provenance を置き換え、全テストを再実行する。

## 統合検証

`tests/test_d1_fixture.py` は実 fixture を SQLite に ingest し、文書検索、`updated_at / commit_date` metadata、`mention` edge の graph activation、利用 node に対する success feedback と edge reinforcement を再現する。
