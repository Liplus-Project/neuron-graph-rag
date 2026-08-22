# Canonical SQLite judgment graph

## 正本境界

NGR の新規 judgment は SQLite の `judgments`、`judgment_revisions`、`judgment_relations` を machine-readable 正本とする。`nodes` と `edges` は既存 retrieval contract を保つ検索 projection であり、判断の lifecycle や provenance の正本ではない。既存 Wiki entry の本番移行は fixture による import / export 検証後の別操作とし、この実装は自動移行しない。

## Domain API

`NeuronGraphRAG.judgments` は add、update、supersede、archive、restore、hard delete を transaction 単位で提供する。update、supersede、archive、restore、hard delete は `expected_revision` による楽観的 concurrency check を要求する。relation target 不在、部分更新、stale revision、同一 predecessor の再 supersede は transaction 全体を失敗させる。

archive 済み judgment の再 archive と active judgment の再 restore は no-op として成功させず、fail closed にする。同一 revision の反復操作によって `updated_at` を暗黙更新しない。

supersede は新しい stable identity を作り、旧判断を archive し、successor から predecessor への `supersedes` relation と predecessor の `superseded_by` を保持する。superseded judgment は restore できない。

archive は通常 retrieval から外す論理的忘却であり、revision、provenance、relation は監査 API から取得できる。hard delete は明示操作で、archived、revision 一致、inbound relation と successor history がない候補にだけ許可する。

MCP の `write_judgment` は同じ domain API へ写像し、model に raw SQL を公開しない。既存 `search`、`record_source_use`、`record_outcome` の contract と既定値は変更しない。

## Portability and recovery

`tools/judgment_graph.py` は次を提供する。

- `export SOURCE_DB OUTPUT_JSON`: key と judgment / relation 順序を固定した UTF-8 LF JSON を出力する。
- `import INPUT_JSON DESTINATION_DB`: 全 identity と relation target を事前検証し、一つの transaction で再構築する。
- `backup SOURCE_DB BACKUP_DB`: SQLite backup API で transaction-consistent copy を作る。既存出力は上書きしない。
- `restore BACKUP_DB DESTINATION_DB`: integrity check 後に新規 destination へ復元する。既存 database は上書きしない。
- `integrity DATABASE`: SQLite integrity、foreign key、dangling relation、二重 successor、および `superseded_by` と archived lifecycle / successor の明示的 `supersedes` relation の双方向整合を fail closed に検査する。

backup は export より多くの revision history と retrieval / feedback state を保持するため、完全復旧の正本である。export は current judgment graph の決定論的 portability surface である。
