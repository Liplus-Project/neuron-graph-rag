# Decision Wiki 差分取り込み

[製品仕様](README.md)

## 境界

既存の judgment store（例: `~/.ngr/db/knowledge.db`）を作り直さずに、Li+ と NGR の Decision Structure Wiki のうち索引に載った判断の差分だけを取り込む。索引に載らないページは読まない。索引と identity の規則は [Decision Wiki pilot migration](../decision-wiki-pilot-migration.md) と同じで、identity は `<repository-name>:<slug>`。

書き込みは MCP の `write_judgment` と同じ domain API（`JudgmentGraph.add` / `supersede` / `update` / `archive`）を通す。judgment、revision、relation と、検索に使う `nodes` / `edges` の projection は同じ transaction で更新される。DB ファイルはその場で開いて更新し、作り直しや差し替えはしない。存在しない DB は作らない。

書く版の provenance は pilot と同じ `page_body`、`page_sha256`、`repository`、`source_state`、`wiki_commit`、`wiki_url` に、`import_mode: incremental` を加えたものとする。

## 判定

取り込み記録は、provenance に `page_sha256` と `wiki_url` を持つ最新の revision とする。pilot と差分取り込みの両方がこの形で書く。

- **store に無い判断** → add。Edges の `supersedes` の対象が active なら supersede で対象を archive する。参照先が先に存在するよう、新しい判断どうしは依存の順に書く。依存が循環する新しい判断は書かない。
- `supersedes` の対象が既に archive されている、または同じ計画の中で先に別の判断に置き換えられた場合、その辺は書かずに理由とともに報告する。
- active な置き換え対象が 2 件以上ある判断は書かない。store の整合検査は 1 つの successor に predecessor を 1 件だけ認めるため。
- **store にあり、wiki 側が変わった判断**（ページの hash または索引の resolution が取り込み記録と違う）→ store の現在の版が取り込み記録のままのときに限り、update で新しい版を足す。store 側で書き直されていれば書かず、`changed_on_both_sides` に出す。取り込み記録の無い判断も同じ一覧に出す。
- 版を足す対象でも、`supersedes` の辺が store と違う、同じ計画で後継に置き換えられる、または store で inactive の場合は書かない。
- 索引から消えた取り込み済みの判断は `absent_from_index` に出す。それと page hash か resolution が一致する新しい identity は slug の変更とみなし、重複を作らないよう書かずに報告する。
- 新しい判断で索引の state が archived / superseded、かつこの計画に後継が無いものは、add の後で archive する。それ以外の索引 state と store の lifecycle の食い違いは `lifecycle_mismatch` に出すだけで、書き込みはしない。

書かなかった判断は理由とともに `blocked` に出す。

## 辺

store の relation type registry で active な type だけを取り込む。それ以外のラベル（`relates to` など）、読めない行、対象が複数ある行、対象が存在しないか書かれない行は取り込まない。その行と理由を `skipped_edges` に、理由ごとの件数を `skipped_edge_reasons` に出す。

辺の対象は、ラベルの後ろで最初のダッシュ（`—`）より前に書かれたリンクから取る。そこにリンクが無ければ行全体から取る。

## 実行

```powershell
python tools/import_decision_wikis_incremental.py `
  --liplus-wiki C:\path\to\liplus-language.wiki `
  --ngr-wiki C:\path\to\neuron-graph-rag.wiki `
  --ngr-index C:\path\to\neuron-graph-rag\docs\Decision-Structure.md `
  --database $HOME\.ngr\db\knowledge.db
```

- 既定は dry run。読み取り専用の接続で計画を作って JSON で出力し、DB には書かない。
- `--write` を付けたときだけ書く。書く操作が無ければ何もしない。書く操作がある場合は、SQLite の online backup API で DB と同じフォルダに `<DB名>.pre-wiki-increment-<UTC時刻>.db` を作り、その integrity を確認する。次に `BEGIN IMMEDIATE` の 1 transaction の中で計画を作り直して適用し、commit の前に `tools/judgment_graph.py integrity` と同じ検査を通す。どこかで失敗すれば全体を rollback する。
- 書いた版は取り込み記録になるため、wiki が変わらない限り再実行しても何も書かない。
