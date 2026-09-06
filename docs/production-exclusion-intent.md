# Production exclusion intent

## 目的

`NeuronGraphRAG.search()` が検索したい肯定条件と、最終結果へ含めない明示的な除外条件を分離する。BM25、dense retrieval、graph propagation、fusion、precision control のどの経路から候補が来ても、同じ最終境界で除外を適用する。

## Query contract

引用外の明示 marker だけを構文として扱う。英語は `without`、`excluding`、`exclude`、`except`、`avoid`、`but not`、日本語は区切られた `除外:` と `<clause>を除外`、`<clause>を除く`、`<clause>は含めない` を対象にする。marker 前を positive query、各 marker の対象を exclusion clause とする。

単独の `not`、`notebook` のような語の一部、引用内の `"not supported"`、`not necessarily` のような曖昧な否定は分解しない。除外節だけで positive query が空になる入力と、対象が空の marker は `ValueError` とする。この parser は保守的な明示構文の契約であり、一般的な否定スコープ解析ではない。

## Candidate decision boundary

positive query を sparse / dense retrieval と graph propagation に渡し、従来どおり全 node の final rank を確定する。precision control が有効な場合はその accepted 候補列を受け取る。その後、全候補へ exclusion decision を適用し、通過列へ最後に `limit` を適用する。これにより、上位候補が除外されても安全な下位候補で枠を補充できる。

判定入力は query から分解した exclusion clause と candidate node の text だけである。比較は Unicode-aware な語境界と正規化した phrase を使い、`rust` で `trust` を除外しない。candidate 内のその出現自体が `without Rust`、`Rust を使わない` のように明示的に否定されている場合は一致と扱わない。一つでも否定されていない出現があれば除外する。gold、expected、forbidden、case ID、評価 label は入力にしない。

## Explanation and compatibility

除外された hit は `SearchTrace.hits` や永続 retrieval row に含めず、trace diagnostics の `exclusion_intent.decisions` に node ID、判定、一致した clause、理由を保持する。通過 hit の `SearchHit.explain()` は `exclusion_intent` に通過判定を保持する。順位・score・path・precision-control decision は変更しない。

exclusion clause が0個なら parser は original query をそのまま positive query とし、従来と同じ retrieval 経路を通す。hit と diagnostics へ exclusion field を追加しない。公開 `search(query, limit, now)` の引数、SQLite schema、feedback contract、`search_channels()` は変更しない。

## Verification boundary

`tests/fixtures/exclusion_intent_v1.cases.json` は実装前に固定した parser / retrieval case である。英語・日本語、複数除外、除外なし・除外だけ、marker の語内出現、引用・曖昧な否定、文書側の否定、limit 補充、graph / precision control 後の除外を回帰する。fixture の成功は、小規模で決定的な corpus 上の契約確認に限定し、github-rag-mcp との統合性能や一般的な自然言語理解を支持しない。
