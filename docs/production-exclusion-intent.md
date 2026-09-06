# Production exclusion intent

## 目的

`NeuronGraphRAG.search()` が検索したい肯定条件と、最終結果へ含めない明示的な除外条件を分離する。BM25、dense retrieval、graph propagation、fusion、precision control のどの経路から候補が来ても、同じ最終境界で除外を適用する。

## Query contract

引用外の明示 marker だけを構文として扱う。英語は `without`、`excluding`、`exclude`、`except`、`avoid`、`but not`、日本語は区切られた `除外:` と `<clause>を除外`、`<clause>を除く`、`<clause>は含めない` を対象にする。marker 前を positive query、各 marker の対象を exclusion clause とする。

単独の `not`、`notebook` のような語の一部、引用内の `"not supported"`、`not necessarily` のような曖昧な否定は分解しない。除外節だけで positive query が空になる入力と、対象が空の marker は `ValueError` とする。この parser は保守的な明示構文の契約であり、一般的な否定スコープ解析ではない。

## Candidate decision boundary

positive query を sparse / dense retrieval と graph propagation に渡し、従来どおり全 node の final rank を確定する。precision control が有効な場合はその accepted 候補列を受け取る。その後、全候補へ exclusion decision を適用し、通過列へ最後に `limit` を適用する。これにより、上位候補が除外されても安全な下位候補で枠を補充できる。

判定入力は query から分解した exclusion clause と candidate node の text だけである。まず Unicode-aware な語境界で正規化した phrase を比較する。phrase が表層一致しない場合は、指示語の `the`、操作語の `returning`、種別語の `file` / `document` を除いた token / identifier 集合が候補にすべて存在する時だけ一致とする。数詞 `three` / `ten` と数字 `3` / `10`、複数形は決定的に正規化する。これにより `returning the three-credit recovery file` は `Boundary recovery credit 3`、`the ten-credit ceiling document` は `Boundary recovery credit 10` と一致する一方、`rust` で `trust` は除外しない。

candidate 側の否定は対象 mention に接する固定構文だけを認める。英語の前置 `without Rust`、`no Rust`、`does not use Rust`、後置・受動否定 `Docker is not used`、`legacy tokens are not accepted`、suffix の `Docker-free` を対象にする。後置否定は `is / are / was / were + not` と固定分詞が mention 直後にある場合だけとする。mention が主語となる `Docker does not use root privileges` は Docker 自体の不使用を意味しないため、否定された mention として扱わず候補を除外する。日本語は mention 直後の助詞に続く `使わない` / `使用しない` / `利用しない`、`使われていない` / `使用されていない` / `利用されていない` 等の固定形を対象にする。離れた否定語は判定に使わない。同じ exclusion clause に否定された mention と否定されていない mention が混在する場合、後者を `matched_exclusions` に保持して候補を除外し、前者も `negated_mentions` に保持する。gold、expected、forbidden、case ID、評価 label は入力にしない。

## Explanation and compatibility

除外された hit は `SearchTrace.hits` や永続 retrieval row に含めず、trace diagnostics の `exclusion_intent.decisions` に node ID、判定、一致した clause、理由を保持する。通過 hit の `SearchHit.explain()` は `exclusion_intent` に通過判定を保持する。順位・score・path・precision-control decision は変更しない。

exclusion clause が0個なら parser は original query をそのまま positive query とし、従来と同じ retrieval 経路を通す。hit と diagnostics へ exclusion field を追加しない。公開 `search(query, limit, now)` の引数、SQLite schema、feedback contract、`search_channels()` は変更しない。

## Verification boundary

`tests/fixtures/exclusion_intent_v1.cases.json` は初回実装前に固定した parser / retrieval case である。英語・日本語、複数除外、除外なし・除外だけ、marker の語内出現、引用・曖昧な否定、文書側の前置否定、limit 補充、graph / precision control 後の除外を回帰する。`tests/fixtures/exclusion_intent_candidate_negation_v1.cases.json` は修正前に、英語の後置・受動否定と `-free`、日本語の受動否定、unsafe 対照、対象なし、否定・肯定混在を3 scenario・15文書に固定する。加えて、既存 source-grounded v3 で観測された2つの development negative query と現行 corpus / gold を read-only に読む bug regression で、表現差のある禁止文書が最終 hit から外れ、diagnostics に判定が残ることを検証する。gold / fixture の期待値は test assertion にだけ使い、production 判定に渡さない。既存 v3 query、corpus、gold、manifest、evidence bytes は変更しない。fixture の成功は、小規模で決定的な corpus 上の契約確認に限定し、github-rag-mcp との統合性能や一般的な自然言語理解を支持しない。
