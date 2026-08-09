# Repository-native controlled corpus v2

これは Neuron Graph RAG repository 内で公開する controlled benchmark 用の source corpus です。evaluation code、evaluation query、gold answer、expected path、feedback schedule、runner、result artifact は含みません。

## 固定 source snapshot

この v2 は、公開済み commit `03527fd3eab76bbdec652eb74dfdec15b172cb34` の repository documentation を限定的に再構成します。この commit は v2 を追加する commit と別であり、各 source document は先頭に対応する public GitHub file URL を持ちます。

## 機械的な split 導出

source document は、固定の `Corpus split`、`Path ordinal`、`Corpus node ID`、`Source URL` を持ちます。split ごとに ordinal 0 から 3 の document を昇順に並べ、本文の `## 文書間の明示的な参照` にある同一 directory 内への相対 Markdown link だけを辿ります。これで各 split の 3 本の directed `mention` edge を導出します。

| Split | Node IDs | Document paths | Source URLs | Credited edge identities |
| --- | --- | --- | --- | --- |
| development | `v2-dev-0` -> `v2-dev-1` -> `v2-dev-2` -> `v2-dev-3` | `development-purpose.md`, `development-trace-boundary.md`, `development-separation.md`, `development-freeze-boundary.md` | 各 document の `Source URL` は別の fixed public file URL | `(v2-dev-0, v2-dev-1, mention)`, `(v2-dev-1, v2-dev-2, mention)`, `(v2-dev-2, v2-dev-3, mention)` |
| holdout | `v2-holdout-0` -> `v2-holdout-1` -> `v2-holdout-2` -> `v2-holdout-3` | `holdout-decision-boundary.md`, `holdout-fixture-boundary.md`, `holdout-channel-boundary.md`, `holdout-observation-boundary.md` | 各 document の `Source URL` は別の fixed public file URL | `(v2-holdout-0, v2-holdout-1, mention)`, `(v2-holdout-1, v2-holdout-2, mention)`, `(v2-holdout-2, v2-holdout-3, mention)` |

この表は split metadata であり、relation を作りません。relation の唯一の source は、各 source document 本文にある上記の相対 Markdown link です。README catalog、見出し、filename、source URL、表の文字列、fixture 用 pseudo edge から relation を作りません。

## 分離条件

development と holdout は node ID、corpus document path、source URL、credited `{source_id, target_id, edge_type}` identity を共有しません。各 split 内の 3 edge は source、target、edge identity が distinct です。

## License and controlled-benchmark scope

この corpus の本文は、この repository の公開 documentation を限定的に再構成したものです。Apache-2.0 で提供します。出典は fixed snapshot で追跡でき、作成時の対象は repository-native controlled benchmark に限られます。外部 corpus、別 repository、一般的な retrieval quality、production adoption への一般化を主張しません。
