# Repository-native controlled corpus v3

これは longitudinal feedback-adaptation の後続検証に向けた、NGR repository 内の public かつ監査可能な controlled corpus です。評価 query、gold、feedback schedule、runner、gate、manifest、result、既定値は含めません。

## Corpus boundary

この corpus は次の三つの独立 cluster で構成します。

- `signal-stability`
- `boundary-recovery`
- `evidence-continuity`

各 cluster は同じ directory 内にある overview と four credit-ceiling document だけで構成されます。文書間 relation は、overview の `## 文書間の明示的な参照` に記載する同一 directory 内の相対 Markdown link だけから導出します。catalog、見出し、source URL、本文の語句から pseudo edge を導出しません。

各 cluster の source document は、credited feedback を 0、1、3、10 回まで受け取る場合を後続で比較できる ceiling を明記します。これは比較可能な corpus topology を提供するだけで、後続 evaluation の query、gold、採点、schedule、実行手順を定義するものではありません。

## Identity isolation

v3 の node ID、document path、source URL、credited edge identity は `repository-native-controlled-v3` 内で一意です。v1、v2、または evaluation artifact の identity を参照・再利用しません。

## License

本文は NGR repository の public documentation として Apache-2.0 で提供します。外部 corpus、別 repository、production adoption、または評価結果への一般化は主張しません。
