# Historical source verification

## 目的

frozen evaluation の hash registry は、過去の protocol を固定した時点の bytes を証明する。後続の正当な repository evolution を拒否せず、現在の working tree を過去の evidence と取り違えない。

## Source-of-truth boundary

- manifest 内の artifact hash は、その manifest の登録 commit にある `git show <commit>:<path>` の exact bytes と照合する。登録 commit は current manifest path を最後に変更した commit とし、current manifest bytes 自体がその commit と一致することを先に検証する。
- manifest が source commit、baseline commit、prior commit を明示する source registry は、その明示 commit の blob を照合する。
- commit object が存在し、現在の `HEAD` の ancestor であることを必須とする。未知 commit、非 ancestor commit、欠落 path、hash 不一致、working-tree manifest 改変は fail closed にする。
- 同名 path の current working-tree bytes は historical evidence ではない。検証後に evaluator が frozen fixture、gold、schedule、gate、source document を読む場合も、検証済み commit blob を使う。

## Newline contract

既存 protocol が raw-first の LF / CRLF portability を明記している場合だけ、raw blob hash 不一致後に whole-file LF -> CRLF または CRLF -> LF の一回の exact 変換を許す。本文差、mixed newline、bare CR、その他の byte 差は許可しない。明記のない registry は exact raw bytes だけを受理する。

## 非変更範囲

この境界修正は frozen manifest、fixture、packet、snapshot、observed result、登録 hash、source commit、観測回数、metric、gate、解釈を変更・再実行・再集計しない。変更するのは hash verifier が bytes を取得する時点だけである。
