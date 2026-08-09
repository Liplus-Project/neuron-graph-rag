# Longitudinal feedback trajectory evaluation

## 目的

Issue #55 は repository-native controlled corpus v3 上で、feedback count が 0、1、3、10 と増えるときの retrieval trajectory を固定 control と credited-path-only treatment で測定する。対象はこの controlled corpus の graph traversal に限られ、外部一般化、production adoption、Agent E2E 効率を主張しない。

## 結果を読まない凍結境界

この protocol は v3 source commit `94c8bc250b7352e3009eeee1b353c3aec677bfb7` だけを source corpus input とする。既存 evaluation の fixture、gold、schedule、manifest、gate、result は入力にも参照にも使わない。事前監査は source node、document path、source URL、credited edge identity に限る。

`fixture` は全 15 source document の raw SHA-256 を持つ。検証は raw hash を優先し、whole-file LF/CRLF の完全変換だけを許容する。本文の正規化、JSON canonicalization、mixed newline、bare CR は許容しない。

development は `signal-stability`、holdout は `boundary-recovery`、trajectory audit は `evidence-continuity` に固定する。三者の node、path、source URL、credited edge identity は相互に重複しない。

## 測定モデル

各評価 split は overview から明示 link された credit document だけを候補とする。control はすべての edge を不変にする。treatment は登録済み overview-to-credit-10 edge だけに `0.5 * feedback_count` を加え、他の edge は変更しない。credit ceiling から決まる候補順序にこの加点を適用し、gold target の reciprocal rank を MRR として記録する。

## 実行順序

1. `fixture`、`gold`、`schedule`、`gate`、`manifest` と exclusive output path を commit/push で凍結する。
2. `python tools/run_longitudinal_feedback_trajectory.py development` を一回だけ実行する。PASS / FAIL のどちらでも create-only development output を保存する。
3. development が PASS のときだけ同じ runner の `holdout` を一回だけ実行する。holdout は development output の manifest hash handoff を検証する。
4. observed output 生成後は evaluator、fixture、gold、schedule、manifest、gate、docs を変更しない。FAIL output も保存し、再実行せず停止する。

## Gate

- treatment MRR は 0 から 10 で strict improvement し、途中 point で後退しない。
- control MRR は各 point で non-regression、MRR は 1.0 ceiling 以下。
- control に edge mutation はなく、treatment の mutation は登録 credited edge だけである。
- source integrity、role identity isolation、exclusive output、development-to-holdout manifest handoff が成立する。
