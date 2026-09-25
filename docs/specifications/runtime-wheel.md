# 通常 wheel の配布境界

## 目的と前提

通常 wheel は、利用者がインストールして実行する NGR の公開 API、CLI、optional MCP adapter を提供する。checkout に残る凍結済み実験 module は再現証拠の source path を維持する。既存 `docs/requirements.md` は過去の frozen manifest に登録されているため、その bytes を変えず、この追加仕様で配布境界を定義する。

## 要件

1. 通常 wheel は公開 package API、CLI の `demo` / `eval` / `benchmark`、optional MCP adapter と共有ローカル HTTP コマンド、利用ガイドで直接案内する opt-in 検索 API と、その実行に必要な module を含む。MCP SDK 自体は `mcp` extra に限る。
2. 実験専用 module は通常 wheel から除外する。checkout の legacy import path、frozen manifest、fixture、観測証拠、one-shot runner は維持する。
3. 配布対象は module 名の印象ではなく、公開 import root、CLI、MCP、文書化された API、動的 import の監査で決める。新しい runtime module の追加は配布リストと隔離 wheel 検証を同じ変更で更新する。
4. 新規実験専用コードは repository 直下の `experiments/` に置く。既存の凍結済み実験 source file は移動しない。
5. NGR の既定検索、feedback 設定、観測結果を変更しない。

## 受入検証

- 隔離した通常 wheel の file list に実験専用 module がなく、公開 API と opt-in API、CLI の既存3コマンドが動く。
- optional MCP の導入環境では、同じ wheel で tool 一覧、検索、source-use feedback、outcome、judgment の書込・検索が動く。
- checkout の core / 実験 test と CI が通る。配布前後の module 数と source bytes の差を記録する。

配布 module の理由、除外範囲、実測値、利用手順は[通常 wheel と checkout 実験コードの境界](../guides/runtime-wheel-boundary.md)を参照する。
