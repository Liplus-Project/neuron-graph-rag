# Cross-encoder precision one-shot observation v3

## 範囲

Issue #147 は freeze merge commit `b762645d2521a3e23ac201b662ea1cbf25e2a260` だけを protocol input として、`github-ngr-cross-encoder-precision-v3` を一度だけ観測する。v3 の fixture、candidate 順、rank-only evaluator、11 hard gate、exact model revision、dependency lock、NGR default は変更しない。

v1/v2 の raw packet、evidence の semantic content、database、run output は v3 観測の入力にしない。predecessor evidence は v3 manifest に固定された byte hash との一致だけを確認する。既存 model cache は凍結 registry に列挙された model bytes の供給面としてだけ再利用し、claim 前に required file 全件の size と LFS SHA-256 または git blob ID を再検証する。shared database、既存 experiment database、github-rag-mcp、feedback/outcome、production serviceへ接続しない。

## Preflight と専用実行面

観測は v3 専用 venv、external run root、fresh SQLite database、fresh process、claim/result/error/transport を使用する。model cache 検証後は `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、local-files-only、`trust_remote_code=False`、CPU/float32/eval/inference-mode、batch size 8へ固定する。synthetic probe は登録 query を含まず、観測 stage の model inference count に含めない。

preflight は凍結 protocol、model bytes、dependency、offline probe、v3 専用/full tests、audit/probe、変更対象 lint、shared database の前後 SHA-256 を検証し、claim 前の evidence として独立 commit へ保存する。Windows では単一 process の resource 蓄積を観測入力と混同しないよう、同じ discover suite の全 test を test method ごとの fresh process で実行する。preflight と remote CI が通過するまで development claim を作らない。

## One-shot lifecycle

development claim を exclusive-createし、baseline primary/replay、base primary/replay、v2-m3 primary/replayを6個のfresh processとfresh SQLite databaseで各一度だけ実行する。worker raw packet は v3 専用 archive へ byte-preserving に保存する。rank-only evaluator で raw packet から一意な result を算出・再検証して即 archive し、全 development hard gate が pass した場合だけ holdout を一度開く。

claim 後の例外または gate failure では evidence を保存し、同じ protocol version の再試行、再評価、tuning、candidate/query/gold/gate/selection変更を行わない。model weight、cache、venv、fresh database は git へ追加しない。

## 状態

preflight evidence と one-shot result は未生成である。claim count、registered query execution count、observed-stage inference count は `0/0/0`、development/holdout はともに unobserved である。
