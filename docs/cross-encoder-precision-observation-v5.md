# Cross-encoder precision observation v5

## 実行境界

Issue #155 の one-shot observation は freeze merge commit
`d5c25d7998d634cac0aa96511f59a9cce0b7725a`を唯一のprotocol inputとし、
`src/neuron_graph_rag/cross_encoder_precision_v5_observation.py`がWindows hostから
WSLC `2.9.4.0`を編成する。freeze済みfixture、evaluator、candidate、gate、
Containerfile、dependency contractは変更しない。

mutable run rootは専用volume
`github-cross-encoder-precision-v5-runtime`だけである。source、model cache、
fresh SQLite、worker output、claim/result/error/archive/transportをvolume内の
distinct pathへ置き、共有Windows DBはSQLite接続せずFILE_SHARE-aware byte readerで
SHA-256だけを取得する。frozen sourceと検証前のWindows model sourceはread-only bind、
git evidenceは専用output bindだけを使用する。

## Preflight

`tools/run_cross_encoder_precision_v5_wslc.ps1 preflight`はtarget volumeのabsentを
確認してexclusive-createし、次をclaim前に記録する。

- exact freeze commitとv1-v4 predecessor evidenceのbyte immutability
- `--no-cache` rebuild後のexact image IDと`--network none` validator
- exact 26 dependency、CPU-only torch、platform/runtime metadata
- 2 exact model revisionのWindows source/volume copy後の全required file identity
- CPU/float32/eval/inference-mode/batch8のmodel別synthetic forward
- v5 test、audit/probe、full suite、変更対象Ruff
- development claim/query/observed inference/result=`0/0/0/0`

preflight evidenceはcommit/pushし、そのcommitのremote check-runが全てgreenになるまで
development claimを作らない。preflight setup failureはexclusive error evidenceとして保存し、
同versionを再試行しない。

## One-shot lifecycle

`tools/run_cross_encoder_precision_v5_wslc.ps1 run`はremote CIを再検証してから
development claimを一度だけ作る。frozen順序のbaseline/base/v2-m3 primary+replayを
6 fresh container process / 6 fresh SQLite DBで実行し、raw packetをv5 rank-only
evaluatorで一意に再計算して即archiveする。全hard gateを通るcandidateがある場合だけ
holdoutを同じ方法で一度開く。selectedなし、failed、errorではholdoutを開かない。

claim後の例外、native signal、OOM、dependency/runtime/container failureはerror archiveし、
timeoutをretryへ変換しない。pass/fail/errorのいずれでもraw packetとgit evidenceの
byte identity、fresh container/process/DB identity、primary/replay determinism、state不変、
shared DB SHA不変をterminal evidenceへ残す。

## 観測結果

implementation commit `8cd56af049fd0afa7bb3b3602f31939df182480b`からpreflightを
一度開始した。exact `--no-cache` build自体はreturn code 0で完了したが、rebuilt image ID
`sha256:00c544ce37579c40eb328acf63269a38874dc21c6b2ccefc2c3a19121a6a9d14`が
freeze registryの
`sha256:bc105cebf12e144ef0e178b18b3ff95367bf7567113fdfe524c6c7c2de2b4dd2`
と一致しなかったため、preflightはfail-closedで停止した。

v5専用volumeは未作成、development / holdout claim=`0/0`、registered query、
model copy/load/inference、observed result/errorはすべて`0`である。shared Windows DB SHA-256は
preflight前後とも
`84a3fc590eee990579e3ef8130294129934fe93e25f18ac249eece19813c261e`で不変だった。
developmentとholdoutを開かず、性能は`not assessed`、同version retryなしで停止した。
