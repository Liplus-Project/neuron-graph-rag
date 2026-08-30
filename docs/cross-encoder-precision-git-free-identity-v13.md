# Cross-encoder precision git-free protocol identity v13

この文書はv13 successorの分割requirements specであり、目的、前提、制約、受入境界の正本である。
中央`docs/requirements.md`はfrozen v8 protocolのbyte registryに含まれるため変更しない。

## 目的

v12 fresh runtimeはmodel copyとsynthetic probe 2 forwardまで成功したが、development one-shotの最初の
claim containerでterminal errorとなった。frozen evaluatorの`verify_protocol_commit`がaccepted v8 image内で
`git` executableを起動しようとし、claim fileのexclusive-create前に`FileNotFoundError`となったためである。

v13は性能観測ではない。v12 raw / preflight / terminal evidenceと実装15 artifactsをbyte不変の
predecessorとして保存し、accepted image内のgit executableやsubprocessに依存せず、frozen protocol commit
identityとexact source bytesを検証できるかだけを固定するresult-free one-shot protocolである。

## git-free identity境界

新protocolは`github-ngr-cross-encoder-precision-v13`、専用volumeは
`github-cross-encoder-precision-v13-commit-freeze`、rootは`/opt/ngr-v13/commit-freeze`である。harness
source、frozen protocol source、model-cache sentinel、commit identity reportはroot配下のdistinct absolute
POSIX pathへ置く。future observation volume `github-cross-encoder-precision-v13-runtime`はfreeze前後とも
absentを維持する。

verifierは次を直接照合する。

- frozen v8 protocol commit literal `d2fdf7720e2a9dde7e8d666cf4fd9f314fd3d12f`
- v12 predecessor archive commit `20a20007bdc4e25a6146a401e147c4c4552aa2a1`
- frozen manifest bytes、23 artifact hashes、artifact registry identity
- frozen corpus registry、commit identity、24 documentsのexact bytes
- complete source identity schema

frozen v8 wrapperとdistinct `_BASE`へv11 parameterized root bindingを適用した後、同じgit-free verifier
callableをwrapper、base、evaluation wrapper / base、nested protocol evaluator / baseのactual object graphへ
明示bindする。wrong commit、incomplete identity、manifest / artifact / corpus mismatchはfail-closedである。
accepted image内のverifierはgit executableもsubprocessも使用しない。imageへgitを追加せず、accepted image
rebuildは0、networkはnoneである。

## predecessorとresult-free境界

v10 runtime / cache-freeze volume、v11 root-freeze volume、v12 runtime volumeはmount、read、copy、reuse
しない。旧`/opt/ngr-v8/runtime/frozen-source`はcreate、mount、readしない。model-cache copy、model import /
load / forward、registered query、development / holdout claim、worker process、observed result、shared SQLite
openはすべて0である。

`tools/run_cross_encoder_precision_v13_freeze_wslc.ps1 freeze`はprebuild implementation commitがpush済みで
Core CI / Optional MCPともgreen、かつcommit-freeze volumeとfuture runtime volumeがabsentの場合だけ開始する。
dedicated volume createとcommit identity verifier runは各exactly once、retryは0である。success / errorの
どちらでもvolumeはterminalかつnon-reusableで、同じv13を再試行しない。

このfreezeのpassはgit-free commit identity bindingとexact offline source-byte validationだけを支持する。
retrieval performance、retrieval parity、物理統合可能性、NGR default変更は支持しない。後続developmentは
terminal volumeを再利用せず、別Issueの新protocolで実行する。
