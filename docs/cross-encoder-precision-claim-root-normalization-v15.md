# Cross-encoder precision claim root normalization v15

この文書はIssue #181の分割requirements specである。中央`docs/requirements.md`はfrozen v8 protocolの
byte registryに含まれるため変更しない。

## 目的と前提

v14 preflightはmodel copyとsynthetic probeまで成功したが、development claimはactual protocol root
`/opt/ngr-v14/runtime/source`とgit-free identityのexpected root
`/opt/ngr-v14/runtime/frozen-source`の不一致によりclaim作成前にterminal errorとなった。同じv14とその
runtime volumeは再利用しない。

v15は性能観測ではない。v14 terminal evidenceと実装15 artifactsをbyte不変のpredecessorとして保存し、
configured exact claim source rootだけをconfigured exact frozen-source rootへ一対一に正規化したうえで、
v13 git-free verifierがexact protocol / corpus bytesを検証できることだけを固定するresult-free one-shotである。

## Root normalization境界

protocolは`github-ngr-cross-encoder-precision-v15`、専用volumeは
`github-cross-encoder-precision-v15-root-normalization-freeze`、rootは
`/opt/ngr-v15/root-normalization-freeze`である。configured claim sourceはroot配下の`source`、configured
frozen sourceは`frozen-source`である。

resolverはobserved rootがconfigured claim sourceとbyte-exactに一致し、absolute POSIX pathで、`..` escapeを
含まない場合だけconfigured frozen sourceを返す。relative root、old root、arbitrary root、sibling / child、
escapeはfail-closedする。v11 parameterized root binderとv13 git-free verifierを適用した後、同じ
resolver-aware verifierをwrapper、distinct base、evaluation wrapper / base、nested protocol evaluator /
baseのactual 6 surfacesへbindする。

verifierはaccepted image内でgit executableやsubprocessを使用せず、frozen protocol commit literal、manifest
bytes、23 artifact hashes、artifact registry identity、corpus registry / commit、24 documentsのexact bytes、
complete source identityを照合する。accepted imageは再buildせず、networkは`none`である。

## Result-free one-shot境界

v10 runtime / cache-freeze、v11 root-freeze、v12 runtime、v13 commit-freeze、v14 runtimeの各volumeはmount、
read、copy、reuseしない。旧`/opt/ngr-v8/runtime/frozen-source`はcreate、mount、readしない。model-cache
copy、model import / load / forward、registered query、development / holdout claim、worker、result、shared
SQLite openはすべて0である。

`tools/run_cross_encoder_precision_v15_freeze_wslc.ps1 freeze`はprebuild implementation commitがpush済みで
Core / Optional MCP CIともgreen、かつ専用freeze volumeとfuture runtime volume
`github-cross-encoder-precision-v15-runtime`がabsentの場合だけ開始する。volume createとroot normalization
verifier runは各exactly once、retryは0である。success / errorのどちらでもvolumeはterminalかつ
non-reusableで、同じv15を再試行しない。future runtimeは実行前後ともabsentを維持する。

このfreezeのpassはexact claim source root normalization、6-surface binding、git-free exact source-byte
validationだけを支持する。retrieval performance、retrieval parity、物理統合可能性、NGR default変更は
支持しない。後続performance observationはterminal v15 volumeを再利用せず、別protocolで実行する。
