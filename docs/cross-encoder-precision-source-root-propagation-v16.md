# Cross-encoder precision source root propagation v16

この文書はIssue #183の分割requirements specである。中央`docs/requirements.md`と凍結済みv15 artifactsは
変更しない。

## 目的と前提

v15 result-free one-shotはactual 6-surface graphへresolver-aware verifierをbindしたが、protocol loaderが
`PosixPath`として運んだclaim source rootをresolverがstring以外として拒否し、verification完了前にterminal
errorとなった。同じv15 protocolとvolumeは再利用しない。

v16はperformance観測ではない。v15 terminal evidenceと14 implementation / evidence artifactsをbyte不変の
predecessorとして保存し、pathまたはstringで運ばれるconfigured exact claim source rootを共通部品でexact
frozen-source rootへ正規化し、同じverifierをactual 6 surfacesへ伝播するresult-free one-shotである。

## Thin composition境界

再利用可能な`source_root_propagation` moduleがroot型の受理、exact root解決、protocol root正規化、distinct
6-surface discovery、verifier binding、result-free contract / evidence auditを所有する。v16固有moduleはprotocol /
volume / path / identity設定、v13 git-free verifierとのcomposition、command、v15 lifecycleへのscoped adapterだけを所有する。
凍結済みv15 moduleは編集しない。

protocolは`github-ngr-cross-encoder-precision-v16`、専用volumeは
`github-cross-encoder-precision-v16-source-root-propagation-freeze`、rootは
`/opt/ngr-v16/source-root-propagation-freeze`である。sourceはroot配下の`source`、frozen sourceは
`frozen-source`、future runtime volumeは`github-cross-encoder-precision-v16-runtime`である。

resolverはpathまたはstringのobserved rootがconfigured sourceとbyte-exactに一致し、absolute POSIX pathで、
`..`を含まない場合だけconfigured frozen sourceを返す。relative、old、arbitrary、sibling、child、escape、
非path型はfail-closedする。verifierはfrozen-source bytesからexact commit / manifest / 23 artifacts / corpus /
24 documents / source identityをgit / subprocessなしで照合する。

## Result-free one-shot境界

v10-v15のruntime / freeze volumeはmount、read、copy、reuseしない。旧
`/opt/ngr-v8/runtime/frozen-source`はcreate、mount、readしない。accepted image rebuild、network、model-cache
copy、model import / load / forward、registered query、development / holdout claim、worker、result、shared
SQLite openはすべて0である。

専用volume createとsource-root propagation verifier runは各exactly once、retryは0とする。success / errorの
どちらでも同一v16を再試行せず、future runtime volumeは前後absentを維持する。one-shotはprebuild commitの
PR CIがCore / Optional MCPともgreenになった後だけ実行する。

## 検証段階

開発中はv16専用testと関連targetedだけを実行する。evidence追加時は専用test、関連targeted、auditを実行する。
full suiteはPR前に1回だけ実行し、以後は共通 / core code変更時だけ再実行する。最終full保証はGitHub CIに置く。

v16のpassはexact source-root propagation、6-surface binding、git-free exact source-byte verificationだけを
支持する。rank performance、retrieval parity、物理統合可能性、NGR default変更は支持しない。
