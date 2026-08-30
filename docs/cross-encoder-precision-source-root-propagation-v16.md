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

## Terminal execution result

prebuild commit `c0fdeffd75a6f21d4abd2784cc937e5ee6420346`のPR CI run `33319201845`がCore /
Optional MCPともgreenで、専用freeze volumeとfuture runtime volumeのabsenceを確認した後、v16 one-shotを
exactly once実行した。statusはpassで、同じv16は再実行しない。

actual loaderはclaim source rootを`PosixPath`として運び、共通resolverはconfigured source
`/opt/ngr-v16/source-root-propagation-freeze/source`とのexact一致を確認してconfigured frozen source
`/opt/ngr-v16/source-root-propagation-freeze/frozen-source`へ正規化した。同一verifierはdistinct 6 surfacesへ
bindされ、frozen protocol 23 artifactsとcorpus 24 documentsのexact bytesをgit / subprocessなしで検証した。

専用volume createは1、verifier runは1、retryは0、future runtime volumeは前後ともabsent、v15 predecessor
14 artifactsはbyte不変だった。accepted image rebuild、model-cache copy、model import / load / forward、
registered query、development / holdout claim、worker、result、shared SQLite open、container git / subprocessは
すべて0で、v10-v15 volumeはmountしていない。

主要SHA-256は次のとおりである。

- `source-root-propagation-verification.json`: `8dc0fc41cde99db000208e05fb203dd1c8cdf57595492c9d8280bd702897825e`
- `source-root-propagation.pass.json`: `50b78e8f2702298fb59edb7a1a446cab298571dc6107f9a690515a9e97eded49`
- `count-audit.json`: `52e849aba9e368ea8159aef6377939d67455fe6f4338c0d5c19c43918b620656`
- `evidence-manifest.json`: `7e7652a9846ef0dfaf68cfacd4270dc04fbe8ef20dea835214d7d504bfccfd87`

このpassはsource-root propagationとexact byte verificationのresult-free範囲だけを支持する。rank performance、
retrieval parity、物理統合可能性は未評価で、NGR defaultは変更しない。後続performance observationはterminal
v16 volumeを再利用せず、別protocolで実行する。
