# Cross-encoder precision observation v14

## 実行境界

Issue #179 の one-shot observation は、successful v13 git-free protocol identity freeze を含む
squash merge commit `56d32bac8144b96b03a6813d8732600a3491f8c9` と accepted v8 image
`ngr-cross-encoder-precision-v8:freeze` / image ID
`sha256:136ad9466799109bf32b4b96b611c9db9a099bcc47cf78243f26c7227bc16742`
だけを実行基盤とする。accepted image は再buildせず、network は `none` とし、containerへgitを追加しない。

v10 runtime / cache-freeze、v11 root-freeze、v12 runtime、v13 commit-freezeの各volumeはmount、read、
copy、再利用しない。v13 source、tests、fixtures、tool、docs、evidenceの15 artifactsをgit上のSHA-256で
固定し、predecessor evidenceのsemantic content、raw packet、result、error、model cache、weightsは観測入力にしない。

唯一のmutable rootは開始時absentを確認してexclusive-createする
`github-cross-encoder-precision-v14-runtime` である。source、frozen protocol source、model cache、fresh
SQLite、worker output、claim / result / error / archive / transportは`/opt/ngr-v14/runtime`配下のdistinct
absolute POSIX pathへ置く。旧`/opt/ngr-v8/runtime`はcreate、mount、readしない。

## Git-free parameterized harness

dependency report、synthetic model probe、claim、worker、finalize、fail-stageの各container processは、
v13 `bind_git_free_commit_verifier` をactual evaluator object graphへ適用する。v8 wrapperとdistinct `_BASE`、
evaluation wrapper / base、nested protocol evaluator / baseの6 surfaceをv14 rootへbindし、git / subprocessを
呼ばずにfrozen manifest 23 artifactsとcorpus 24 documentsのexact bytesを検証する。bindingはv14 commandの
scoped lifecycleに限定し、NGR既定retrieval pathとpredecessor moduleのhost-side defaultは変更しない。

## Preflight

`tools/run_cross_encoder_precision_v14_observation_wslc.ps1 preflight` は、実装commitをpushしたPRのCore /
Optional MCP CIがgreenになった後だけ一度実行する。専用runtime volume create、source initialization、
frozen-source import、model-cache copy、dependency report、synthetic probeは各最大一度である。source
initializationはmodel-cacheを作成せず、frozen v10 verifierだけがread-only Windows cacheから2 revisions /
12 required files / 3,427,616,927 bytesをexact post-copy hash付きで専用volumeへcopyする。

preflightはv8 evaluator audit / probe、v10 / v11 / v13 audit、v14 targeted tests、full suite、変更対象Ruff、
shared Windows DBのFILE_SHARE-aware SHA-256前後一致を固定する。共有DBはSQLite接続しない。失敗はterminal
evidenceとして固定し、同じv14 preflightを再試行しない。successful preflight evidenceをcommit / pushし、
そのremote check-runがすべてgreenになった後だけdevelopment claimを作れる。

## One-shot lifecycle

`tools/run_cross_encoder_precision_v14_observation_wslc.ps1 run` はpushed remote HEADとCI greenを再検証し、
development claimをexactly once作る。baseline primary / replay、base primary / replay、v2-m3 primary /
replayを固定順の6 fresh container process / 6 fresh SQLite DBで各一度だけ実行する。

developmentで全frozen hard gateがpassしselected candidateがある場合だけholdout claimをexactly once開く。
selectedなし、fail、errorではholdoutを開かない。success / fail / errorのいずれもretry countは0であり、
失敗後に同じv14 protocolを再実行しない。shared Windows DBは開始前後でbyte不変とし、fresh observation
SQLiteだけをmutableにする。

## 主張境界

観測前はperformanceを`not assessed`とする。terminal evidence固定後も、主張は実測したrank-only gate、
selected candidate、holdout開閉、count、hashの範囲に限定する。retrieval parity、physical integration、
production performance、NGR default変更をこのprotocolだけから主張しない。

## Preflight実測

prebuild implementation commit `8b10b7b9e5d71e65d1929a33bfc68c4f8fc2641f` のCore / Optional MCP
CI run `33294338833`がgreenになった後、v14 preflightをexactly once実行した。runtime volumeは開始時absentで
exclusive-createされ、accepted image rebuild、旧volumeのmount / read / reuse、旧v8 rootのcreate / mount /
read、registered query、development / holdout claim、result、retryはいずれも0だった。model-cacheはsource
initializationでは作成されず、read-only source cacheから2 revisions / 12 files / 3,427,616,927 bytesをexact
hash付きでcopyした。synthetic probeは2 forward inferenceを完了した。

shared DB SHA-256はpreflight前後とも
`84a3fc590eee990579e3ef8130294129934fe93e25f18ac249eece19813c261e`で不変だった。preflight evidence
manifest SHA-256は`beac5d13447d2d326c549ca85e9111d57f54d2afddb3ba0dd67b25b46155a182`である。
development / holdout performanceは引き続き`not assessed`であり、preflight evidence commitのCI green前には
developmentを開かない。

## Development実測

preflight evidence commit `2847cfd6e49879cf313a8c704efdf1fd464b8d17` のCore / Optional MCP CI
run `33294680229`がgreenになった後、development one-shotをexactly once開始した。preflight evidence syncと
shared DB hash再検証は成功したが、最初のdevelopment claim containerでterminal errorになった。

v13 git-free verifierはfrozen source identity root
`/opt/ngr-v14/runtime/frozen-source`を要求した一方、actual claim pathは
`/opt/ngr-v14/runtime/source`からprotocolをloadしたため、
`ValueError: protocol root does not match frozen source identity`でfail-closedした。claim fileの
exclusive-create前に停止したためdevelopment / holdout claim=`0/0`、worker process=`0`、observed result=`0`、
retry=`0`であり、holdoutは開いていない。shared DBの観測後SHA-256も
`84a3fc590eee990579e3ef8130294129934fe93e25f18ac249eece19813c261e`で不変だった。

raw execution error SHA-256は`a5de91beb8677ed9db1d0d3acc63cb610dafd3538caee22200af574100dcd50f`、
terminal evidence manifest SHA-256は
`6d5ca7f72362c7dba3fbacf6ff7a6ba8a922e26de87b59063789ad052424d028`である。同じv14 preflight、claim、
development、holdoutは再実行しない。performanceは`not assessed`であり、retrieval parity、physical
integration、production performance、NGR default変更を支持しない。successorはactual claim pathがloadする
sourceとgit-free identity verifierのexpected rootをresult-free protocolで一致させてから、別protocolとして
freeze・観測する必要がある。
