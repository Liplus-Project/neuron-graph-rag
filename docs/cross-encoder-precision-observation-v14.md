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
