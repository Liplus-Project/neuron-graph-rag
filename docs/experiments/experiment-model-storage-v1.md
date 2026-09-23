# 実験モデル共有保管 v1

## 変更前の観測

2026-09-23、外部 `workspace/experiments` の論理サイズは17.59 GiBだった。内訳は Docker 用 bundle 7.86 GiB、model-cache 5.50 GiB、venv 4.18 GiB、その他0.05 GiB。同じ `bge-reranker-v2-m3` の2,271,071,852 byteの重みが5パスで同一 SHA-256 `d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286` だった。

変更前に新規作成した `workspace/maintenance/247-model-plan.json` の SHA-256 は `fe50407cbe2d1e152f9873d33b7e5614797a12c822b1db2d6c81b64c1ca6c43d`。対象は9つの明示的 cache root 内の実ファイル68件、内容21種類、論理合計14,733,626,895 byte、同一内容の物理共有による最大回収見込み10,731,911,199 byte。`practical_two_stage_retrieval_v1.models.json` と `v2_m3_chunk_shortlist_ablation_v2.models.json` の登録値に対して112 file instanceのhashとsizeを照合した。古い cross-encoder v1 cacheのWindowsから読めないWSL形式MiniLM reparse point 6件はplanに記録し、変更対象から外した。

作業は #247 の独立ブランチで行い、#246 が使用したcross-encoder v2 venvとablation v2 bundleのパスを維持する。frozen evidence、manifest、fixture、one-shotスクリプトは変更しない。共有storeは同じD: volumeへ作り、元パスをhardlinkとして残す。`apply` 後の元実体は退避し、全パスのhashとfile identityを再確認してから `finalize` で退避を除く。

## 適用結果

モデル共有化はplan SHA-256 `fe50407cbe2d1e152f9873d33b7e5614797a12c822b1db2d6c81b64c1ca6c43d` に従い、47個の別実体を退避付きでhardlinkへ置換した。68対象ファイルの全hashと同一file identityを検証後、47退避ファイルを除いた。receipt SHA-256 は `9a2769bb491e75589b934c36f255d813f9c4b67e2c1c0fb071029e2d682ee02c`。D: 空き容量は493,406,867,456 byteから504,138,637,312 byteへ増え、差分は10,731,769,856 byte（9.995 GiB）だった。再度の全68件検証は通過し、退避ファイルは0件。E5とv2-m3は#246の既存パス・既存venvから登録queryを使わない合成入力で読み込めた。

venvは3環境に対して、変更前に `importlib.metadata` のexact installed-distribution inventoryと `*.dist-info` directory一覧を外部workspaceへ保存した。v1/v3はそれぞれ29/29件、#246が使用するv2は35/35件だった。`workspace/maintenance/247-venv-plan.json`（SHA-256 `e30763aa0a0a4846407614b1aab0194c59be78f90f53ebeab48173386ec6adbc`）は `Lib/site-packages` 内の1 MiB以上の `.lib` / `.dll` / `.pyd` だけ62ファイル・内容21種類を選び、最大回収見込み2,220,954,600 byteとした。Python code、metadata、設定ファイル、その他のvenvファイルは対象外にする。

venvの共有化はplanの通り41個の別実体を退避付きでhardlinkへ置換した。62対象ファイルの全hashと同一file identityを検証後、41退避ファイルを除いた。receipt SHA-256 は `3fa1304d160d379b672654aa0ea9a6ba9f6340b7da23f41a0d8266fb95ccb26c`。D: 空き容量は504,138,571,776 byteから506,358,996,992 byteへ増え、差分は2,220,425,216 byte（2.068 GiB）だった。再度の全62件検証は通過し、退避ファイルは0件。

3環境とも変更前後のinstalled-distribution inventory JSONがbyte単位で一致した（v1 `75a3e47933c20246a2c61c9fcc79649e884dbb6040757ce10d075791a7a06901`、v2 `a5442031aeac82abf4b4a487e81a9e2f0098a00fe8359bc26d0ef5643272a591`、v3 `054b11dff27bbca8b996406b4984ed0e8adde49e5f763b7169fc2f8976d904da`）。各venvで `torch 2.4.1+cpu`、`transformers 4.44.2`、`tokenizers 0.19.1` をimportできた。#246 の既存パス・venvによる合成入力probeも共有化後に再実行し、E5次元384、v2-m3 logit `-1.4248462915420532` で変更前と一致した。

2段階の実測回収合計は12,952,195,072 byte（12.063 GiB）。`workspace/experiments` の各パスを足した論理サイズはhardlink化後もほぼ17.59 GiBと表示されるが、同一実体の重複計上を除いた使用量とD:の空き容量は改善した。venv全体と凍結証拠は保持した。共有ファイルはread-onlyなので、将来のパッケージ更新は新しいvenvで行う。既存の凍結スクリプトはbundleへモデルを再コピーするため、今後の新規実験では共有storeのread-only mountを別途設計・検証する必要がある。Docker/WSLC volume内の容量は今回の対象外。
