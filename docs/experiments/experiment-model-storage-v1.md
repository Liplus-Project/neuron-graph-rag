# 実験モデル共有保管 v1

## 変更前の観測

2026-09-23、外部 `workspace/experiments` の論理サイズは17.59 GiBだった。内訳は Docker 用 bundle 7.86 GiB、model-cache 5.50 GiB、venv 4.18 GiB、その他0.05 GiB。同じ `bge-reranker-v2-m3` の2,271,071,852 byteの重みが5パスで同一 SHA-256 `d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286` だった。

変更前に新規作成した `workspace/maintenance/247-model-plan.json` の SHA-256 は `fe50407cbe2d1e152f9873d33b7e5614797a12c822b1db2d6c81b64c1ca6c43d`。対象は9つの明示的 cache root 内の実ファイル68件、内容21種類、論理合計14,733,626,895 byte、同一内容の物理共有による最大回収見込み10,731,911,199 byte。`practical_two_stage_retrieval_v1.models.json` と `v2_m3_chunk_shortlist_ablation_v2.models.json` の登録値に対して112 file instanceのhashとsizeを照合した。古い cross-encoder v1 cacheのWindowsから読めないWSL形式MiniLM reparse point 6件はplanに記録し、変更対象から外した。

作業は #247 の独立ブランチで行い、#246 が使用したcross-encoder v2 venvとablation v2 bundleのパスを維持する。frozen evidence、manifest、fixture、one-shotスクリプトは変更しない。共有storeは同じD: volumeへ作り、元パスをhardlinkとして残す。`apply` 後の元実体は退避し、全パスのhashとfile identityを再確認してから `finalize` で退避を除く。

## 適用結果

結果前の時点では未適用。外部receiptと適用後の検査結果をここに追記する。
