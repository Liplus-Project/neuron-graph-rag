# 実験モデルの共有保管

`tools/deduplicate_experiment_models.py` は、外部実験 workspace のモデル snapshot を内容アドレス付き store へ結び、既存の snapshot path を保ったまま同一 D: volume 上の実体だけを共有する。`--mode venv-binaries` は凍結済みvenvの `Lib/site-packages` 内にある1 MiB以上の `.lib` / `.dll` / `.pyd` だけを別の共有storeへ結ぶ。凍結済み evidence、fixture、実験コード、観測結果、Python metadataと設定ファイルは変更しない。Docker / WSLC volume はこのツールの対象外である。

## 手順

1. 実験プロセスが動いていないこと、対象 source root が同一 volume にあることを確認する。`plan` に workspace root、まだ存在しない store path、対象の `model-cache` / `ce-cache` / `e5-cache` を `--source` で明示する。plan は新規 path にだけ作成される。
2. plan の `files`、SHA-256、`skipped_reparse_points`、参照中の実験を確認する。Windows から読めない WSL 形式の MiniLM reparse point は記録するが変更しない。登録済み model fixture の hash と、使うモデルの実ファイルが一致することも確認する。
3. `apply --plan <plan.json>` は全対象を先に再ハッシュし、共有 store を hardlink で作り、重複した snapshot file を同じ内容の hardlink に置き換える。元の実体は各ファイルの `.ngr-dedup-backup` として残る。この段階ではディスク容量はまだ回収しない。
4. `verify --plan <plan.json>` で全既存 path と store の hash・file identity・退避ファイルを検査する。問題があれば `rollback --plan <plan.json>` で退避ファイルへ戻す。
5. 検査を通過したら `finalize --plan <plan.json> --receipt <receipt.json>` を一度だけ実行する。退避ファイルを除き、共有実体を read-only にし、新規 receipt に plan hash と D: 空き容量を記録する。再度 `verify` を実行する。

例（各 source は既存の snapshot cache root を指定する）:

```powershell
python tools/deduplicate_experiment_models.py plan --workspace D:\path\to\Codex --store D:\path\to\Codex\workspace\model-store --source D:\path\to\Codex\.semantic230-model --source D:\path\to\Codex\workspace\experiments\github_cross_encoder_precision_v1\model-cache --output D:\path\to\Codex\workspace\maintenance\model-plan.json
python tools/deduplicate_experiment_models.py apply --plan D:\path\to\Codex\workspace\maintenance\model-plan.json
python tools/deduplicate_experiment_models.py verify --plan D:\path\to\Codex\workspace\maintenance\model-plan.json
python tools/deduplicate_experiment_models.py finalize --plan D:\path\to\Codex\workspace\maintenance\model-plan.json --receipt D:\path\to\Codex\workspace\maintenance\model-receipt.json
```

既存の one-shot スクリプトは frozen source として残し、今後の実験では同じ content hash の共有 snapshot を検証してから読み取り専用で渡す。Docker / WSLC への read-only mount が利用できるかは、将来の実験 protocol で個別に検証する。新しい build context へ再コピーする既存スクリプトを再実行すれば、その新しい context には再び物理コピーが生じる。

ハッシュはモデル内容の一致を示す。完全な再現には、source commit、query / corpus / gate、依存 package の exact set、CPU / runtime 条件、offline 実行記録も残す必要がある。venv全体は削除しない。大きな共有binaryはread-onlyになるため、後から依存パッケージを変更するときは新しいvenvを作る。

venvの共有化では、先に `tools/inventory_experiment_runtime.py --output <新規path>` を各venvのPythonから実行し、installed distributionと `*.dist-info` の両一覧を保存する。`plan --mode venv-binaries` に3つのvenv rootと新しい `runtime-store` を渡し、上記と同じ apply → verify → finalize → verify を行う。終了後に各venvから再度inventoryを作り、変更前後のdistributionとmetadata hashを比較し、importと合成入力を試す。
