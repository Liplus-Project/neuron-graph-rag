# Engine-backed feedback trajectory experiment

## 目的

repository-native controlled corpus v3 に対し、実際の `NeuronGraphRAG` instance、`search_channels` の relation channel、保存された relation trace、`record_success` を使って、credited feedback count 0、1、3、10 の retrieval trajectory を測定する。結果の主張は、この固定 corpus 上の NGR engine adaptation に限定する。

## Result-free freeze

- Source corpus commit: `94c8bc250b7352e3009eeee1b353c3aec677bfb7`
- Fixture: `tests/fixtures/engine_feedback_trajectory_v3.fixture.json`
- Gold: `tests/fixtures/engine_feedback_trajectory_v3.gold.json`
- Schedule: `tests/fixtures/engine_feedback_trajectory_v3.schedule.json`
- Gate: `tests/fixtures/engine_feedback_trajectory_v3.gate.json`
- Audit: `tests/fixtures/engine_feedback_trajectory_v3.audit.json`
- Manifest: `tests/fixtures/engine_feedback_trajectory_v3.manifest.json`

manifest artifact は manifest path の初回追加 commit、source corpus は上記 source corpus commit の exact blob bytes を読む。各 commit の存在と current `HEAD` の ancestor 関係を検証し、current working tree の同名 fixture、source document、evaluator は historical evidence として扱わない。既存の raw-first LF / CRLF whole-file alternate だけを維持する。

development は `signal-stability` と `boundary-recovery`、holdout は `evidence-continuity` を使う。node ID、document path、source URL、explicit-link edge は split 間で重複させない。edge は overview 文書に記載された同一 directory 内の相対 Markdown link だけから固定する。

result-free commit を push するまで development と holdout の runner を実行しない。exclusive output が存在しないことを audit に固定し、runner は既存 output の上書きを拒否する。

## Registered execution

各 stage の control と treatment は、それぞれ一つの `NeuronGraphRAG` instance に source document と explicit-link edge を投入する。0、1、3、10 の各 checkpoint で engine が返した relation rank、relation trace ID、raw path、endpoint / type へ射影した path、role 別 MRR、edge snapshot を保存する。

各 cluster の feedback event は固定 query で `search_channels` を呼び、relation trace に固定 used node と credited path が含まれることを確認する。control は同じ query と used node を成功記録へ残すが edge update を空にする。treatment は relation trace ID と used node を `record_success` へ渡し、返された reinforced edge を保存する。独自順位式、手作業の weight 加点、engine hit 外からの rank 生成は行わない。

development は一回だけ実行する。全 gate が通過した場合だけ holdout を一回だけ実行する。失敗 output も保存し、その時点で停止する。観測後は evaluator、fixture、gold、schedule、manifest、gate、この文書、requirements を変更しない。

## Fixed gates

- source hash、source commit、split / cluster identity、manifest 内 artifact hash が一致する。
- score と feedback の全 trace が relation channel で、期待 path が endpoint / type identity と一致する。
- treatment の headroom MRR は 0 から 10 で厳密改善し、0、1、3、10 の途中 checkpoint で退行しない。
- treatment の control case と ceiling case は checkpoint 間で退行せず、control arm 全体も退行しない。
- control は feedback を記録しても edge mutation がゼロである。
- treatment は固定 credited edge だけを変更し、各 cluster の feedback count が checkpoint と一致する。
- registered run count、exclusive output、development から holdout への停止規則を満たす。

## 実行

result-free commit の push 後に、次の development を一回だけ実行する。

```powershell
python tools/run_engine_feedback_trajectory.py development `
  --manifest tests/fixtures/engine_feedback_trajectory_v3.manifest.json `
  --output tests/fixtures/engine_feedback_trajectory_v3.development.observed.json
```

development の全 gate が通過した場合だけ、次の holdout を一回だけ実行する。

```powershell
python tools/run_engine_feedback_trajectory.py holdout `
  --manifest tests/fixtures/engine_feedback_trajectory_v3.manifest.json `
  --development-result tests/fixtures/engine_feedback_trajectory_v3.development.observed.json `
  --output tests/fixtures/engine_feedback_trajectory_v3.holdout.observed.json
```

## 解釈境界

通過結果が支持するのは、固定した repository-native controlled corpus v3 と現行 NGR engine の relation-trace credit mechanism における longitudinal trajectory だけである。external generalization、Agent E2E 効率、production adoption、default 変更は主張しない。
