# Feedback rank elasticity

## 目的と境界

この文書は、success feedback による edge reinforcement が現行の fusion と max normalization を通って順位へ伝わるまでの感度を測る、rank elasticity 評価の仕様を定める。評価は transport-neutral な NGR core の実挙動を使い、learning rate、fusion、normalization、既定値を変更しない。結果は固定 corpus 内の診断であり、production 品質や外部 corpus への一般化を主張しない。

source SQLite database は read-only で扱う。runner は baseline と各 feedback checkpoint を source から作成した別々の SQLite backup 上で再生し、実行前後に次を照合する。

- database file の SHA-256
- node、edge、retrieval、success feedback、source-use、delayed outcome の各 row 数
- 全 edge の weight と reinforced count

source に差分が生じた場合、runner は結果を返さず失敗する。出力先が既に存在する場合も上書きせず失敗する。

## 固定 schedule

schedule は `ngr.rank-elasticity/v1` JSON とし、全 checkpoint で同じ source、config、query、used node、timestamp、limit を使う。feedback 回数は 0 から始まる重複のない昇順整数で指定し、各 checkpoint は fresh clone 上でその回数だけ独立に累積再生する。前 checkpoint の clone を次へ流用しないため、実行順による状態依存を持ち込まない。

各 scenario は一つの `relation_target` と、次の control をそれぞれ明示する。

- `direct_control`: 直接一致する node の順位安定性
- `lexical_control`: lexical evidence の順位安定性
- `directional_negative_control`: edge 方向を逆にした非対象 node の順位安定性

repository 付属 schedule は `[0, 1, 3, 5, 10]` を固定し、複数回でも順位が変わらない max normalization の ceiling case と、同一 credited path が1回目の独立 feedback で順位境界を跨ぐ threshold case を分離する。

## 出力指標

各 case と checkpoint は、少なくとも次を JSON に保存する。

- feedback count と target rank
- entry score、raw graph activation、normalized graph activation、final score
- baseline で直上にいた候補との final-score margin
- weight または reinforced count が変わった edge
- baseline からの全 top-k rank delta（top-k への entry / exit は仮想 rank `k + 1` との差として表す）
- target 以外で順位が変わった node 数、絶対 rank delta 合計、node ID

scenario の診断は次を区別する。

- `rank_flip_threshold`: baseline より良い rank を初めて得た checkpoint が存在する
- `edge_changed_but_rank_unchanged`: edge は変わったが target rank は schedule 全体で変わらない
- `rank_stable_through_schedule`: 上記の edge-change 条件を満たさず、target rank が schedule 全体で変わらない

`fusion_side_ceiling` は、edge が変化した一方で normalized graph score、final score、rank がすべて不変な場合にのみ `true` となる。これは raw graph signal の増加が max normalization 後に消える境界を示すもので、feedback 自体が無効だったことや、別の fusion 構造が優れることを意味しない。

## 実行方法

既存 source database に対して実行する。

```powershell
uv run python tools/run_rank_elasticity.py `
  --database path/to/source.sqlite `
  --schedule tests/fixtures/rank_elasticity_v1.schedule.json `
  --output path/to/rank-elasticity.result.json
```

決定論的な synthetic regression は `tests/fixtures/rank_elasticity_v1.fixture.json` から source database を構築する。fixture と schedule は ceiling、threshold、direct、lexical、directional-negative の境界検査専用であり、実 corpus の観測結果ではない。
