# Cross-encoder precision benchmark freeze v4

## 目的と result-free 境界

v3 one-shot development は baseline primary/replay と bge-base primary の 3/6 worker packet を生成した後、Windows native process が `0xC0000005` で終了し、result、candidate selection、hard gate 判定へ到達しなかった。v4 は cross-encoder rank-only の検索性能を未評価のまま保ち、successor observation の実行 substrate と再現 lock だけを WSL2/Linux へ切り替える。

freeze input は main `043ab31a0960687fa2ac598a1a3768f3fd073f36`、v3 freeze `b762645d2521a3e23ac201b662ea1cbf25e2a260`、source corpus commit `c32b3049fd3daaa2190faf5e3e85955a195ee88c`、canonical judgment revision 6 である。v3 outcome は `protocol_error_not_assessed`、development worker `3/6`、holdout unopened のまま扱う。

本 freeze は登録 query、model inference、observed result を生成しない。v1/v2/v3 evidence は manifest の SHA-256 registry により byte identity だけを検証し、JSON や実験結果として解釈しない。model cache、weight、venv、fresh DB、shared Windows `C:\Users\smile\.ngrdb\knowledge.db` を開かず、Windows で v3/v4 observation を再試行しない。

## v3 から維持する semantic contract

v4 evaluator は frozen v3 evaluator を独立 module namespace へ load し、v4 protocol ID、fixture stem、artifact set だけを bind する。24 corpus identity、development/holdout 各8件の bilingual query/gold、2 exact model revision、passage 480/80、batch 8、NGR top24、model prefilter exact top20、4 rank-only candidate 順、CE/RRF 式、top5、tie-break、selection rule、11 hard gate、one-time lifecycle は v3 と同一である。

core fixture は protocol ID 以外 v3 と等しい。Linux lock は v3 lock を constraint とし、共通 package version を変更せず、Linux wheel/source artifact hash と Linux-only transitive dependency を隔離する。v3/v4 の許可差分は protocol identity/path、platform contract、Python/Linux dependency lock だけである。

## WSL2/Linux reproduction lock

- substrate: WSL2 Ubuntu、Linux x86_64 GNU
- interpreter: `cpython-3.11.15-linux-x86_64-gnu`
- source: python-build-standalone release `20260807`
- interpreter artifact SHA-256: `69dfac9d0f15a0b9281a38486f212cbf76421609228c184dc0d34a0533d57ba6`
- resolver: `uv 0.12.3 (507230998)`
- platform tag: `x86_64-unknown-linux-gnu`
- dependency input: v3 と byte-identical
- PyTorch artifact backend: `cpu`（`torch==2.4.1+cpu`）
- Linux lock SHA-256: `db3310ea9f1b27b63d0c4e4085223502e3353787835c6233355ae3c23bff6df4`

successor observation は WSL ext4 上の protocol 専用 absolute run root に project environment、fresh SQLite DB、worker output、runtime/archive/transport を exclusive-create する。`/mnt` 以下、とくに `/mnt/c` では worker process と SQLite DB を実行しない。

exact model bytes の既存 Windows cache からの read-only reuse は successor preflight に限る。required file の size/hash をすべて再検証してから専用 ext4 cache へ copy し、以後 offline/local-files-only とする。freeze は cache を open しない。実行は CPU、float32、eval、inference mode、batch 8、fresh process/fresh DB である。

## One-time lifecycle

v4 専用 claim/result/error/archive/transport path は互いに distinct で、exclusive-create する。freeze test は synthetic unobserved、development archived pass/fail/error、holdout archived、negative/positive/mixed logit、5件未満、0件、tie、CE/RRF derived field と11 gate の tamper rejectionを同じ v4 phase verifierへ通す。

v4 freeze merge commit だけを successor observation の protocol input にできる。successor は v1/v2/v3 packet を移送、変換、再利用せず、WSL ext4 の fresh process/DB で新しい v4 development packet を exactly once 生成する。claim 後の例外は error archive し、同 version を再試行しない。全 hard gate pass 時だけ holdout を一度開く。

## Result-free verification

```text
python -m neuron_graph_rag.cross_encoder_precision_v4_evaluation audit
python -m neuron_graph_rag.cross_encoder_precision_v4_evaluation probe
python -m unittest tests.test_cross_encoder_precision_v4
```

audit/probe と synthetic tests の freeze count は query/inference/result=`0/0/0` である。
