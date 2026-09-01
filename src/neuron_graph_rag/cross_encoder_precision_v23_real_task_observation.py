from __future__ import annotations

from pathlib import Path, PurePosixPath

from . import cross_encoder_precision_v21_intent_aware_observation as v21
from . import intent_aware_observation_engine, intent_aware_observation_runtime

PROTOCOL_ID = "github-ngr-cross-encoder-precision-v23-real-tasks"
FREEZE_COMMIT = "79b456d620f1b37746669ea1fe1e57c385f5e4ed"
ROOT = Path(__file__).resolve().parents[2]
MODULE = "neuron_graph_rag.cross_encoder_precision_v23_real_task_observation"
MANIFEST = Path(
    "tests/fixtures/github_cross_encoder_precision_v23_real_observation.manifest.json"
)
SOURCE_IDENTITY = Path(
    "tests/fixtures/github_cross_encoder_precision_v23_real.source-identity.json"
)
OBSERVATION_AUDIT = Path(
    "tests/fixtures/github_cross_encoder_precision_v23_real.observation-audit.json"
)
CORPUS = Path("tests/fixtures/github_cross_encoder_precision_v23_real.corpus.json")
QUERIES = Path("tests/fixtures/github_cross_encoder_precision_v23_real.queries.json")
GOLD = Path("tests/fixtures/github_cross_encoder_precision_v23_real.gold.json")
MODEL_REGISTRY = Path("tests/fixtures/github_cross_encoder_precision_v8.models.json")
EVIDENCE = Path(
    "tests/evidence/github_cross_encoder_precision_v23_real_task_observation"
)
VOLUME = "github-cross-encoder-precision-v23-real-tasks-runtime"
CONTAINER_ROOT = PurePosixPath("/opt/ngr-v23-real/runtime")
CONTAINER_SOURCE = CONTAINER_ROOT / "source"
CONTAINER_CACHE = CONTAINER_ROOT / "model-cache"
CONTAINER_PROTOCOL_SOURCE = CONTAINER_ROOT / "frozen-source"
CONTAINER_DATABASES = CONTAINER_ROOT / "databases"
CONTAINER_RUNS = CONTAINER_ROOT / "runs"
CONTAINER_ARCHIVE = CONTAINER_ROOT / "archive"
CONTAINER_TRANSPORT = CONTAINER_ROOT / "transport"

FORBIDDEN_VOLUMES = {
    **v21.FORBIDDEN_VOLUMES,
    "v21_runtime_volume": v21.VOLUME,
}
PREDECESSOR_ANCHOR_SHA256 = {
    "src/neuron_graph_rag/cross_encoder_precision_v21_intent_aware_observation.py": (
        "04b5c8fe589809203ece1aec32f9b6041990e9c738bd57bbb93cfc9f0e1cc512"
    ),
    "src/neuron_graph_rag/cross_encoder_precision_v22_intent_aware_observation.py": (
        "63f7369fd3ca28d435261c438d28d0682397aa9a870b142116ab0af6a8116bc7"
    ),
    "tests/evidence/github_cross_encoder_precision_v21_observation/"
    "terminal-evidence-manifest.json": (
        "0cbf5235dc312c67db5fa2f9d8f4116b8ff5d09141b085e7abf15fe6cb43693c"
    ),
    "tests/fixtures/github_cross_encoder_precision_v21_observation.manifest.json": (
        "ea7d83097c00ed1a262f4c9209b409964bbc73795f1822c88b56c20e078c317b"
    ),
    "tests/fixtures/github_cross_encoder_precision_v22.engine-contract.json": (
        "845dfa3d5c53eec3302a9fff9e146af96f651e84598dfdd865c7e4dfa03108f3"
    ),
}
PREDECESSOR_ARTIFACT_COUNT = 44

ENGINE_SPEC = intent_aware_observation_engine.IntentAwareObservationSpec(
    protocol_id=PROTOCOL_ID,
    fixture_paths=intent_aware_observation_engine.ObservationFixturePaths(
        corpus=CORPUS,
        queries=QUERIES,
        gold=GOLD,
        model_registry=MODEL_REGISTRY,
    ),
    stage_identities=(
        (
            "development",
            "github-ngr-cross-encoder-precision-v23-real-development-1f7c9b42",
        ),
        (
            "holdout",
            "github-ngr-cross-encoder-precision-v23-real-holdout-83d4ea61",
        ),
    ),
    models=(
        intent_aware_observation_engine.ModelIdentity(
            kind="base",
            candidate_id="base-intent-aware",
            model_id="BAAI/bge-reranker-base",
            revision="2cfc18c9415c912f9d8155881c133215df768a70",
        ),
        intent_aware_observation_engine.ModelIdentity(
            kind="v2-m3",
            candidate_id="v2-m3-intent-aware",
            model_id="BAAI/bge-reranker-v2-m3",
            revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        ),
    ),
    baseline_evidence_id="original-full-query-ngr-default",
    baseline_kind="default",
    baseline_query_mode="full",
    ablation_arms=(
        intent_aware_observation_engine.RetrievalArmIdentity(
            kind="baseline",
            evidence_id="positive-clause-ngr-prefilter-ablation",
            query_mode="positive",
        ),
    ),
    selection_policy="lowest-development-primary-latency",
)
ENGINE = intent_aware_observation_engine.IntentAwareObservationEngine(ENGINE_SPEC)

FIXTURE_CONTRACT = intent_aware_observation_runtime.RealCorpusFixtureContract(
    engine=ENGINE,
    corpus_path=CORPUS,
    predecessor_query_paths=(
        Path("tests/fixtures/github_cross_encoder_precision_v8.queries.json"),
        Path("tests/fixtures/github_cross_encoder_precision_v21.queries.json"),
    ),
    expected_provenance=(
        ("repository_url", "https://github.com/Liplus-Project/neuron-graph-rag"),
        ("commit", FREEZE_COMMIT),
        ("method", "git show <commit>:<path>"),
        ("read_only", True),
        (
            "generated_by",
            "tools/acquire_cross_encoder_precision_v23_real_corpus.py",
        ),
    ),
    document_count=12,
    cohort_cardinality=(
        ("direct_lexical", 2),
        ("semantic_paraphrase", 2),
        ("relation_linked", 2),
        ("negative_control", 2),
    ),
    maximum_predecessor_similarity=0.72,
    separation_result_prefix="v19_v21",
)


def _verification_commands(root: Path) -> tuple[list[str], ...]:
    python = root / ".venv" / "Scripts" / "python.exe"
    return (
        [
            "uvx",
            "--offline",
            "ruff",
            "check",
            "src/neuron_graph_rag/intent_aware_observation_engine.py",
            "src/neuron_graph_rag/intent_aware_observation_runtime.py",
            "src/neuron_graph_rag/cross_encoder_precision_v23_real_task_observation.py",
            "tests/test_intent_aware_observation_engine.py",
            "tests/test_cross_encoder_precision_v23_real_task_observation.py",
            "tools/acquire_cross_encoder_precision_v23_real_corpus.py",
        ],
        [
            str(python),
            "-m",
            "unittest",
            "tests.test_intent_aware_observation_engine",
            "tests.test_cross_encoder_precision_v22_intent_aware_observation",
            "tests.test_cross_encoder_precision_v23_real_task_observation",
        ],
        [
            str(python),
            "tools/acquire_cross_encoder_precision_v23_real_corpus.py",
            "--root",
            str(root),
            "--output",
            str(CORPUS),
            "--verify",
        ],
        [str(python), "-m", v21.MODULE, "audit"],
        [
            str(python),
            "-m",
            "neuron_graph_rag.cross_encoder_precision_v22_intent_aware_observation",
            "validate",
        ],
        [str(python), "-m", MODULE, "audit"],
    )


RUNTIME = intent_aware_observation_runtime.IntentAwareObservationRuntime(
    intent_aware_observation_runtime.IntentAwareRuntimeConfig(
        engine=ENGINE,
        fixture_contract=FIXTURE_CONTRACT,
        protocol_id=PROTOCOL_ID,
        freeze_commit=FREEZE_COMMIT,
        root=ROOT,
        module_name=MODULE,
        manifest_path=MANIFEST,
        source_identity_path=SOURCE_IDENTITY,
        audit_path=OBSERVATION_AUDIT,
        evidence_path=EVIDENCE,
        model_registry_path=MODEL_REGISTRY,
        runtime_volume=VOLUME,
        container_root=CONTAINER_ROOT,
        predecessor_artifact_count=PREDECESSOR_ARTIFACT_COUNT,
        predecessor_anchor_sha256=PREDECESSOR_ANCHOR_SHA256,
        forbidden_volumes=FORBIDDEN_VOLUMES,
        verification_commands_factory=_verification_commands,
        protocol_artifact_registry_field="v23_protocol_artifact_sha256",
        protocol_artifact_count=14,
        manifest_boundaries=(
            ("current_default_baseline", "original-full-query-ngr-default"),
            ("positive_clause_ablation", "positive-clause-ngr-prefilter-ablation"),
            (
                "candidate_selection_policy",
                "lowest-development-primary-latency",
            ),
            ("network", "none"),
            ("corpus_acquisition", "read-only-fixed-git-objects"),
        ),
        container_identity_environment="NGR_V23_CONTAINER_IDENTITY",
        container_identity_prefix="ngr-v23-real",
    )
)

WORKERS = RUNTIME.workers
SOURCE_ROOT_SPEC = RUNTIME.source_root_spec
STAGE_CONTRACT = RUNTIME.stage_contract
V23RankObservationSpec = intent_aware_observation_runtime.IntentAwareRankObservationSpec
SPEC = RUNTIME.spec
TERMINAL_AUDIT = RUNTIME.terminal_audit

_validate_worker_fixtures = FIXTURE_CONTRACT.validate_worker
_validate_protocol_fixtures = FIXTURE_CONTRACT.validate_protocol
_container_worker = RUNTIME.container_worker
_container_claim = RUNTIME.container_claim
_container_finalize = RUNTIME.container_finalize
_container_fail_stage = RUNTIME.container_fail_stage
validate_prebuild = SPEC.validate_prebuild
verify_preflight = SPEC.verify_preflight
preflight = RUNTIME.preflight
finalize_preflight_error = RUNTIME.finalize_preflight_error
run_once = RUNTIME.run_once
audit_evidence = RUNTIME.audit_evidence
main = RUNTIME.main


if __name__ == "__main__":
    raise SystemExit(main())
