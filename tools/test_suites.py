"""Explicit module roles; mixed product/experiment modules stay in normal runs."""

GROUPS = {
    "product": {
        "reason": "公開API、永続化、検索、feedback、CLI/MCPの製品回帰。",
        "modules": """
test_cli_and_eval test_confirmed_outcome_feedback test_database_home
test_decision_wiki_import test_decision_wiki_incremental test_engine test_exclusion_intent
test_feedback_ledger test_node_expiry test_node_provenance
test_github_rag_compatibility test_judgments test_late_supersession test_mcp_adapter
test_outcome_feedback_deactivation test_outcome_feedback_deactivation_interleaving
test_sibling_relation_feedback test_soft_start_feedback test_semantic_retrieval
test_cpu_shortlist_retrieval test_cpu_shortlist_benchmark test_cuda_shortlist_retrieval
test_runtime_wheel test_verified_updates test_windows_package_artifacts test_tray_icon
test_shared_home test_native_registration test_shared_home_mcp
""".split(),
    },
    "shared": {
        "reason": "製品engine、共通選択・監査・実験runtime契約を含むため、混在モジュールを丸ごと通常実行に残す。",
        "modules": """
test_anchored_hybrid test_channels test_d1_fixture test_dynamics_experiment
test_evidence_quorum_feedback test_fusion_calibration test_local_competition
test_longitudinal_controlled_corpus_v3 test_precision_control
test_precision_control_observation test_real_corpus_benchmark
test_real_task_shadow_v2 test_real_task_shadow_v3
test_deduplicate_experiment_models test_suite_selection
""".split(),
    },
}

SUITES = {
    "normal": ("product", "shared"),
    "experiments": ("shared",),
    "all": ("product", "shared"),
}
