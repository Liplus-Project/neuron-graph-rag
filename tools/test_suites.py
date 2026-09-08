"""Explicit module roles; mixed product/experiment modules stay in normal runs."""

GROUPS = {
    "product": {
        "reason": "公開API、永続化、検索、feedback、CLI/MCPの製品回帰。",
        "modules": """
test_cli_and_eval test_confirmed_outcome_feedback test_database_home
test_decision_wiki_import test_engine test_exclusion_intent test_feedback_ledger
test_github_rag_compatibility test_judgments test_mcp_adapter
test_outcome_feedback_deactivation test_outcome_feedback_deactivation_interleaving
test_sibling_relation_feedback test_soft_start_feedback
""".split(),
    },
    "shared": {
        "reason": "製品engine、共通選択・監査・実験runtime契約を含むため、混在モジュールを丸ごと通常実行に残す。",
        "modules": """
test_anchored_hybrid test_baseline_aware_soft_start_snapshot_evaluation
test_blind_selection test_canonical_gate_evaluation test_channels
test_corpus_integrity test_d1_fixture test_dynamics_experiment
test_engine_feedback_trajectory test_evidence_quorum_feedback
test_feedback_adaptation_experiment test_feedback_adaptation_reproduction
test_feedback_policy_comparison_corpus test_feedback_policy_comparison_evaluation
test_fresh_native_feedback_evaluation test_fusion_calibration
test_github_retrieval_parity test_github_retrieval_parity_observation
test_github_retrieval_parity_v2 test_github_retrieval_parity_v3
test_github_retrieval_parity_v4
test_intent_aware_observation_engine test_local_competition
test_longitudinal_controlled_corpus_v3 test_node_first_selection
test_outcome_feedback_deactivation_evaluation test_precision_control
test_precision_control_observation test_rank_elasticity test_real_corpus_benchmark
test_real_task_shadow test_real_task_shadow_v2 test_real_task_shadow_v3
test_sibling_normalization_evaluation test_soft_start_snapshot_evaluation
test_source_grounded_relation_observation test_source_grounded_relation_observation_v2
test_source_grounded_relation_observation_v3 test_source_grounded_relation_observation_v3_probe
test_suite_selection
""".split(),
    },
    "historical": {
        "reason": "凍結cross-encoder各版の再現・監査。共有runtime/検索変更時はallで検証し、版間の依存を推測して間引かない。",
        "modules": """
test_cross_encoder_precision test_cross_encoder_precision_observation
test_cross_encoder_precision_v2 test_cross_encoder_precision_v2_observation
test_cross_encoder_precision_v3 test_cross_encoder_precision_v3_observation
test_cross_encoder_precision_v4 test_cross_encoder_precision_v4_observation
test_cross_encoder_precision_v5 test_cross_encoder_precision_v5_observation
test_cross_encoder_precision_v6 test_cross_encoder_precision_v7
test_cross_encoder_precision_v8 test_cross_encoder_precision_v8_observation
test_cross_encoder_precision_v9 test_cross_encoder_precision_v9_performance_observation
test_cross_encoder_precision_v10 test_cross_encoder_precision_v10_performance_observation
test_cross_encoder_precision_v11_observation test_cross_encoder_precision_v12_performance_observation
test_cross_encoder_precision_v13_observation test_cross_encoder_precision_v14_performance_observation
test_cross_encoder_precision_v15_observation test_cross_encoder_precision_v16_observation
test_cross_encoder_precision_v17_performance_observation test_cross_encoder_precision_v18_performance_observation
test_cross_encoder_precision_v19_performance_observation test_cross_encoder_precision_v20_intent_aware_freeze
test_cross_encoder_precision_v21_intent_aware_observation test_cross_encoder_precision_v22_intent_aware_observation
test_cross_encoder_precision_v23_real_task_observation
""".split(),
    },
}

SUITES = {
    "normal": ("product", "shared"),
    "experiments": ("shared", "historical"),
    "all": ("product", "shared", "historical"),
}
