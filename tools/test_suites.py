"""Explicit module roles; mixed product/experiment modules stay in normal runs."""

GROUPS = {
    "product": {
        "reason": "公開API、永続化、検索、feedback、CLI/MCPの製品回帰。",
        "modules": """
test_cli_and_eval test_confirmed_outcome_feedback test_database_home
test_decision_wiki_import test_decision_wiki_incremental test_engine test_exclusion_intent
test_evidence_quorum_feedback test_feedback_ledger test_node_expiry test_node_provenance
test_github_rag_compatibility test_judgments test_late_supersession test_mcp_adapter
test_outcome_feedback_deactivation test_outcome_feedback_deactivation_interleaving
test_precision_control
test_sibling_relation_feedback test_soft_start_feedback test_semantic_retrieval
test_cpu_shortlist_retrieval test_cuda_shortlist_retrieval
test_runtime_wheel test_verified_updates test_windows_package_artifacts test_tray_icon
test_shared_home test_native_registration test_shared_home_mcp
""".split(),
    },
    "shared": {
        "reason": "製品のD1 fixture・benchmark経路と、テスト分類自身およびdocs→Wiki同期のsource側検証。通常実行にも残す。",
        "modules": """
test_d1_fixture test_real_corpus_benchmark test_suite_selection test_wiki_sync_source
""".split(),
    },
}

SUITES = {
    "normal": ("product", "shared"),
    "experiments": ("shared",),
    "all": ("product", "shared"),
}
