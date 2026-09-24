"""Keep the checkout's experiment imports without shipping them in the runtime wheel."""

from setuptools import setup
from setuptools.command.build_py import build_py


# Runtime API, three existing CLI commands, optional MCP adapter, and documented
# opt-in retrieval APIs. New modules are checkout-only until explicitly reviewed.
RUNTIME_MODULES = frozenset(
    {
        "__init__",
        "__main__",
        "benchmark",
        "cli",
        "config_provenance",
        "cpu_shortlist_retrieval",
        "d1_fixture",
        "database_home",
        "dynamics",
        "engine",
        "evaluation",
        "evidence_feedback",
        "exclusion_intent",
        "feedback",
        "judgments",
        "models",
        "ontology",
        "precision_control",
        "retrieval",
        "sample",
        "semantic_retrieval",
        "storage",
    }
)


class RuntimeBuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if package == "neuron_graph_rag":
            return [item for item in modules if item[1] in RUNTIME_MODULES]
        return modules


setup(cmdclass={"build_py": RuntimeBuildPy})
