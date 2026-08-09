# github-rag-mcp replacement compatibility spike

## Purpose

This spike checks whether NGR can remain a replacement candidate for the
smallest read-only GitHub search path. It does not claim production parity with
`github-rag-mcp`.

The experiment has three separable boundaries:

1. `tools/acquire_github_snapshot.py` reads selected files from one public
   GitHub repository through `GET` requests, resolves the requested ref to a
   commit SHA, and writes a reviewable pinned snapshot.
2. `neuron_graph_rag.github_source` is the source adapter. It accepts that
   snapshot and upserts documents and provenance into a local NGR index. The
   NGR engine has no GitHub client or GitHub-specific retrieval path.
3. `tools/run_github_rag_compatibility.py` searches the local index and
   compares its source URLs and explanations against fixed expected-source
   contracts. It never calls the production github-rag-mcp service.

## Reproduction

Acquire a single public repository read-only. The `--path` values should be
chosen before the comparison result is created.

```powershell
python tools/acquire_github_snapshot.py `
  --repo Liplus-Project/github-rag-mcp `
  --ref main `
  --path docs/Home.md `
  --output local.github-rag-mcp.snapshot.json
```

Run the deterministic local comparison, then acquire one later snapshot of the
same repository and pass it as `--updated-snapshot`.

```powershell
python tools/run_github_rag_compatibility.py `
  --snapshot tests/fixtures/github_rag_compatibility.snapshot.json `
  --updated-snapshot tests/fixtures/github_rag_compatibility.updated.snapshot.json `
  --cases tests/fixtures/github_rag_compatibility.cases.json `
  --output local.github-rag-mcp.compatibility.result.json
```

The committed snapshots preserve one historical public GitHub document and its
one-document follow-up. They exercise the adapter and runner, but do not query
or measure a live production index. A candidate result is meaningful only when
its snapshot and expected-source cases are both preserved alongside the
generated result.

## Result contract

The result contains, for each fixed query:

- the explicit expected source from the github-rag-mcp comparison contract;
- NGR rank, source URL, node ID, and `SearchHit.explain()` rationale;
- whether the expected source is present in NGR's result set; and
- the changed paths and reindexed node IDs after one source update.

The runner reports exactly one bounded conclusion:

- `continue_candidate`: every expected source is found and a provided update is
  reflected in the local index;
- `incompatible`: an update was provided but retrieval or update following did
  not meet that contract; or
- `inconclusive`: no update snapshot was supplied.

## Boundaries and missing functionality

This is not a production replacement. NGR still lacks GitHub issue, pull
request, review, release, comment, and commit-diff acquisition; GitHub
authentication; webhooks and polling; vector infrastructure; reranking; MCP
transport; authentication; and remote deployment. The default NGR API and the
github-rag-mcp production service are unchanged.
