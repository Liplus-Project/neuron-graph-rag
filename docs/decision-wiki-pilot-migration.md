# Decision Wiki pilot migration

## Boundary

This pilot imports only entries enumerated by each repository-owned `Decision-Structure.md` index. By default the tool reads the mirror in each Wiki clone; `--liplus-index` and `--ngr-index` select an explicit canonical index when the mirror is behind its repository. Unclassified Wiki pages are not discovered or imported. The Wiki clones, existing search databases, and frozen feedback experiment databases remain read-only. The pilot database is a disposable, dedicated SQLite file and does not switch the source-of-truth boundary.

Each stable identity is `<repository-name>:<slug>`, so equal slugs in different repositories do not collide. Provenance retains the full page body, page SHA-256, repository, Wiki URL, source index state, and exact Wiki commit. The import accepts `supersedes`, `depends on`, `conflicts with`, `refines`, and `informs`; spaces are normalized to underscores in SQLite relation identifiers.

Index states `active` and `evaluating` import as active. `archived` and `superseded` import as archived. A `supersedes` relation also archives its predecessor and records `superseded_by`. Duplicate identities, missing indexed pages, unknown Decision targets, multiple successors, ambiguous edge declarations, and partial publication fail closed.

## Reproduction

Run from the repository root with clean, independently acquired Wiki clones and paths that do not exist yet:

```powershell
python tools/import_decision_wikis.py `
  --liplus-wiki C:\path\to\liplus-language.wiki `
  --ngr-wiki C:\path\to\neuron-graph-rag.wiki `
  --ngr-index C:\path\to\neuron-graph-rag\docs\Decision-Structure.md `
  --database C:\path\to\pilot\decisions.sqlite `
  --export C:\path\to\pilot\decisions.export.json `
  --backup C:\path\to\pilot\decisions.backup.sqlite `
  --manifest C:\path\to\pilot\decisions.manifest.json
```

The command refuses to overwrite any output. It builds all four outputs under temporary sibling names, imports all records in one transaction, checks SQLite and judgment supersession integrity, independently exports the database twice and compares the bytes, verifies the SQLite backup, and records source commits and counts in the manifest. Only after every validation succeeds are the database, export, backup, and manifest published as one bundle. Any build or publication failure removes every temporary and newly published output, leaving no partial set.

The committed `decision-wiki-pilot-manifest.json` records the completed pilot without committing the SQLite binary, backup, export, or Wiki clones.
