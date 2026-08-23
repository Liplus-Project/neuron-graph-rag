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

The command refuses to overwrite any output. It builds the database under a temporary sibling name, imports all records in one transaction, checks SQLite and judgment supersession integrity, writes a deterministic export, creates a SQLite backup, and records source commits and counts in the manifest. Running the export twice over the same database produces byte-identical JSON.

The committed `decision-wiki-pilot-manifest.json` records the completed pilot without committing the SQLite binary, backup, export, or Wiki clones.
