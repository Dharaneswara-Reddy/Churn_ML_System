# Knowledge Graph

A navigable map of this codebase — 1,659 nodes and 2,765 edges across 72 clusters,
built from the source, the docs and the architecture diagrams together.

## Why it exists

Answering "how does promotion actually work?" normally means reading a dozen files
across four packages. The graph holds those relationships explicitly, so the
answer is a traversal rather than a re-read. Measured on this repository:

```
Corpus:          128,549 words  →  ~171,398 tokens if read naively
Graph:           1,659 nodes, 2,765 edges
Avg query cost:  ~12,167 tokens
Reduction:       14.1x fewer tokens per question
```

It also survives a change of editor. The graph is plain files in this repository,
so it works the same in VS Code, Cursor, JetBrains, or a terminal — nothing is
tied to one IDE's index.

## Files

| File | What it is |
|---|---|
| [`knowledge-graph.html`](knowledge-graph.html) | Interactive graph. Open in any browser, no server needed. |
| [`knowledge-graph.json`](knowledge-graph.json) | Raw graph, for programmatic queries or GraphRAG. |
| [`GRAPH_REPORT.md`](GRAPH_REPORT.md) | Audit report: clusters, cohesion scores, god nodes, surprises. |

## How to read it

**God nodes** are the most connected symbols — the abstractions everything else
leans on. Change one and the blast radius is wide:

| Symbol | Edges |
|---|---:|
| `ModelRegistry` | 90 |
| `OutboxEvent` | 89 |
| `OutboxStatus` | 62 |
| `ErrorBody` | 45 |
| `PredictionEvent` | 40 |
| `ReadWriteLock` | 24 |
| `verify_bundle_signature()` | 21 |

That `ModelRegistry` and `OutboxEvent` sit at the top is the architecture stating
itself: this system is a serving layer around a hot-swappable model, and a
durable event trail behind it.

**Every edge carries a confidence tag**, so the map is honest about what was found
versus inferred:

- `EXTRACTED` — explicit in the source (an import, a call, a citation)
- `INFERRED` — a reasonable deduction (shared data structure, implied dependency)
- `AMBIGUOUS` — uncertain, flagged rather than dropped

The `AMBIGUOUS` tag earned its keep on the first run: it flagged two documentation
contradictions that had gone unnoticed — `docs/training.md` claiming the selection
metric defaults to `roc_auc` while `settings.yaml` and `docs/config.md` said
`pr_auc`, and a stale description of the training split that survived the change
from a tenure-sorted temporal split to a stratified holdout. Both are now fixed.

**Rationale is a first-class node type.** Design decisions with a written *why*
become nodes linked by `rationale_for` edges — why geography was excluded, why the
split is stratified, why the signature is verified before unpickling, why batch
chunking was removed. That is usually the part lost when a codebase changes hands.

## Rebuilding it

The graph is a snapshot; it goes stale as the code moves.

```bash
/graphify .                # full rebuild
/graphify . --update       # only re-extract new or changed files
```

`--update` skips the language model entirely when only code changed — AST
extraction is deterministic and free.

Then refresh the published copies:

```bash
cp graphify-out/graph.html       docs/graph/knowledge-graph.html
cp graphify-out/graph.json       docs/graph/knowledge-graph.json
cp graphify-out/GRAPH_REPORT.md  docs/graph/GRAPH_REPORT.md
```

## Querying it

```bash
/graphify query "how does a model get promoted to production"
/graphify path "ModelRegistry" "OutboxEvent"
/graphify explain "verify_bundle_signature"
```

For agents, `--mcp` serves the graph over MCP so tools like `get_neighbors` and
`shortest_path` are callable directly.
