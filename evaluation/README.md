# OmniAgent Evaluation

Run the deterministic offline baseline with:

```bash
python -m evaluation.run
```

The command executes 75 labelled cases and writes a JSON report to
`artifacts/evaluation/latest.json`. The bundled corpus is a transparent engineering
fixture, not a claim about production traffic. Before the final resume metrics are frozen,
add the private/public-document release corpus and retain its source manifest.

Case groups cover lexical retrieval, citation provenance, task idempotency, schema
rejection, and rank fusion. The same report also compares BM25, deterministic vector
retrieval, and hybrid RRF on 12 labelled queries using Recall@3, MRR, and NDCG@3.
The deterministic embedding fixture makes CI reproducible and does not represent a
production model score. Provider, timeout, MCP subprocess, cross-channel retry, and WebUI
contracts remain covered by pytest because they require async/process fixtures.

The report also executes the career workflow through real tool contracts and isolated
SQLite/Cron stores. It reports separate pass rates for the successful path, guardrail
rejections, and restart/idempotency recovery. These are deterministic offline scenario
rates—not production availability, traffic, or user-success metrics.
