# OmniAgent Evaluation

Run the deterministic offline baseline with:

```bash
python -m evaluation.run
```

The command executes 60 labelled cases and writes a JSON report to
`artifacts/evaluation/latest.json`. The bundled corpus is a transparent engineering
fixture, not a claim about production traffic. Before the final resume metrics are frozen,
add the private/public-document release corpus and retain its source manifest.

Case groups cover lexical retrieval, citation provenance, task idempotency, schema
rejection, and rank fusion. Provider, timeout, MCP subprocess, cross-channel retry, and
WebUI contracts remain covered by pytest because they require async/process fixtures.
