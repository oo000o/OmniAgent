# OmniAgent Evaluation

Run the deterministic offline baseline with:

```bash
python -m evaluation.run
```

The command executes 92 labelled cases and writes a JSON report to
`artifacts/evaluation/latest.json`. The bundled corpus is a transparent engineering
fixture, not a claim about production traffic.

## Seal suite (ablation + benchmark + fault aggregate)

```bash
python -m evaluation.run_seal
```

Writes:

- `artifacts/evaluation/retrieval_ablation_latest.json` — ~40 manually labelled queries
  across BM25 / Vector / Hybrid+RRF / +Rewrite / +Rerank / Final
- `artifacts/evaluation/benchmark_latest.json` — local P50/P95 style timings
- `artifacts/evaluation/fault_injection_report_latest.json` — compact fault aggregates

**These numbers are local/repository evaluation results, not production SLA.**

Retrieval pipeline under test:

```text
Query Analysis
→ Conditional Rewrite (complex / vague / multi_intent only; fallback to original)
→ BM25 + Vector
→ RRF
→ Cross-Encoder Rerank (RRF Top-N only; fallback to RRF)
→ Citation / evidence binding
```

CI keeps rewrite/rerank disabled by default so the original 12-query deterministic
fixture and the 92-case baseline stay stable. Optional neural CrossEncoder lives behind
the `rerank` extra (`sentence-transformers`); offline seal runs use a deterministic
lexical overlap scorer for reproducibility.

Case groups in the baseline cover lexical retrieval, citation provenance, task
idempotency, schema rejection, rank fusion, career-workflow reliability, run
observability aggregates, and fault-injection recovery. The same baseline report also
compares BM25, deterministic vector retrieval, and hybrid RRF on 12 labelled queries
using Recall@3, MRR, and NDCG@3.
