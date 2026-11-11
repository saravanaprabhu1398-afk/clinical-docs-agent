# Clinical Docs Agent

An AI agent that answers clinicians' and pharmacists' questions about prescription drugs, citing the exact FDA label section behind every answer, and saying so when the labels don't cover the question.

> **Status: Week 1.** The repo skeleton is in place and 149 public FDA drug labels are collected. Retrieval, the agent, evals and deployment are next (see [Roadmap](#roadmap)).

## The problem

Drug labels are long (often 30+ pages), dense, and change often. Questions such as *"Can a patient on warfarin take fluconazole?"* or *"What's the renal dose adjustment for metformin?"* need an answer in seconds, and a wrong one is costly. A plain chatbot hallucinates. This agent must:

- answer only from the labels, with a citation (drug, section, label date) for every claim
- refuse or hedge when the labels don't cover the question
- route structured questions (counts, comparisons across drugs) to SQL instead of free-text search

## Data

| | |
|---|---|
| Source | [openFDA drug label API](https://open.fda.gov/apis/drug/label/): FDA Structured Product Labels, public domain |
| Size | 149 commonly prescribed generics across cardiology, diabetes, psychiatry, neurology, pain, respiratory, GI, infection, endocrine and oncology |
| Format | `data/docs/<generic>.md`: YAML front matter (generic, brand, manufacturer, route, class, set ID, label date, DailyMed link) + one `##` section per label part (boxed warning, indications, dosage, contraindications, warnings, adverse reactions, interactions, specific populations, ...) |
| PHI | None. These are published product labels, not patient records. |

Rebuild the corpus:

```bash
python scripts/fetch_openfda_labels.py          # writes data/raw/*.json (git-ignored) and data/docs/*.md
```

## Architecture (planned)

```
question ──► agent (tool calling) ──► search_labels   (hybrid BM25 + vectors, rerank, filters by drug/section)
                    │              ──► sql_lookup      (DuckDB over label metadata: class, route, boxed warnings)
                    │              ──► compare_drugs   (side-by-side section pull for 2+ drugs)
                    ▼
            answer + citations ──► guardrails (PII redaction, out-of-scope refusal) ──► FastAPI ──► UI
                    │
                    └──► tracing (MLflow / Langfuse) + eval harness (golden set, recall@k, faithfulness)
```

## Roadmap

- [x] **Week 1:** domain, repo skeleton, corpus of 149 public drug labels
- [ ] **Week 2:** ingestion (section-aware chunking, embeddings, store); hybrid search + rerank; CLI that answers with citations
- [ ] **Week 3:** agent with 3–4 tools (search labels, SQL lookup, compare drugs, report)
- [ ] **Week 4:** 50-question golden eval set; baseline scores; one measured improvement; tracing
- [ ] **Week 5:** FastAPI + Pydantic, guardrails, pytest, Dockerfile
- [ ] **Week 6:** UI, public deployment, architecture diagram, eval results, 3-minute demo

## Evaluation (planned)

| Metric | What it measures |
|---|---|
| Recall@5 / MRR | Did retrieval find the right label section? |
| Faithfulness | Is every claim supported by a cited chunk? (LLM-as-judge, spot-checked by hand) |
| Refusal accuracy | Does it decline questions the labels don't answer? |
| Latency p50 / p95, cost per question | Is it usable in practice? |

## Repo layout

```
data/docs/            Markdown corpus (one file per drug label)
scripts/              data collection
src/clinical_docs_agent/   package (ingestion, retrieval, agent, API): Week 2+
tests/                pytest
```

## Disclaimer

A portfolio project for learning. Not medical advice; always check the current official label.
