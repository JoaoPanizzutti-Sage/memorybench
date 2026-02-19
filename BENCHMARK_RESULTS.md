# MemoryBench Results

## LongMemEval (500 questions, 115k+ tokens)

### Overall Comparison

| Provider | Accuracy |
|----------|----------|
| Supermemory | 85.9% |
| **Our RAG** (GPT-5 + GPT-5-mini) | **80.8%** |
| OpenClaw QMD | 58.3% |
| Filesystem (Claude code style) | 54.2% |

Supermemory numbers from [@DhravyaShah's post](https://x.com/DhravyaShah) (Feb 17, 2026).

### Per-Category Breakdown (Our RAG)

| Category | Questions | Accuracy |
|----------|-----------|----------|
| single-session-assistant | 56 | 96.4% (54/56) |
| single-session-preference | 30 | 90.0% (27/30) |
| single-session-user | 70 | 87.1% (61/70) |
| knowledge-update | 78 | 85.9% (67/78) |
| temporal-reasoning | 133 | 75.9% (101/133) |
| multi-session | 133 | 70.7% (94/133) |

### Retrieval Quality (K=10)

| Metric | Value |
|--------|-------|
| Hit@K | 96.8% |
| Recall@K | 96.8% |
| MRR | 0.906 |
| NDCG | 0.917 |

## Architecture

Fully open-source RAG pipeline, no proprietary memory APIs:

```
Conversations
    |
    v
LLM Extraction (GPT-5-mini)
    |
    v
Chunking + Embedding (text-embedding-3-small)
    |
    v
Hybrid Search (BM25 + Vector + LLM Reranker)
    |
    v
Entity Graph (knowledge graph for multi-hop)
    |
    v
Answer Generation (GPT-5, reasoningEffort: medium)
```

## What's Working

- **Preferences (90%)**: Chain of thought + synthesis instructions let the model use retrieved context about user preferences to generate personalized suggestions.
- **Assistant recall (96.4%)**: Structured extraction captures assistant-generated content well.
- **Knowledge-update (85.9%)**: "Most recent wins" logic in the answer prompt is effective.
- **Multi-session (70.7%)**: Entity graph + multi-hop retrieval handles relationship chains.
- **Temporal (75.9%)**: Explicit "calculate from Question Date, NOT today" instruction fixed date arithmetic.
- **Retrieval quality**: 96.8% Hit@K, 0.906 MRR. Hybrid BM25+vector search with LLM reranker finds the right documents.

## Remaining Gap

The 5.1 point gap to supermemory comes primarily from `single-session-user` (87.1%). This category tests recall of facts stated by the user in conversation. The extraction step likely drops some user-stated details that supermemory's native ingestion captures.
