# Agentic AI Course

Hands-on curriculum to learn AI system evolution from **GenAI -> AI Agents -> Agentic AI** with practical coding.

This course is designed for:

- CSE students building strong AI fundamentals + implementation skills
- QA engineers transitioning to AI engineering with an evaluation-first mindset

Primary course guide: `agentic_ai_tutorial.md`

## 5-day learning path

### Day 1 - GenAI fundamentals

- LLM basics, prompting, temperature/system prompts
- Files: `examples/day_01_genai/`

### Day 2 - AI agents

- Tool use, memory, basic agent workflows
- Files: `examples/day_02_agents/`

### Day 3 - Agentic orchestration

- Multi-step autonomous workflows with retrieval-first design
- Files: `examples/day_03_agentic_ai/`

### Day 4 - Local RAG with Chroma

- Build local vector DB flows, grounding, and retrieval quality checks
- Files: `examples/day_04_bonus_rag/`

### Day 5 - Frameworks and production path

- LangChain, LangGraph, Hugging Face embeddings, FastAPI/Streamlit/pgvector
- Files: `examples/day_05_level_2/` and `examples/day_05_level_3/`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` at repo root:

```bash
GEMINI_API_KEY=your_api_key_here
```

Optional for Day 5 pgvector:

```bash
PGVECTOR_DATABASE_URL=postgresql://user:password@localhost:5432/dbname
```

## Where to study next

- Day-wise curated references: `resources/references/`
- Prompt templates: `resources/prompt_templates/study_buddy_prompts.md`
- QA evaluation lane: `resources/qa_lane/`
- RAG seed corpus: `resources/rag_corpus/`

## Recommended daily workflow

1. Read one day section in `agentic_ai_tutorial.md`.
2. Run the corresponding scripts in `examples/`.
3. Save notes/reports under `resources/study_notes/` and `resources/study_reports/`.
4. Execute QA checks from `resources/qa_lane/evaluation_checklist.md`.

## Quick verification

```bash
python -m compileall examples
python examples/day_05_level_3/level3_rag_vs_finetune_comparison.py
```

