# Agentic AI Course

A hands-on, project-first tutorial for students to learn Generative AI -> AI Agents -> Agentic AI systems by building one evolving project: **AI Study Buddy**.

This repository is organized for a **3-5 day learning path** based on `agentic_ai_tutorial.md`.

## What students build

- Day 1: a GenAI-powered study notes generator
- Day 2: a tool-using agent with memory and quiz generation
- Day 3: an autonomous multi-step exam coach
- Bonus: RAG with ChromaDB
- Level 2: web search, RAG integration, LangGraph orchestration
- Level 3: FastAPI service, Streamlit UI, production VectorDB concepts

## Course source

- Full tutorial: `agentic_ai_tutorial.md`
- Starter script currently in repo: `main.py`

## 3-5 day step-by-step guide

### Day 1 (Core): Generative AI fundamentals

- Goal: understand LLM basics, API calls, and prompt engineering.
- Modules from tutorial:
  - Module 1.1: What is Generative AI?
  - Module 1.2: Setup + first API call
  - Module 1.3: Prompt engineering
  - Module 1.4: Build study notes generator
  - Module 1.5: Limitations of basic GenAI
- Examples to add under `examples/day_01_genai/`:
  - `verify_setup.py`
  - `day1_hello_genai.py`
  - `day1_prompt_engineering.py`
  - `day1_system_prompt.py`
  - `day1_temperature.py`
  - `day1_study_notes_generator.py`

### Day 2 (Core): AI agents (tools + memory)

- Goal: build agents that can reason, use tools, and remember context.
- Modules from tutorial:
  - Module 2.1: What is an AI agent?
  - Module 2.2: Tools
  - Module 2.3: Memory
  - Module 2.4: Build research and quiz agent
- Examples to add under `examples/day_02_agents/`:
  - `day2_function_calling_basics.py`
  - `day2_memory.py`
  - `day2_research_quiz_agent.py`

### Day 3 (Core): Agentic AI systems

- Goal: multi-step orchestration and autonomous behavior.
- Modules from tutorial:
  - Day 3 architecture and autonomous flow
  - Module 3.4: Comparing versions
  - Module 3.5: Best practices
- Examples to add under `examples/day_03_agentic_ai/`:
  - `day3_orchestration_basics.py`
  - `day3_autonomous_exam_coach.py`

### Day 4 (Bonus): VectorDB + RAG

- Goal: ground answers on your own content and reduce hallucinations.
- Tutorial build section:
  - StudyBuddy RAG demo with ChromaDB
- Example to add under `examples/day_04_bonus_rag/`:
  - `bonus_vectordb_rag_demo.py`

### Day 5 (Advanced tracks): Level 2 and Level 3

- Goal: move from prototypes to production-ready patterns.
- Level 2 examples under `examples/day_05_level_2/`:
  - `level2_web_search_agent.py`
  - `level2_studybuddy_with_rag.py`
  - `level2_langgraph_studybuddy.py`
- Level 3 examples under `examples/day_05_level_3/`:
  - `level3_fastapi_service.py`
  - `level3_streamlit_app.py`
  - `level3_pgvector_demo.py`
  - `level3_rag_vs_finetune_comparison.py`

## Repository structure

```text
agentic_ai_course/
├── README.md
├── agentic_ai_tutorial.md
├── requirements.txt
├── .gitignore
├── main.py
├── examples/
│   ├── day_01_genai/
│   ├── day_02_agents/
│   ├── day_03_agentic_ai/
│   ├── day_04_bonus_rag/
│   ├── day_05_level_2/
│   └── day_05_level_3/
└── resources/
    ├── study_notes/
    ├── study_reports/
    └── vector_store/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in repo root:

```bash
GEMINI_API_KEY=your_api_key_here
```

## Recommended student workflow

1. Read the corresponding day section in `agentic_ai_tutorial.md`.
2. Copy/build the example file for that module in the matching `examples/` folder.
3. Run, test, and iterate before moving to the next module.
4. Keep outputs in `resources/study_notes/` and `resources/study_reports/`.
5. At the end of each day, write a short reflection on what changed from previous version.

## Notes

- Use `google-genai` (not `google-generativeai`) for this tutorial.
- Core completion target: Days 1-3.
- Strong portfolio target: complete Bonus + Level 2.
- Deployment-ready target: complete Level 3.

