# 🤖 Agentic AI: From Zero to Autonomous Systems
### A 5-Day Hands-On Tutorial for College Students and QA Engineers

> **Prerequisites:** Basic Python (functions, loops, dictionaries, pip installs)
> **Time Commitment:** 3-4 hours per day
> **Trainer's Note:** Every concept in this course is built around one evolving story - your AI Study Buddy. Day 1-3 build foundations, Day 4 adds local Chroma RAG, and Day 5 adds LangChain/LangGraph/Hugging Face + production path.
> **Code Policy (2026 update):** Runnable code is maintained under `examples/`. This tutorial focuses on concepts, architecture, and file references.

---

## 🗺️ Course Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   COMPLETE LEARNING JOURNEY                                          │
├──────────────┬─────────────────┬──────────────┬────────────────┬──────────────────┬─────────────────┤
│  DAY 1       │  DAY 2          │  DAY 3       │  DAY 4         │  DAY 5           │                 │
│ Generative AI│  AI Agents      │ Agentic AI   │ Chroma RAG     │ Frameworks+Prod  │                 │
│ (~3.5 hrs)   │  (~3.5 hrs)     │ (~4 hrs)     │ (~2.5 hrs)     │  (~4-5 hrs)      │                 │
├──────────────┼─────────────────┼──────────────┼────────────────┼──────────────────┼─────────────────┤
│ • Gen AI     │ • Agent Loop    │ • Multi-Agent│ • Embeddings   │ • Real web search│ • FastAPI       │
│ • LLMs/APIs  │ • Tools/Memory  │ • Orchestrat.│ • ChromaDB RAG │ • RAG in v3.0    │ • Streamlit UI  │
│ • Prompts    │ • ReAct pattern │ • Auto coach │ • RAG pipeline │ • LangGraph      │ • pgvector      │
├──────────────┼─────────────────┼──────────────┼────────────────┼──────────────────┼─────────────────┤
│ v1.0         │ v2.0            │ v3.0         │ v3.5           │ v3.6             │ v4.0            │
│ Smart        │ Tutor with      │ Autonomous   │ Coach with     │ Coach with live  │ Production      │
│ textbook     │ tools           │ coach        │ YOUR docs      │ web + LangGraph  │ web service     │
└──────────────┴─────────────────┴──────────────┴────────────────┴──────────────────┴─────────────────┘

  ✅ Core: Days 1-3
  ✅ RAG milestone: Day 4
  ✅ Framework + production milestone: Day 5
```

## 🔗 Code and Resources Map

- Day 1 examples: `examples/day_01_genai/`
- Day 2 examples: `examples/day_02_agents/`
- Day 3 examples: `examples/day_03_agentic_ai/`
- Day 4 examples: `examples/day_04_bonus_rag/`
- Day 5 examples: `examples/day_05_level_2/` and `examples/day_05_level_3/`
- Resource hub: `resources/README.md`
- Further-study videos (<=2h/topic): `resources/references/`

---

## 🎯 Our Evolving Use Case: The AI Study Buddy

Imagine you have an AI that helps you prepare for your **Data Structures exam**. Over 3 days, we'll build it progressively:

| Day | Version | What It Does | How Smart Is It? |
|-----|---------|-------------|-----------------|
| 1 | **Basic Study Notes Generator** | You ask a topic → it explains it | Knows only what it was trained on |
| 2 | **Research & Quiz Agent** | Searches the web, makes flashcards, solves problems | Uses tools, remembers your conversation |
| 3 | **Autonomous Exam Coach** | Plans your week, researches topics, makes quizzes, grades you, adapts | Runs on its own, reports back to you |

---

# 📅 DAY 1: Generative AI — Teaching a Machine to Create

> ⏱️ **Estimated Time:** 3.5 hours
> 🎯 **Goal:** Understand how Gen AI works and build your first AI-powered tool

### 🏗️ Day 1 Architecture: Generative AI (Single-Turn)

```
╔═══════════════════════════════════════════════════════════════════════╗
║               GENERATIVE AI — HOW DATA FLOWS                         ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║   ① YOU TYPE A PROMPT          ② LLM PROCESSES IT                    ║
║   ┌───────────────────┐         ┌────────────────────────────────┐   ║
║   │  "Explain Binary  │─────►   │      GEMINI LLM (in cloud)     │   ║
║   │   Search Trees"   │         │                                │   ║
║   └───────────────────┘         │  Predicts next token, one     │   ║
║                                 │  at a time:                   │   ║
║                                 │  "A"→"BST"→"is"→"a"→"tree"   │   ║
║                                 └──────────────────┬─────────────┘   ║
║                                                    │                  ║
║                                    ③ RESPONSE RETURNED               ║
║                                    ▼                                  ║
║                            ┌────────────────┐                        ║
║                            │  Study Notes   │                        ║
║                            │  "A BST is a   │                        ║
║                            │   tree where..." │                       ║
║                            └────────────────┘                        ║
║                                                                       ║
║  ┌──────────────── WHAT THIS SYSTEM CANNOT DO ──────────────────┐   ║
║  │  ❌  Search the web for up-to-date info                       │   ║
║  │  ❌  Remember your name or past conversations                 │   ║
║  │  ❌  Run code or use any tools                                │   ║
║  │  ❌  Take any action on its own                               │   ║
║  └───────────────────────────────────────────────────────────────┘   ║
╚═══════════════════════════════════════════════════════════════════════╝
```

---

## Module 1.1 — What is Generative AI? (45 min)

### The Big Idea

Generative AI is AI that can **create**  — text, images, code, music, and more.

Think of it like this:
- 📚 **Traditional AI:** "Is this email spam? YES or NO."
- 🎨 **Generative AI:** "Write me a professional email declining a meeting."

Traditional AI *classifies* or *predicts*. Generative AI *creates*.

### The Magic Inside: Large Language Models (LLMs)

An LLM is the engine behind tools like ChatGPT, Gemini, and Claude.

**How was it trained?**

> 🧠 **First — What is a Neural Network? (30 seconds)**
> Think of your brain. It has billions of neurons connected by wires. When you learn something, those connections get stronger. A **neural network** is a software imitation of that — millions of tiny math functions (called "neurons") connected together. Feed it data, it makes a guess, it's corrected, the connections adjust. Repeat billions of times. That's how it learns. An LLM is a *very* large neural network trained specifically on text.

1. Developers collected hundreds of billions of words from the internet, books, and articles.
2. They trained a neural network to predict: *"Given these words, what word comes next?"*
3. After trillions of such predictions and corrections, the model learned language deeply.

```
Training Phase (Done by researchers — very expensive!):
─────────────────────────────────────────────────────
"The capital of France is ___"   →  Model guesses "Berlin"  ❌
                                 →  Correct answer: "Paris"
                                 →  Model adjusts internally
                                 →  Repeat 10 trillion times
─────────────────────────────────────────────────────
Result: A model that deeply understands language patterns
```

**Inference Phase (What YOU use):**

```
┌──────────────┐        ┌─────────────────────┐        ┌──────────────┐
│  Your Prompt │  ───►  │   LLM (e.g. Gemini) │  ───►  │   Response   │
│              │        │                     │        │              │
│ "Explain     │        │  Predicts next token│        │ "A binary    │
│  binary      │        │  one at a time...   │        │  search tree │
│  trees"      │        │                     │        │  is a data   │
└──────────────┘        └─────────────────────┘        └──────────────┘
```

### Key Terms You Must Know

| Term | Simple Explanation | Analogy |
|------|--------------------|---------|
| **Token** | A chunk of text (word/part of word) | Syllables in speech |
| **Context Window** | Max tokens the model can "see" at once | Short-term memory |
| **Temperature** | How creative/random the output is | 0 = boring but accurate, 1 = creative but wild |
| **Prompt** | Your input to the model | A question to a very smart person |
| **Completion/Response** | The model's output | The smart person's answer |
| **Parameters** | Internal numbers the model learned | Billions of tiny dials tuned during training |

### 🎬 Watch This First (~20 min)
> **"Intro to Large Language Models"** by Andrej Karpathy
> 🔗 https://www.youtube.com/watch?v=zjkBMFhNj_g
> *(Former Tesla & OpenAI researcher explains LLMs visually — perfect starting point!)*

---

## Module 1.2 — Setting Up & Your First API Call (30 min)

### Installation

```bash
# Create a virtual environment (good practice!)
python -m venv ai_study_env
source ai_study_env/bin/activate   # Mac/Linux
# ai_study_env\Scripts\activate   # Windows

# Install the Google Gemini SDK
# ⚠️  IMPORTANT: The package is called "google-genai" (NOT "google-generativeai")
#     These are TWO different packages with different import styles!
pip install google-genai python-dotenv
```

> 💡 **Tip: Install everything for this whole course in one go**
>
> The project `requirements.txt` has all dependencies pre-listed. Run:
> ```bash
> pip install -r requirements.txt
> ```
>
> For just Day 1, the minimal install is:
> ```bash
> pip install google-genai python-dotenv
> ```
>
> All other imports (`os`, `json`, `datetime`, `pathlib`, `random`, `dataclasses`, `typing`) are **Python standard library** — they ship with Python and need no installation.

> 💡 **Common Beginner Mistake:** If you `pip install google-generativeai` (the old package), Python will give you an `ImportError` when you write `from google import genai`. Always use `google-genai` for this tutorial.

### API Key Setup

1. Go to https://aistudio.google.com/app/apikey
2. Click **"Create API Key"** (it's FREE for learning!)
3. Create a `.env` file in your project:

```bash
# .env
GEMINI_API_KEY=your_api_key_here
```

> ⚠️ **NEVER commit your API key to GitHub.** Add `.env` to your `.gitignore`!

### Project Folder Structure

Before writing any code, set up your folder like this. All files in this tutorial go here:

```
ai_study_buddy/          ← your project root folder
├── .env                 ← API keys (NEVER share this!)
├── .gitignore           ← tells Git to ignore .env
├── day1_hello_genai.py
├── day1_prompt_engineering.py
├── day1_system_prompt.py
├── day1_study_notes_generator.py
├── day2_function_calling_basics.py
├── day2_memory.py
├── day2_research_quiz_agent.py
├── day3_orchestration_basics.py
├── day3_autonomous_exam_coach.py
├── study_notes/         ← auto-created when you run Day 1 code
└── study_reports/       ← auto-created when you run Day 3 code
```

Create the `.gitignore` file with this content (one line):
```
.env
```

### Verify Your Setup Works

Before going further, run this quick sanity check:

📄 **[Run this first → `examples/day_01_genai/verify_setup.py`](examples/day_01_genai/verify_setup.py)**

> **What it does:** Loads your `.env` file, checks that `GEMINI_API_KEY` is present, and verifies that `google-genai` was installed correctly. You'll see ✅ or ❌ for each requirement. Fix every ❌ before moving on.
>
> **Run it:**
> ```bash
> python examples/day_01_genai/verify_setup.py
> ```

---

### 🚨 Troubleshooting Common Setup Issues

Before we continue, here are the top 5 errors beginners hit and how to fix them:

```
ERROR 1: ModuleNotFoundError: No module named 'google.genai'
FIX:     pip install google-genai
         (NOT google-generativeai — that's a different package!)

ERROR 2: GEMINI_API_KEY is None / empty
FIX:     Make sure your .env file is in the SAME folder as your .py file.
         The file should look exactly like:
         GEMINI_API_KEY=AIzaSy...your_key_here
         (no quotes, no spaces around the = sign)

ERROR 3: 429 Resource Exhausted / Rate Limit
FIX:     You've hit the free tier limit. Wait 60 seconds and try again.
         The free tier allows ~15 requests per minute.

ERROR 4: 400 Bad Request / Invalid API Key
FIX:     Your key may have been copy-pasted wrong. Go to
         https://aistudio.google.com/app/apikey and generate a fresh key.

ERROR 5: SyntaxError on f-strings or type hints
FIX:     This tutorial requires Python 3.10+.
         Check your version: python --version
         Upgrade if needed: https://python.org/downloads
```

### 🤝 What To Do If You Get Stuck

```
STEP 1 — Read the full error message.
          The last line usually tells you exactly what went wrong.
          e.g., "FileNotFoundError: .env" → your .env is in wrong folder.

STEP 2 — Check the Troubleshooting section above.
          90% of beginner errors are covered there.

STEP 3 — Add a print() before the broken line.
          print("I reached this point")   ← if you see it, error is after
          print("My variable is:", value) ← see what value actually looks like

STEP 4 — Google the exact error message + "google-genai".
          e.g., "google.api_core.exceptions.ResourceExhausted google-genai"

STEP 5 — Ask for help with the FULL error traceback (copy everything in red).
          Good places: Stack Overflow, Reddit r/learnpython, Google AI Forum.
```

### Your Very First Gen AI Call

This is your "Hello, World!" of Generative AI. The entire program is just 3 meaningful lines:
create a client → call Gemini → read `response.text`. Everything else you write in this course builds on this single pattern.

📄 **[View full example → `examples/day_01_genai/day1_hello_genai.py`](examples/day_01_genai/day1_hello_genai.py)**

> **What this demonstrates:** Creates a `genai.Client`, sends a question to `gemini-2.0-flash`, and prints the response. Pay attention to `response.text` — that's **always** where the AI's reply lives.
>
> **Run it:**
> ```bash
> python examples/day_01_genai/day1_hello_genai.py
> ```


**Expected Output:**
```
A binary search tree (BST) is a tree data structure where each node has
at most two children. For any given node, all values in its left subtree
are smaller, and all values in its right subtree are larger. This property
makes searching, insertion, and deletion operations very efficient with
O(log n) average time complexity.
```

> ⚠️ **Your output will be worded slightly differently** — that's normal! LLMs don't give
> identical output each time. What matters is the meaning is correct.

**Congratulations! 🎉 You just talked to an AI!**

---

## Module 1.3 — Prompt Engineering (45 min)

### The Most Valuable Skill in AI Today

Prompt Engineering is the art of writing instructions to get the BEST output from an AI.

> **Analogy:** Imagine hiring a new intern. If you say *"Do the report,"* you'll get something mediocre. If you say *"Write a 2-page executive report summarizing Q3 sales with bullet points, targeting our CEO audience, in a formal tone,"* you'll get exactly what you need. **The AI is your intern — be specific!**

### The Anatomy of a Good Prompt

```
┌─────────────────────────────────────────────────────────┐
│                    PROMPT STRUCTURE                      │
├─────────────────────────────────────────────────────────┤
│  1. ROLE    → "You are an expert computer science tutor" │
│  2. TASK    → "Explain binary search trees"              │
│  3. FORMAT  → "Use bullet points and include code"       │
│  4. CONTEXT → "My audience is a first-year CS student"   │
│  5. EXAMPLE → "For example, like how you explained AVL"  │ (optional)
└─────────────────────────────────────────────────────────┘
```

### Bad Prompt vs. Good Prompt

Notice the dramatic difference when you give the AI a *role*, a *task*, a *format*, and *context* vs. a one-word request. The AI is not magic — it's a very obedient intern. Garbage in, garbage out.

📄 **[View full example → `examples/day_01_genai/day1_prompt_engineering.py`](examples/day_01_genai/day1_prompt_engineering.py)**

> **What this demonstrates:** Runs the same topic through a vague bad prompt and a structured good prompt side by side, so you can see the quality difference in the output.
>
> **Run it:**
> ```bash
> python examples/day_01_genai/day1_prompt_engineering.py
> ```

### The System Prompt — Giving AI a Personality

A **system prompt** sets the AI's role and behavior for the entire conversation. Think of it as the job description you give to your AI intern at the start of the day.

> 💡 **Key difference:** User prompt = what you ask *this time*. System prompt = the standing rules the AI follows *every* time. The system prompt persists across all turns in a session.

📄 **[View full example → `examples/day_01_genai/day1_system_prompt.py`](examples/day_01_genai/day1_system_prompt.py)**

> **What this demonstrates:** Creates `StudyBuddy` — an AI tutor personality defined by a system prompt that enforces simple language, real-world analogies, and ends every answer with a quiz question.
>
> **Run it:**
> ```bash
> python examples/day_01_genai/day1_system_prompt.py
> ```

### Temperature — Controlling Creativity

> 🌡️ **Think of temperature like a spice dial:** 0.0 is plain rice — safe, predictable, repeatable. 1.5 is ghost-pepper curry — exciting and surprising, but might be incoherent. For study notes, you want 0.2–0.4 (accurate). For brainstorming, try 0.8–1.0.

📄 **[View full example → `examples/day_01_genai/day1_temperature.py`](examples/day_01_genai/day1_temperature.py)**

> **What this demonstrates:** Runs the same prompt at three different temperature values (0.0, 0.5, 1.5) and prints the results so you can compare deterministic vs. creative outputs.
>
> **Run it:**
> ```bash
> python examples/day_01_genai/day1_temperature.py
> ```

---

## Module 1.4 — 🛠️ Build: Study Notes Generator (60 min)

Now let's build the Day 1 version of our **AI Study Buddy** — a simple but useful tool.

**What it does:** Student enters a topic → AI generates structured study notes.

```
┌──────────────┐     prompt      ┌──────────────────┐    structured    ┌──────────────┐
│   Student    │ ──────────────► │   Gemini LLM     │ ───────────────► │  Study Notes │
│              │                 │                  │      notes       │              │
│ "Explain     │                 │ Generates text   │                  │ # Binary     │
│  Binary      │                 │ based on prompt  │                  │ Search Trees │
│  Search Tree"│                 │                  │                  │ ## What is..│
└──────────────┘                 └──────────────────┘                  └──────────────┘

         LIMITATION: Can only answer from training data. 
         No real-time info. No memory between sessions. No tools.
```

📄 **[View full example → `examples/day_01_genai/day1_study_notes_generator.py`](examples/day_01_genai/day1_study_notes_generator.py)**

> **What this builds:** `StudyBuddy v1.0` — a topic → structured study notes generator. Enter a CS topic, get back definition, real-world analogy, steps, Python code example, complexity table, key takeaways, and practice questions.
>
> **Key patterns to notice in the code:**
> - `types.GenerateContentConfig(system_instruction=..., temperature=0.4)` — low temperature for factual, consistent notes
> - `pathlib.Path("study_notes").mkdir(exist_ok=True)` — creates the output folder if it doesn't exist
> - Notes are saved as `.md` files with a timestamp in the filename
>
> **Run it:**
> ```bash
> python examples/day_01_genai/day1_study_notes_generator.py
> ```
>
> **Try changing the topic** from "Binary Search Tree" to "Dynamic Programming" or "Graph Algorithms" and see the output adapt instantly — that's the power of prompt engineering.

### 🧪 Exercise: Extend the Generator

Now that you understand how the study notes generator works, try adding these features on your own. Open `examples/day_01_genai/day1_study_notes_generator.py` and add them as new functions:

**Exercise 1 — Difficulty levels:**
Add a `level` parameter (`"beginner"`, `"intermediate"`, `"advanced"`) to `generate_study_notes()`. Hint: insert the level into the prompt. The AI will automatically calibrate complexity.

**Exercise 2 — Quiz from notes:**
Write a `generate_quiz_from_notes(notes: str, num_questions: int = 5)` function that takes the notes text and calls Gemini to generate MCQ questions from it. Hint: pass the notes as context in the prompt and ask for JSON output.

> 💡 **Why these exercises matter:** Exercise 1 teaches *conditional prompt crafting*. Exercise 2 teaches *chaining* calls — taking AI output as input to the next AI call. Both patterns appear repeatedly in Days 2-5.

---

## Module 1.5 — Limitations of Basic Gen AI (15 min)

This is crucial. Understanding what Gen AI **cannot** do helps you know when to use Day 2 & Day 3 techniques.

```
╔══════════════════════════════════════════════════════════════════╗
║              LIMITATIONS OF BASIC GEN AI                        ║
╠══════════════════════════════════════════════════════════════════╣
║ ❌ No real-time info   ║ "What's the latest AI paper on BSTs?"   ║
║                        ║  → Gives outdated or hallucinated answer ║
╠════════════════════════╬═════════════════════════════════════════╣
║ ❌ No memory           ║ "Remember what I told you yesterday?"   ║
║                        ║  → "I don't have access to previous     ║
║                        ║      conversations."                    ║
╠════════════════════════╬═════════════════════════════════════════╣
║ ❌ No tools            ║ "Calculate the time complexity of my    ║
║                        ║   specific code (500 lines)"            ║
║                        ║  → Can only approximate, can't run code ║
╠════════════════════════╬═════════════════════════════════════════╣
║ ❌ No autonomous action║ "Search for the 3 best YouTube videos   ║
║                        ║  on heaps and summarize them"           ║
║                        ║  → Can't browse the web                 ║
╚════════════════════════╩═════════════════════════════════════════╝

             These limitations → We need AI AGENTS (Day 2)!
```

---

## 📺 Day 1 Further Learning — YouTube Resources

| # | Video | Channel | Duration | What You'll Learn |
|---|-------|---------|----------|-------------------|
| 1 | [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) | Andrej Karpathy | ~60 min | How LLMs work internally |
| 2 | [But what is a GPT?](https://www.youtube.com/watch?v=wjZofJX0v4M) | 3Blue1Brown | ~27 min | Visual explanation of transformers |
| 3 | [Prompt Engineering Tutorial](https://www.youtube.com/watch?v=_ZvnD73m40o) | freeCodeCamp | ~40 min | Master prompt writing |
| 4 | [Google Gemini API Getting Started](https://www.youtube.com/watch?v=o8iyrtQyrZM) | Google Developers | ~20 min | Hands-on Gemini API |

---

## ✅ Day 1 Recap

```
What You Built:  StudyBuddy v1.0 — A topic → study notes generator
What You Learned:
  ✓ What Generative AI is and how LLMs work
  ✓ How to call the Gemini API from Python  
  ✓ Prompt engineering (role, task, format, context)
  ✓ System prompts and temperature
  ✓ Limitations of basic Gen AI → Why we need Agents

Tomorrow: We give StudyBuddy TOOLS and MEMORY — it becomes an Agent!
```

### 🧪 Day 1 Knowledge Check — Answer Before Moving On!

> Try answering these without looking back. If you can't, re-read that section.

1. What is the difference between Traditional AI and Generative AI? Give one example of each.
2. What does "temperature" control in an LLM? What would you set it to for a factual study notes generator — 0.1 or 1.5? Why?
3. What are the 4 parts of a well-structured prompt?
4. What is a system prompt, and how is it different from a user prompt?
5. Name 3 things StudyBuddy v1.0 cannot do that a real tutor can do.

<details>
<summary>💡 Click to reveal answers</summary>

1. Traditional AI classifies/predicts (spam filter = Yes/No). Gen AI creates (write an email).
2. Temperature controls creativity/randomness. Use **0.1–0.4** for study notes — you want accuracy, not wild creativity.
3. Role, Task, Format, Context (+ optional Example).
4. System prompt defines the AI's persistent persona/rules. User prompt is what you type each turn.
5. Examples: can't search the web, can't remember yesterday's session, can't run/test code, can't quiz you interactively, can't give real-time information.

</details>

---
---

# 📅 DAY 2: AI Agents — Giving AI Eyes, Hands & Memory

> ⏱️ **Estimated Time:** 3.5 hours
> 🎯 **Goal:** Understand AI Agents, build one with tools and memory

### 🏗️ Day 2 Architecture: AI Agent (Cyclic Loop)

```
╔═══════════════════════════════════════════════════════════════════════════╗
║               AI AGENT — HOW DATA FLOWS (THE AGENT LOOP)                 ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║   ① USER SENDS A MESSAGE                                                  ║
║   ┌──────────────────────────────────────────┐                           ║
║   │  "My DSA exam is June 20. I need to      │                           ║
║   │   study BSTs. How many days do I have    │                           ║
║   │   and is that enough time?"              │                           ║
║   └──────────────────────┬───────────────────┘                           ║
║                          ▼                                                ║
║  ┌─────────────────────────────────────────────────────────────────────┐ ║
║  │                       AGENT CORE                                    │ ║
║  │                                                                     │ ║
║  │  ┌──────────────────────────┐   ┌──────────────────────────────┐   │ ║
║  │  │     LLM BRAIN (Gemini)   │◄─►│  CONVERSATION MEMORY         │   │   ║
║  │  │                          │   │  [Turn 1]: "My exam is..."   │   │   ║
║  │  │  ② THINKS:               │   │  [Turn 2]: "search result..."│   │   ║
║  │  │  "I need 2 tools:        │   │  [Turn 3]: "difficulty is..."│   │   ║
║  │  │   days_until_exam() and  │   └──────────────────────────────┘   │   ║
║  │  │   get_difficulty()"      │                                      │ ║
║  │  └────────────┬─────────────┘                                      │ ║
║  │               │                                                     │ ║
║  │       ③ CALLS TOOLS                                                 │ ║
║  │               ▼                                                     │ ║
║  │  ┌────────────────────────────────────────────────────────────┐    │ ║
║  │  │                     TOOL DISPATCHER                        │    │ ║
║  │  ├────────────────────┬───────────────────┬───────────────────┤    │ ║
║  │  │ calculate_days()   │ get_difficulty()  │ save_progress()   │    │ ║
║  │  │ → "48 days"        │ → "Medium, 4hrs"  │ → "Saved ✅"      │    │ ║
║  │  └────────────────────┴───────────────────┴───────────────────┘    │ ║
║  │               │ ④ RESULTS RETURNED TO BRAIN                        │ ║
║  │               └────────────────────────────────┐                   │ ║
║  │                                                 ▼                   │ ║
║  │               ⑤ LLM BRAIN COMPOSES FINAL ANSWER                    │ ║
║  └─────────────────────────────────────────────────────────────────────┘ ║
║                          │                                                ║
║                          ▼                                                ║
║              "You have 48 days until your exam.                          ║
║               BSTs take ~4 hours to study. You have                      ║
║               plenty of time — let's make a plan!"                       ║
║                                                                           ║
║  ┌──────────── WHAT THIS ADDS OVER GEN AI ──────────────────────────┐   ║
║  │  ✅  Calls real Python functions (tools) to get accurate data     │   ║
║  │  ✅  Remembers the full conversation within a session             │   ║
║  │  ❌  Still needs YOU to ask for each next step                    │   ║
║  │  ❌  Only ONE AI brain — can't parallelize complex work           │   ║
║  └───────────────────────────────────────────────────────────────────┘   ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## Module 2.1 — What is an AI Agent? (45 min)

### The Leap from Gen AI to Agents

On Day 1, our StudyBuddy was like a **very smart encyclopedia** — you ask, it answers, but it can't do anything beyond that.

An **AI Agent** is like giving that encyclopedia a **phone, a computer, and a calendar**. Now it can:
- 🔍 Search the internet
- 🧮 Run Python code
- 📅 Schedule things
- 💾 Remember past conversations
- 🔄 Take actions and observe results

### The Agent Loop: Perceive → Think → Act → Observe

```
┌─────────────────────────────────────────────────────────────┐
│                    THE AGENT LOOP                           │
│                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│   │ PERCEIVE │───►│  THINK   │───►│   ACT    │            │
│   │          │    │          │    │          │            │
│   │ Read the │    │ LLM plans│    │ tool or  │            │
│   │ task &   │    │ what to  │    │ take     │            │
│   │ context  │    │ do next  │    │ action   │            │
│   └──────────┘    └──────────┘    └──────────┘            │
│        ▲                          └────┬─────┘            │
│        │                               │                  │
│        │         ┌──────────┐          │                  │
│        └─────────│ OBSERVE  │◄─────────┘                  │
│                  │          │                              │
│                  │ Get the  │                              │
│                  │ result & │                              │
│                  │ loop back│                              │
│                  └──────────┘                              │
│                                                             │
│   This loop continues until the GOAL is achieved!          │
└─────────────────────────────────────────────────────────────┘
```

**Concrete Example:**

> **User:** "Find resources for studying BSTs, then make me a study plan for this week."

```
PERCEIVE: Student wants BST resources + a weekly study plan
   │
   ▼
THINK: "I need to: 1) Search for BST resources 2) Get today's date 3) Create plan"
   │
   ▼
ACT: Call search_web("best BST tutorial resources 2024")
   │
   ▼
OBSERVE: Got 5 tutorial links from search results
   │
   ▼
THINK: "Now I have resources. Let me get today's date to plan the week."
   │
   ▼
ACT: Call get_current_date()
   │
   ▼
OBSERVE: "Today is Monday, May 5, 2026"
   │
   ▼
THINK: "I have everything. Now I can create the study plan."
   │
   ▼
ACT: Generate study plan combining resources + dates
   │
   ▼
OBSERVE: Plan is complete. Goal achieved! ✅
   │
   ▼
RESPOND: Give the full plan to the student
```

### The ReAct Pattern (Reason + Act)

The most popular pattern for AI Agents is **ReAct** — the model explicitly **Reasons** about what to do, then **Acts** on it.

```
Thought: I need to find BST resources. Let me search.
Action: search_web("binary search tree tutorials")
Observation: [Wikipedia, GeeksforGeeks, YouTube results...]

Thought: I found resources. Now I need today's date.
Action: get_current_date()
Observation: May 5, 2026

Thought: I have resources and date. I can now create the study plan.
Action: None (generate final answer)
Final Answer: Here is your personalized BST study plan...
```

### 🎬 Watch This (Before Coding)

> **"What are AI Agents?"** by IBM Technology
> 🔗 https://www.youtube.com/watch?v=F8NKVhkZZWI
> *(Clear, visual explanation of what agents are — ~13 min)*

---

## Module 2.2 — Tools: the Agent's Superpowers (45 min)

### What Are Tools?

Tools are **Python functions** that the AI can call to interact with the world.

```
┌───────────────────────────────────────────────────────────┐
│                    TOOLS THE AI CAN USE                    │
├─────────────────┬───────────────────────────────────────── │
│   Tool Name     │   What It Does                           │
├─────────────────┼─────────────────────────────────────────┤
│ search_web()    │ Searches Google/DuckDuckGo               │
│ read_file()     │ Reads a file from your computer          │
│ run_python()    │ Executes Python code, returns result     │
│ get_date()      │ Gets current date/time                   │
│ send_email()    │ Sends an email                           │
│ save_notes()    │ Saves notes to a file                    │
│ get_weather()   │ Gets weather for a city                  │
│ Your own tools! │ Anything you can write in Python!        │
└─────────────────┴─────────────────────────────────────────┘
```

> **Key Insight:** The AI doesn't *run* the tools — **your code does**. The AI just *decides* which tool to call and with what arguments. Your Python code executes it and returns the result.

### 📦 Quick Python Concept: `@dataclass`

Before the code below, you'll see `@dataclass`. If you haven't seen it before — no worries! Here's what it does in 30 seconds:

```python
# WITHOUT @dataclass (lots of boilerplate):
class Student:
    def __init__(self, name, exam_date, score):
        self.name = name
        self.exam_date = exam_date
        self.score = score

# WITH @dataclass (Python writes __init__ for you!):
from dataclasses import dataclass, field

@dataclass
class Student:
    name: str           # required field
    exam_date: str      # required field
    score: int = 0      # optional field with default value
    topics: list = field(default_factory=list)  # mutable default (use field() for lists/dicts!)

# Both work the same way:
s = Student(name="Alex", exam_date="2026-06-15")
print(s.name)   # "Alex"
print(s.score)  # 0
```

> `field(default_factory=list)` is needed for lists/dicts because Python doesn't allow mutable defaults directly. Think of it as saying: "create a fresh empty list for each new object." You'll see this pattern a lot in our agent code.

### How Gemini Returns Tool Call Requests

Before reading the function-calling code, understand this diagram — it explains WHY the code looks the way it does:

```
STEP 1: You send a message + list of available tools
  Your code  ──────────►  Gemini API

STEP 2: Gemini says "I want to call a tool" (NOT text yet!)
  Gemini API ──────────►  response.candidates[0].content.parts
                           │
                           └─► Part contains a function_call object:
                               {
                                 name: "get_current_date",
                                 args: {}
                               }
                           (This is NOT a text answer — it's a tool REQUEST)

STEP 3: YOUR code runs the actual Python function
  Your code  ──────────►  get_current_date()  →  "Monday, May 5, 2026"

STEP 4: You send the result BACK to Gemini
  Your code  ──────────►  Gemini API  (with tool result attached)

STEP 5: NOW Gemini gives a text answer using the tool result
  Gemini API ──────────►  "Today is Monday, May 5, 2026. Your exam..."

WHY "candidates[0].content.parts"?
  - candidates → Gemini can generate multiple candidate responses (we use [0])
  - content     → the full content of that response
  - parts       → a response can have MULTIPLE parts (text part, tool-call part, etc.)
                  We loop through parts to find tool call requests.
```

### Function Calling with Gemini

📄 **[View full example → `examples/day_02_agents/day2_function_calling_basics.py`](examples/day_02_agents/day2_function_calling_basics.py)**

> **What this demonstrates:** The complete 5-step function calling cycle:
> 1. Define real Python functions (`get_current_date`, `calculate_days_until_exam`, `get_cs_topic_difficulty`)
> 2. Describe them to the AI using `types.FunctionDeclaration` (like a job posting for each function)
> 3. The AI reads a question, *decides* which tools to call, and returns a `function_call` object — not text
> 4. Your Python code executes the actual function and collects the result
> 5. You send the result back and Gemini produces the final text answer
>
> **Key insight:** On the last test query — *"My DSA exam is on June 20. I need to study BSTs. How many days do I have and is that enough time?"* — watch how the agent calls **two tools in sequence** without you explicitly asking.
>
> **Run it:**
 > ```bash
> python examples/day_02_agents/day2_function_calling_basics.py
> ```

---

## Module 2.3 — Memory: Giving the Agent a Notebook (30 min)

By default, every time you call the AI, it forgets everything. To add memory, we pass the **conversation history** in every request.

```
WITHOUT MEMORY:                   WITH MEMORY:
──────────────                    ────────────
You: "My exam is June 20"         You: "My exam is June 20"
AI: "OK! What do you need?"       AI: "OK! What do you need?"
                                  [History stored: exam = June 20]
You: "How long do I have?"        You: "How long do I have?"
AI: "I don't know your exam       AI: "Based on your exam on
     date."  ❌                        June 20, you have 46 days!" ✅
```

📄 **[View full example → `examples/day_02_agents/day2_memory.py`](examples/day_02_agents/day2_memory.py)**

> **What this demonstrates:** `StudyBuddyWithMemory` — a `@dataclass` that holds a `history` list. Every `.chat()` call appends to that list and passes the **entire** history to Gemini. By the 4th and 5th messages the AI correctly recalls your name and exam date — proving that memory is just a list.
>
> **Key pattern:** `contents=self.history` — if you forget this and pass just the latest message, the AI forgets everything from previous turns.
>
> **Run it:**
 > ```bash
> python examples/day_02_agents/day2_memory.py
> ```

---

## Module 2.4 — 🛠️ Build: Research & Quiz Agent (60 min)

Now let's upgrade StudyBuddy to v2.0 — a full **Research & Quiz Agent**.

**What it does:**
1. Searches for explanations of topics
2. Creates personalized flashcard quizzes
3. Checks your answers
4. Remembers your progress within the session

```
┌──────────────────────────────────────────────────────────────────┐
│                  STUDYBUDDY v2.0 — AGENT ARCHITECTURE            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Student Input                                                  │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                         AGENT                            │   │
│  │   ┌─────────────────┐      ┌────────────────────────┐   │   │
│  │   │   LLM Brain     │      │       Memory           │   │   │
│  │   │  (Gemini)       │◄────►│  Conversation History  │   │   │
│  │   │                 │      │  + Student Profile     │   │   │
│  │   └────────┬────────┘      └────────────────────────┘   │   │
│  │            │                                                     │
│  │       ③ CALLS TOOLS                                                 │
│  │               ▼                                                     │
│  │  ┌────────────────────────────────────────────────────────────┐    │ ║
│  │  │                     TOOL DISPATCHER                        │    │ ║
│  │  ├────────────────────┬───────────────────┬───────────────────┤    │ ║
│  │  │ calculate_days()   │ get_difficulty()  │ save_progress()   │    │ ║
│  │  │ → "48 days"        │ → "Medium, 4hrs"  │ → "Saved ✅"      │    │ ║
│  │  └────────────────────┴───────────────────┴───────────────────┘    │ ║
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                  │
│       LIMITATION: Still requires a human to drive each step.    │
│       "What should I study next?" → Human must ask.             │
│       This is what we fix on Day 3! 🚀                          │
└──────────────────────────────────────────────────────────────────┘
```

📄 **[View full example → `examples/day_02_agents/day2_research_quiz_agent.py`](examples/day_02_agents/day2_research_quiz_agent.py)**

> **What this builds:** `StudyBuddy v2.0` — a full Research & Quiz Agent. It simulates searching for topics, generates MCQ quizzes via Gemini (JSON mode), checks your answers, tracks your score, and remembers everything within the session.
>
> **Key patterns to notice:**
> - `StudentProfile` `@dataclass` as shared state between agent steps
> - `response_mime_type="application/json"` inside `GenerateContentConfig(...)` constructor — forces Gemini to produce raw JSON (no markdown fences)
> - `json.loads(response.text)` to parse the structured output
>
> **Run it:**
> ```bash
> python examples/day_02_agents/day2_research_quiz_agent.py
> ```

---

### 🧩 Connecting Day 2 to Day 3

**StudyBuddy v2.0 has a fundamental bottleneck:** the human must drive every step. Want the next quiz? Ask. Want a study plan? Ask. Want a progress report? Ask.

Day 3 fixes this with **Agentic AI** — an orchestrator that runs all 5 agents *automatically* based on a single high-level goal.

# 📅 DAY 3: Agentic AI — Autonomous Multi-Agent Systems

> ⏱️ **Estimated Time:** 4 hours
> 🎯 **Goal:** Understand what makes a system "agentic", build a 5-agent autonomous exam coach

The leap from Day 2 to Day 3 is the most important in the whole course:

```
Day 2 Agent:     Human asks → AI answers with tool help → Human asks again → ...
Day 3 Agentic:   Human gives ONE goal → System plans, researches, quizzes, grades,
                  adapts, and reports — all autonomously — until the exam!
```

> **Analogy:** Day 2 is like a very smart assistant who waits for your next question. Day 3 is like hiring a full project manager who runs the entire operation and only bothers you when truly needed.

## Module 3.1 — What Makes AI "Agentic"? (30 min)

**Agentic AI** has three properties that agents (Day 2) don't: **autonomy**, **persistence**, and **self-direction**.

| Property | AI Agent (Day 2) | Agentic AI (Day 3) |
|----------|-----------------|-------------------|
| Who decides what to do next? | Human | The orchestrator |
| How long does it run? | One task at a time | Multiple sessions over days |
| Can it adapt its own plan? | No | Yes — based on performance data |
| How many AI brains? | One | Five specialized agents |

## Module 3.2 — The Orchestrator Pattern (30 min)

An **orchestrator** is the agent-of-agents — it decides which specialized sub-agent to call, when, and with what state.

```
ORCHESTRATOR (AutonomousExamCoach)
    │
    ├─► PlannerAgent   — "create a 7-day study schedule"
    ├─► ResearchAgent  — "fetch content for today's topic"
    ├─► QuizAgent      — "generate 5 questions on weak areas"
    ├─► EvaluatorAgent — "grade the answers, find weak spots"
    └─► ReportAgent    — "write today's progress report"

All share one StudentProfile object as the single source of truth.
```

📄 **[View orchestration basics → `examples/day_03_agentic_ai/day3_orchestration_basics.py`](examples/day_03_agentic_ai/day3_orchestration_basics.py)**

> **What this shows:** A stripped-down orchestration loop — no Gemini calls yet. Just the pattern of how an orchestrator creates sub-agents, passes shared state between them, and sequences their execution.
>
> **Run it:**
> ```bash
> python examples/day_03_agentic_ai/day3_orchestration_basics.py
> ```

## Module 3.3 — 🛠️ Build: Autonomous Exam Coach (90 min)

The full `StudyBuddy v3.0` — 5 specialized agents coordinated by an autonomous orchestrator.

📄 **[View full example → `examples/day_03_agentic_ai/day3_autonomous_exam_coach.py`](examples/day_03_agentic_ai/day3_autonomous_exam_coach.py)**

> **What this builds:** `AutonomousExamCoach` — give it a student's name, exam date, and topics. It then:
> 1. Creates a personalized study schedule (PlannerAgent)
> 2. Researches today's topic (ResearchAgent)
> 3. Generates a targeted quiz — biased toward weak areas (QuizAgent)
> 4. Grades the answers and updates the student profile (EvaluatorAgent)
> 5. Writes a progress report (ReportAgent)
> 6. Repeats for `num_sessions` days, adapting as it goes — all without you asking
>
> **Key patterns:**
> - `StudentProfile` dataclass as the single shared state object — all agents read and write to it
> - `response_mime_type="application/json"` in `GenerateContentConfig(...)` for structured agent outputs
> - `time.sleep(10)` between sessions to stay within Gemini free-tier rate limits
> - `max_iterations` guard in every agent loop to prevent infinite execution
>
> **Run it (simulation mode — no user input needed):**
> ```bash
> python examples/day_03_agentic_ai/day3_autonomous_exam_coach.py
> ```
>
> ⚠️ This makes ~15 API calls per session. With 3 sessions it will take 2-3 minutes. Normal!

---

## Module 3.4 — Comparing All Three Versions (30 min)

Let's see the exact same task handled by all three versions:

**Task:** *"I have a BST exam in 7 days. Help me prepare."*

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SAME TASK — THREE DIFFERENT APPROACHES                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  📘 GEN AI (Day 1)                                                           │
│  ──────────────────                                                          │
│  Student: "I have a BST exam in 7 days. Help me prepare."                   │
│  AI: [writes a static 7-day study plan]                                      │
│  Done. ← One turn. One response.                                             │
│                                                                              │
│  Problems:                                                                   │
│  ❌ Can't search for real resources                                           │
│  ❌ Can't quiz you or check your answers                                      │
│  ❌ Can't adapt based on your weak spots                                      │
│  ❌ Can't run for 7 days autonomously                                         │
│                                                                              │
│  🤖 AI AGENT (Day 2)                                                         │
│  ───────────────────                                                         │
│  Student: "I have a BST exam in 7 days. Help me prepare."                   │
│  Agent: [searches for BST content, asks what to start with]                 │
│                                                                              │
│  Student: "Start with insertion"                                             │
│  Agent: [explains insertion, generates a quiz question]                      │
│                                                                              │
│  Student: "Quiz me"                                                          │
│  Agent: [generates quiz, waits for answer, evaluates]                        │
│                                                                              │
│  Student: "What should I focus on tomorrow?"                                 │
│  Agent: [based on your answer, suggests next topic]                          │
│                                                                              │
│  Better, but:                                                                │
│  ✅ Can use tools (search, quiz)                                              │
│  ✅ Has memory within session                                                 │
│  ❌ Still needs student to drive every step                                   │
│  ❌ Only one agent — can't parallelize complex work                           │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  🚀 AGENTIC AI (Day 3)                                                       │
│  ─────────────────────                                                       │
│  Student: "I have a BST exam in 7 days. Here's my schedule. Go."            │
│                                                                              │
│  [Orchestrator activates — no more student input needed]                     │
│                                                                              │
│  Day 1 AM (while student sleeps):                                            │
│    PlannerAgent creates 7-day schedule ✅                                     │
│    ResearchAgent pulls all BST content ✅                                     │
│    QuizAgent prepares Day 1 quiz ✅                                           │
│                                                                              │
│  Day 1 PM (student has 15 min):                                              │
│    Student takes auto-generated quiz                                          │
│    EvaluatorAgent instantly grades + spots: "weak on deletion"               │
│    ReportAgent sends summary to student                                       │
│                                                                              │
│  Day 2 AM (without being asked):                                             │
│    Orchestrator sees "weak: deletion" → re-plans Day 2 to focus there       │
│    ResearchAgent pulls extra content on BST deletion                         │
│    QuizAgent generates 5 deletion-focused questions                          │
│    ...and so on for all 7 days                                               │
│                                                                              │
│  ✅ Fully autonomous — student just inputs the goal once                    │
│  ✅ Multiple agents work in parallel                                          │
│  ✅ Self-adapts based on performance                                          │
│  ✅ Generates reports without being asked                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 3.5 — Real-World Agentic AI & Best Practices (30 min)

### Where Agentic AI Is Used Today

| Industry | Agentic AI Application | Agents Involved |
|----------|------------------------|-----------------|
| **Software Dev** | GitHub Copilot Workspace | Planner + Coder + Tester + PR Reviewer |
| **Finance** | Automated trading analysis | Data Fetcher + Analyzer + Risk Assessor + Reporter |
| **Healthcare** | Patient data processing | Records Agent + Drug Interaction Checker + Scheduler |
| **Customer Support** | Autonomous ticket resolution | Classifier + Knowledge Base + Responder + Escalator |
| **Research** | Literature review automation | Searcher + Summarizer + Fact-checker + Writer |

### Best Practices for Building Agentic Systems

```
╔══════════════════════════════════════════════════════════════╗
║              AGENTIC AI BEST PRACTICES                       ║
╠══════════════════════════════════════════════════════════════╣
║  1. HUMAN-IN-THE-LOOP                                        ║
║     Don't give agents unlimited autonomy on day 1           ║
║     Add approval gates for high-stakes actions              ║
║     Example: "Agent wants to send email — approve? Y/N"     ║
╠══════════════════════════════════════════════════════════════╣
║  2. CLEAR AGENT BOUNDARIES                                   ║
║     Each agent should do ONE thing well                     ║
║     Don't make agents do too many different tasks           ║
║     Single Responsibility Principle applies to agents too!  ║
╠══════════════════════════════════════════════════════════════╣
║  3. SHARED STATE CAREFULLY                                   ║
║     Use a shared data store (like StudentProfile)           ║
║     Make it clear what each agent can read vs. write        ║
║     Avoid agents overwriting each other's work              ║
╠══════════════════════════════════════════════════════════════╣
║  4. ITERATION LIMITS                                         ║
║     Always set max_iterations on agent loops                ║
║     Agents can get stuck in infinite loops                  ║
║     Example: max_attempts = 3 before giving up              ║
╠══════════════════════════════════════════════════════════════╣
║  5. LOGGING EVERYTHING                                       ║
║     Log every agent action and tool call                    ║
║     Makes debugging MUCH easier                             ║
║     Helps you understand what the system decided and why    ║
╠══════════════════════════════════════════════════════════════╣
║  6. GRACEFUL FAILURE                                         ║
║     What if an agent fails? Have fallback behavior          ║
║     Don't let one agent crash the whole system              ║
║     Example: try/except around every agent call             ║
╚══════════════════════════════════════════════════════════════╝
```

---

# 📌 Quick Reference Card

## 🗺️ The Complete Picture — All 4 Concepts Together

```
╔══════════════════════════════════════════════════════════════════════════════╗
║          STUDYBUDDY COMPLETE ARCHITECTURE (v3.5 — Full Stack)                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   STUDENT: "Prepare me for my DSA exam"  ← One-time goal                   ║
║                         │                                                    ║
║                         ▼                                                    ║
║   ┌─────────────────────────────────────────────────────────────────────┐   ║
║   │                  ORCHESTRATOR (Agentic AI — Day 3)                  │   ║
║   │  Autonomous goal manager. Delegates without being asked.            │   ║
║   └──────┬──────────────┬──────────────┬──────────────┬─────────────────┘   ║
║          │              │              │              │                       ║
║          ▼              ▼              ▼              ▼                       ║
║   ┌──────────┐  ┌────────────────┐  ┌──────────┐  ┌──────────┐            ║
║   │ PLANNER  │  │  RESEARCHER    │  │  QUIZ    │  │EVALUATOR │            ║
║   │  AGENT   │  │    AGENT       │  │  AGENT   │  │  AGENT   │            ║
║   │(Day 3)   │  │  (Day 3+Bonus) │  │(Day 3)   │  │(Day 3)   │            ║
║   └──────────┘  └──────┬─────────┘  └──────────┘  └──────────┘            ║
║                         │ queries                                            ║
║                         ▼                                                    ║
║              ┌──────────────────────────────────────┐                       ║
║              │  CHROMADB  (Bonus — VectorDB & RAG)  │                       ║
║              │  Your docs → vectors → fast retrieval│                       ║
║              └──────────────────┬───────────────────┘                       ║
║                                 │ top-3 relevant chunks                      ║
║                                 ▼                                            ║
║              ┌──────────────────────────────────────┐                       ║
║              │  RAG PROMPT:  context + question      │                       ║
║              └──────────────────┬───────────────────┘                       ║
║                                 │                                            ║
║                                 ▼                                            ║
║              ┌──────────────────────────────────────┐                       ║
║              │  GEMINI LLM  (Gen AI — Day 1)        │                       ║
║              │  Generates grounded, accurate answer  │                       ║
║              └──────────────────────────────────────┘                       ║
║                                                                              ║
║  Layer by layer:                                                             ║
║  Gen AI (Day 1) → +Tools+Memory = Agent (Day 2)                            ║
║  Agent (Day 2)  → +Multi-Agent+Autonomy = Agentic AI (Day 3)               ║
║  Agentic AI     → +VectorDB+RAG = Full Knowledge System (Bonus)            ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

# 🎓 Free Certifications to Add to Your Resume

These are 100% free to complete and respected by employers!

## Tier 1 — Essential (Complete These First)

| Certification | Provider | Time | Link |
|--------------|----------|------|------|
| **Generative AI Fundamentals** | Google Cloud Skills Boost | ~8 hours | https://www.cloudskillsboost.google/paths/118 |
| **Intro to Generative AI** Badge | Google Cloud | ~45 min | https://www.cloudskillsboost.google/course_templates/536 |
| **AI For Everyone** (audit free) | DeepLearning.AI / Coursera | ~6 hours | https://www.coursera.org/learn/ai-for-everyone |

## Tier 2 — Technical (For CS Students)

| Certification | Provider | Time | Link |
|--------------|----------|------|------|
| **Hugging Face NLP Course** | Hugging Face | ~35 hours | https://huggingface.co/learn/nlp-course |
| **Prompt Engineering for LLMs** | DeepLearning.AI | ~1 hour | https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/ |
| **LangChain for LLM Applications** | DeepLearning.AI | ~1 hour | https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/ |
| **AI Agents in LangGraph** | DeepLearning.AI | ~1.5 hours | https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/ |
| **Multi-Agent Systems** | DeepLearning.AI | ~1.5 hours | https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/ |

## Tier 3 — Industry/Enterprise

| Certification | Provider | Time | Cost | Link |
|--------------|----------|------|------|------|
| **Azure AI Fundamentals (AI-900)** | Microsoft | ~10 hours | Free to study, ~$165 to certify | https://learn.microsoft.com/en-us/credentials/certifications/azure-ai-fundamentals/ |
| **NVIDIA Generative AI Explained** | NVIDIA DLI | ~4 hours | Free | https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-FX-07+V1 |
| **IBM AI Developer Professional** | IBM / Coursera | ~4 months | Free to audit | https://www.coursera.org/professional-certificates/applied-artifical-intelligence-ibm-watson-ai |

## 💡 Resume Tips

```
✅ Add certifications to your LinkedIn "Licenses & Certifications" section
✅ Mention the specific skills (Prompt Engineering, Gemini API, Multi-Agent)
✅ Include your StudyBuddy project on GitHub — it shows practical experience!
✅ DeepLearning.AI badges are especially recognized at tech companies
✅ Google Cloud badges include a shareable URL — perfect for resumes
```

---

# 🔗 Complete Resource Index

## Core Tools & Libraries
- **Google Gemini API:** https://ai.google.dev
- **Google AI Studio (free API key):** https://aistudio.google.com
- **LangChain:** https://www.langchain.com
- **LangGraph:** https://www.langchain.com/langgraph
- **CrewAI:** https://www.crewai.com
- **AutoGen (Microsoft):** https://microsoft.github.io/autogen/

## Learn More
- **DeepLearning.AI Short Courses:** https://www.deeplearning.ai/short-courses/
- **Google Generative AI Learning Path:** https://www.cloudskillsboost.google/paths/118
- **Hugging Face Courses:** https://huggingface.co/learn
- **Fast.ai (Practical DL):** https://www.fast.ai

## Stay Updated
- **AI News:** https://www.aiweekly.co
- **Papers With Code:** https://paperswithcode.com
- **Hugging Face Blog:** https://huggingface.co/blog

---

---

# 🔮 Upcoming: LangChain & LangGraph — Should We Use Them?

> 📌 **This is a "future improvement" note for instructors/students.** No code changes are made to the tutorial yet — but here's the reasoning for when you're ready to evolve beyond Day 3.

## Why We Didn't Use LangChain/LangGraph in This Tutorial

In this course, we deliberately built everything from scratch using the raw Gemini SDK. Here's why — and why you might want frameworks later:

| Aspect | Built from Scratch (this tutorial) | LangChain / LangGraph |
|--------|------------------------------------|-----------------------|
| **Learning value** | ⭐⭐⭐ You understand every line | ⭐ Framework hides complexity |
| **Code length** | More lines, more explicit | Much shorter |
| **Debugging** | Easy — you wrote it | Harder — layers of abstraction |
| **Flexibility** | Full control | Framework constraints |
| **Production speed** | Slower to write | Fast to scaffold |
| **Community tools** | Build your own | 500+ pre-built integrations |

## When Would LangGraph Help? (Next Course Iteration)

**LangGraph** would specifically improve our Day 3 Agentic AI system:

```
Current (Day 3 — manual orchestration):
  OrchestratorAgent.run_autonomous_session() → calls each agent manually
  → Hard to visualize the flow
  → Hard to add conditional branches (e.g., "if score < 50%, trigger re-research")
  → Hard to add parallel execution properly

With LangGraph:
  Define a STATE GRAPH where nodes = agents and edges = conditions
  → Visual graph you can render and debug
  → Built-in parallel execution (fan-out/fan-in nodes)
  → Human-in-the-Loop checkpoints built in
  → Persistent state across sessions automatically
  → Easy to add new agents without rewriting the whole system

Example (conceptual):
  graph = StateGraph(StudentProfile)
  graph.add_node("planner",    planner_agent)
  graph.add_node("researcher", researcher_agent)
  graph.add_node("quiz",       quiz_agent)
  graph.add_node("evaluator",  evaluator_agent)
  graph.add_edge("planner",    "researcher")
  graph.add_conditional_edges("evaluator",
      condition=needs_replan,      # if accuracy < 60%
      then="planner",              # re-plan
      else_="quiz"                 # next quiz session
  )
```

**Recommended order for next steps:**
1. ✅ Master this tutorial (scratch-built — understand internals)  ← you are here
2. 🔜 Take DeepLearning.AI "AI Agents in LangGraph" (free, 1.5 hrs)
3. 🔜 Rebuild Day 3 using LangGraph — compare the experience
4. 🔜 Add CrewAI for role-based agent personas

---

---

# 🗄️ Day 4: VectorDB & RAG — Making AI Know YOUR Knowledge

> ⏱️ **Estimated Time:** 2.5 hours
> 🎯 **Goal:** Understand why VectorDB exists, what RAG is, and build a local demo with ChromaDB

**Run code from:** `examples/day_04_bonus_rag/bonus_vectordb_rag_demo.py`

**Further study links:** `resources/references/day_04_rag_chroma.md`

---

## Where Does RAG Fit In Our Story?

We've built StudyBuddy through 3 versions. There's one remaining problem:

```
StudyBuddy v1.0 (Gen AI)
  Knowledge source: Gemini's training data (general internet, cut off in 2024)
  Problem: Can't answer questions about YOUR notes, YOUR textbook, YOUR professor's slides

StudyBuddy v2.0 (AI Agent)
  Knowledge source: Same training data + hardcoded CS_KNOWLEDGE_BASE dict
  Problem: Dict has only 3 topics. Can't scale. Can't add new topics without code changes.

StudyBuddy v3.0 (Agentic AI)
  Knowledge source: Same hardcoded dict + AI training data
  Problem: ResearchAgent still uses a fixed Python dict. What if we have 500 topics?

StudyBuddy v3.5 (Agentic AI + RAG) ← This bonus module
  Knowledge source: YOUR documents → stored in ChromaDB → retrieved on demand
  ✅ Add any number of topics (PDF, text, markdown) without code changes
  ✅ Semantic search finds relevant knowledge even with different wording
  ✅ Answers are grounded in YOUR material — no hallucination
  ✅ ResearchAgent now QUERIES ChromaDB instead of looking up a Python dict
```

---

## What Problem Does VectorDB Solve?

Remember Day 1's limitation: *"Gen AI only knows what it was trained on."*

What if you want the AI to answer questions about YOUR specific data — your company docs, your codebase, your textbooks?

You have 3 options:

```
OPTION 1: Put everything in the prompt (Naive)
──────────────────────────────────────────────
Prompt = system_prompt + ALL 500 pages of textbook + your question
Problem: Context window limit (~100k tokens). Expensive. Slow.
✅ Works for tiny docs.  ❌ Fails for large knowledge bases.

OPTION 2: Fine-tune the model (Expensive)
──────────────────────────────────────────
Retrain the model with your data baked in.
Problem: Costs $10,000–$100,000. Takes days. Need ML expertise.
✅ Great for domain-specific models.  ❌ Way overkill for most cases.

OPTION 3: RAG — Retrieval Augmented Generation (The Smart Way) ✅
──────────────────────────────────────────────────────────────────
1. Store your docs in a VectorDB
2. When user asks a question → find the 3 most RELEVANT doc chunks
3. Put ONLY those chunks in the prompt + the question
4. AI answers using fresh, specific context

Result: Accurate, current, domain-specific answers — cheaply!
```

## What is a Vector / Embedding?

This is the magic that makes semantic search work.

```
TRADITIONAL SEARCH (keyword): finds exact words
  Query: "how fast is BST search?"
  Result: Only finds docs containing those exact words

VECTOR SEARCH (semantic): finds meaning
  Query: "what is BST time complexity?"
  Result: Finds docs about "O(log n) performance" even without those words!

HOW?
  1. Every piece of text gets converted to a list of numbers (a "vector")
     e.g., "Binary Search Tree" → [0.12, -0.45, 0.78, ..., 0.33]  (768 numbers)
     
  2. Similar MEANINGS get similar vectors (close together in space)
     "car" and "automobile" → very similar vectors
     "car" and "potato" → very different vectors

  3. A VectorDB stores these vectors and finds nearest neighbours fast

ANALOGY:
  Think of every sentence as a GPS coordinate on a map.
  "BST search efficiency" and "tree lookup speed" end up on the same street.
  "BST search" and "pasta recipe" are on different continents.
  VectorDB = the GPS that finds your nearest neighbours.
```

## RAG Architecture with VectorDB

```
╔══════════════════════════════════════════════════════════════════════════╗
║              RAG (RETRIEVAL AUGMENTED GENERATION) FLOW                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  SETUP PHASE (done once — runs offline)                                  ║
║  ─────────────────────────────────────                                   ║
║  Your Documents/Notes                                                    ║
║  ┌──────────────────┐   ① Split    ┌────────────────────┐              ║
║  │  textbook.pdf     │  into chunks │  Chunk 1: "BST is" │              ║
║  │  lecture_notes.md │ ──────────► │  Chunk 2: "Insert" │              ║
║  │  my_notes.txt     │             │  Chunk 3: "O(log n)"│              ║
║  └──────────────────┘             └────────┬───────────┘               ║
║                                            │ ② Convert to vectors       ║
║                                            ▼  (embedding model)         ║
║                                   ┌────────────────────┐               ║
║                                   │   CHROMADB (local) │               ║
║                                   │   [0.12, -0.45...]  │               ║
║                                   │   [0.88, 0.23...]   │               ║
║                                   │   [-0.11, 0.67...]  │               ║
║                                   └────────────────────┘               ║
║                                                                          ║
║  QUERY PHASE (every user question)                                       ║
║  ──────────────────────────────────                                      ║
║  ③ User asks: "What is BST time complexity?"                            ║
║                       │                                                  ║
║                       ▼ (convert to vector too)                          ║
║  ④ VectorDB finds TOP 3 most similar chunks                             ║
║       Chunk 3: "O(log n) average case for BST..."         similarity=0.94║
║       Chunk 1: "BST is a tree where left < parent..."     similarity=0.82║
║       Chunk 7: "Balanced trees maintain O(log n)..."      similarity=0.79║
║                       │                                                  ║
║  ⑤ INJECT into prompt:                                                  ║
║  ┌────────────────────────────────────────────────────────────────┐     ║
║  │ CONTEXT (from VectorDB):                                        │     ║
║  │ "O(log n) average case for BST..."                              │     ║
║  │ "BST is a tree where left < parent..."                          │     ║
║  │                                                                  │     ║
║  │ QUESTION: "What is BST time complexity?"                        │     ║
║  └────────────────────────────────────────────────────────────────┘     ║
║                       │                                                  ║
║                       ▼                                                  ║
║  ⑥ Gemini uses ONLY the relevant context to answer accurately           ║
║     "Based on the study materials: BST search is O(log n)..."           ║
╚══════════════════════════════════════════════════════════════════════════╝
```

## 🛠️ Build: StudyBuddy RAG Demo with ChromaDB

ChromaDB is a free, open-source vector database that runs **entirely on your laptop** — no cloud account needed.

> 📦 **Install dependencies first:**
> ```bash
> pip install chromadb google-genai python-dotenv
> ```
> - `chromadb` — free, local, open-source vector database (runs on your laptop)
> - `google-genai` — Gemini API SDK
> - `python-dotenv` — loads your `.env` API key

📄 **[View full example → `examples/day_04_bonus_rag/bonus_vectordb_rag_demo.py`](examples/day_04_bonus_rag/bonus_vectordb_rag_demo.py)**

> **What this builds:** `StudyBuddy v3.5` — ChromaDB-powered RAG pipeline. Seeds a local
> vector database with CS study notes, then answers questions by:
> 1. **Retrieve** — semantic search in ChromaDB for the most relevant chunks
> 2. **Augment** — build a context-rich prompt from those chunks
> 3. **Generate** — Gemini answers grounded ONLY in your notes
>
> **Key insight to look for:** Run Demo 2 and compare "without RAG" vs "with RAG" answers.
> Gemini without context makes up a plausible but generic answer;
> RAG Gemini gives the exact answer from your notes — that's the power of grounding.
>
> **Run it:**
> ```bash
> python examples/day_04_bonus_rag/bonus_vectordb_rag_demo.py
> ```

---

## How RAG Enhances StudyBuddy v3.0

The current Day 3 system uses `CS_KNOWLEDGE_BASE = {}` — a hardcoded Python dictionary. Replacing it with ChromaDB RAG gives you:

```
BEFORE (hardcoded dict):                AFTER (ChromaDB + RAG):
────────────────────────                ─────────────────────────────
CS_KNOWLEDGE_BASE = {                   → Store ANY amount of text
  "bst": {"summary": "...",             → Semantic search, not key lookup
          "concepts": [...]}            → Add new material without code changes
}                                       → Retrieve only what's relevant
                                        → Grounded, citation-able answers
ResearchAgent looks up by exact key     ResearchAgent queries VectorDB
→ Only works for known topics           → Works for any topic/question
→ No fuzzy/semantic matching            → "O(log n)" found by "tree speed"
```

## 🧪 VectorDB & RAG Knowledge Check

1. What is an "embedding" (vector)? Why do similar concepts end up close together?
2. What does RAG stand for? What are the 3 steps in the pipeline?
3. Why is RAG better than putting the entire textbook in the prompt?
4. What is the difference between `Client()` (in-memory) and `PersistentClient()` in ChromaDB?
5. I search for "how fast is tree search?" but my docs say "O(log n) complexity". Would ChromaDB find it? Would a traditional `Ctrl+F` find it? Why?

<details>
<summary>💡 Click to reveal answers</summary>

1. An embedding is a list of numbers (a vector) representing the **meaning** of a piece of text. Similar meanings → similar vectors because the embedding model learned from billions of examples that "car" and "automobile" are used in the same contexts.
2. RAG = **R**etrieval **A**ugmented **G**eneration. Steps: ① Retrieve relevant chunks from VectorDB → ② Augment the prompt with those chunks → ③ Generate an answer using the augmented prompt.
3. Context window limits (~100k tokens can hold maybe 75 pages). A textbook is 500+ pages. RAG selects only the 3–5 most relevant pages → much cheaper, faster, and focused.
4. `Client()` = in-memory, lost when script ends. `PersistentClient(path=...)` = saved to disk, survives restarts. Use persistent for real applications.
5. **ChromaDB: YES.** Semantic search finds meaning matches. **Ctrl+F: NO.** Keyword search only finds exact text matches.

</details>

---

# 🔮 Day 5 Overview — Frameworks and Production Path

**Run code from:** `examples/day_05_level_2/` and `examples/day_05_level_3/`

**Further study links:** `resources/references/day_05_frameworks_prod.md`

```
Day 5 Modules:
  📗 Module 5.1 — Real web search with DuckDuckGo
  📗 Module 5.2 — Plug ChromaDB RAG into full StudyBuddy agent
  📗 Module 5.3 — Rebuild orchestration using LangGraph
  📘 Module 5.4 — Deploy StudyBuddy as a FastAPI web service
  📘 Module 5.5 — Build a Streamlit chat UI
  📘 Module 5.6 — ChromaDB vs pgvector
  📘 Module 5.7 — RAG vs Fine-tuning
```

---
---

# 📗 Day 5 Part A: Intermediate — Upgrade Your StudyBuddy

> ⏱️ **Estimated Time:** 3–4 hours
> 🎯 **Goal:** Add live web search, plug RAG into the full agent, rebuild orchestration with LangGraph
> 📌 **Prerequisites:** Complete Days 1-4 first. Install dependencies from `requirements.txt` before starting.

---

## 🏗️ Day 5 Part A Architecture: StudyBuddy v3.6

```
╔══════════════════════════════════════════════════════════════════════════════╗
║          STUDYBUDDY v3.6 — WHAT LEVEL 2 ADDS (★ = new in this level)        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  STUDENT GOAL ──► LANGGRAPH ORCHESTRATOR ★ (replaces manual loops)         ║
║                           │                                                  ║
║          ┌────────────────┼───────────────┬──────────────┐                  ║
║          ▼                ▼               ▼              ▼                  ║
║      PLANNER          RESEARCHER ★     QUIZ MASTER   EVALUATOR              ║
║      (graph node)   (uses ChromaDB     (graph node)  (with conditional      ║
║                      RAG ★ + web                      edge ★ → replan       ║
║                      search ★)                        if score < 60%)       ║
║          │                │                                                  ║
║          │    ┌───────────▼──────────────────────┐                         ║
║          │    │  KNOWLEDGE LAYER ★               │                         ║
║          │    │  1. Query ChromaDB first          │                         ║
║          │    │  2. If not found → DuckDuckGo ★  │                         ║
║          │    │  3. Store web results → ChromaDB  │                         ║
║          │    └──────────────────────────────────┘                         ║
║          │                                                                   ║
║          └─► CONDITIONAL EDGE ★ (score<60%? → back to planner)            ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Module L2.1 — Real Web Search Tool (45 min)

### Why DuckDuckGo?

- 🆓 **Free** — no API key, no account needed
- 🔒 **Private** — no tracking, no quotas
- 📦 **One install** — `pip install duckduckgo-search`

```
Without Web Search (Day 2):         With Web Search (Level 2):
────────────────────────────        ──────────────────────────────────
search_topic("BST") returns         web_search("BST tutorial 2026") returns
  hardcoded dict with 3 topics       live GeeksForGeeks, YouTube, GitHub links
  same result every run              fresh, real-world, always up-to-date
```

📄 **[View full example → `examples/day_05_level_2/level2_web_search_agent.py`](examples/day_05_level_2/level2_web_search_agent.py)**

> **What this builds:** `StudyBuddy v3.6` — adds live web search via DuckDuckGo (free, no API key).
> The agent can now answer questions about recent topics by searching the web first.
>
> **New library:** `duckduckgo-search` — `pip install duckduckgo-search`
>
> **Run it:**
> ```bash
> python examples/day_05_level_2/level2_web_search_agent.py
> ```

**Expected output:**
```
👤 Student: Find me the best beginner tutorials for Binary Search Trees.
  🔧 Calling: web_search({'query': 'best BST binary search tree tutorial beginner 2026'})
  📤 Result: Result 1:
    Title: Binary Search Tree - GeeksForGeeks
    URL: https://www.geeksforgeeks.org/binary-search-tree-data-structure/...

🤖 StudyBuddy: Here are the best resources I found for BSTs:

1. **GeeksForGeeks BST Guide** (https://geeksforgeeks.org/...)
   Step-by-step insertion/deletion with diagrams — perfect for beginners.

2. **Visualgo BST Visualizer** (https://visualgo.net/bst)
   Watch the tree build itself visually. Great for visual learners!
...
```

---

## Module L2.1b — LangChain Introduction (45 min)

### Why LangChain? (After You Understand the Basics)

You've now built agents from scratch with the raw Gemini SDK. LangChain abstracts common patterns — chains, memory, retrievers — into reusable components. The trade-off: less control but much less boilerplate for common scenarios.

```
Raw Gemini SDK (Days 1-4):          LangChain (Day 5):
────────────────────────────        ────────────────────────────────────
Write every prompt manually         Use pre-built prompt templates
Manually manage history list        ChatMessageHistory handles it
Manually wire RAG pipeline          RetrievalQA chain in 3 lines
Build tool loops yourself           LangChain agent auto-handles loops
```

📄 **[View full example → `examples/day_05_level_2/day5_langchain_intro.py`](examples/day_05_level_2/day5_langchain_intro.py)**

> **What this demonstrates:** Your first LangChain program — builds a `ConversationChain` with built-in memory and then a simple `RetrievalQA` chain connected to ChromaDB. Compare the line count to the equivalent raw SDK code from Day 2-4.
>
> **New packages:** `langchain`, `langchain-google-genai` — `pip install langchain langchain-google-genai`
>
> **Run it:**
> ```bash
> python examples/day_05_level_2/day5_langchain_intro.py
> ```

---

## Module L2.2 — RAG in StudyBuddy v3.0 ResearchAgent (60 min)

### The Problem

Day 3's `ResearchAgent` uses a hardcoded Python dict with only 3 topics. Replace it with ChromaDB so it works for **any topic** — and learns new ones automatically.

```
BEFORE (Day 3):                          AFTER (Level 2):
─────────────────────                    ──────────────────────────────────────
result = CS_KNOWLEDGE_BASE[topic]        results = collection.query(topic)
→ KeyError if topic not in dict          → Semantic match for ANY topic
→ Only 3 pre-coded topics                → New topics stored automatically
→ Static, never grows                    → Knowledge base grows with every query
```

📄 **[View full example → `examples/day_05_level_2/level2_studybuddy_with_rag.py`](examples/day_05_level_2/level2_studybuddy_with_rag.py)**

> **What this builds:** StudyBuddy agent combining ChromaDB RAG + LangChain.
> Shows how to wire a retrieval chain into a conversational agent using LangChain LCEL syntax.
>
> **Run it:**
> ```bash
> python examples/day_05_level_2/level2_studybuddy_with_rag.py
> ```

---

## Module L2.3 — LangGraph Orchestration (90 min)

### Why LangGraph Improves Our Day 3 Orchestrator

```
DAY 3 (manual Python):                LEVEL 2 (LangGraph graph):
─────────────────────────────         ────────────────────────────────────
def run_autonomous_session():         graph nodes: plan→research→quiz→eval
  planner.create_plan()               conditional edge: score<60%→replan
  researcher.research()               ← visual, easy to extend
  quiz.generate()                     ← routing logic is first-class
  evaluator.evaluate()                ← can add parallel nodes easily
  if weak: manual re-call planner     ← built-in state persistence
```

📄 **[View full example → `examples/day_05_level_2/level2_langgraph_studybuddy.py`](examples/day_05_level_2/level2_langgraph_studybuddy.py)**

> **What this builds:** The Day 3 autonomous exam coach *rebuilt* using LangGraph.
> Instead of manually calling each agent in sequence, you define a `StateGraph` where
> nodes = agent functions and edges = flow conditions.
>
> **Why LangGraph wins:** The graph is visual, has built-in state persistence, and handles
> parallel execution (fan-out) without extra code.
>
> **Run it:**
> ```bash
> python examples/day_05_level_2/level2_langgraph_studybuddy.py
> ```

---

## Module L2.4 — HuggingFace Embeddings: Local AI (30 min)

### What is HuggingFace?

HuggingFace is the GitHub of AI models — a platform with 500,000+ open-source models for everything from text generation to image classification to speech recognition.

The key package is `sentence-transformers` — it lets you generate embeddings **locally** using open-source models, with NO API key and NO cost.

```
API-based embeddings (e.g., OpenAI Embeddings):        Local embeddings (HuggingFace):
────────────────────────────────────────────────        ─────────────────────────────────────
Requires paid API key                                   Free, runs on your CPU
Data leaves your machine for each request               Data stays on your machine
Fast at scale, cheap per token                          Free always, private always
~$0.0001 per 1k tokens                                 $0 forever
```

📄 **[View full example → `examples/day_05_level_2/day5_huggingface_embeddings.py`](examples/day_05_level_2/day5_huggingface_embeddings.py)**

> **What this demonstrates:** Using `sentence-transformers` to generate embeddings locally,
> storing them in ChromaDB, and comparing with the default embedding function.
> The model (`all-MiniLM-L6-v2`) downloads **once** (~80MB) and runs entirely on your CPU after that.
>
> **New package:** `sentence-transformers` — `pip install sentence-transformers`
>
> **Run it:**
> ```bash
> python examples/day_05_level_2/day5_huggingface_embeddings.py
> ```

---

## 🧪 Level 2 Knowledge Check

1. Why is DuckDuckGo better than Google Search API for this tutorial?
2. In L2.2, the agent stores Gemini-generated content back into ChromaDB. What long-term benefit does this create?
3. In LangGraph, what is a `StateGraph`? Why is it better than a Python `for` loop for orchestration?
4. What does `should_replan()` return and why does it return a **string** rather than a boolean?
5. Each LangGraph node returns only a partial dict (e.g., `{"study_plan": plan}`). Why not return the full state?

<details>
<summary>💡 Click to reveal answers</summary>

1. DuckDuckGo is **free with zero setup** — no API key, no billing account. Google Search API requires account creation, billing, and a paid quota. Removing that friction is crucial for a learning course.
2. **The knowledge base grows automatically** — topics researched once are cached for future queries. The second query for "BST" skips Gemini and returns instantly from ChromaDB, saving API calls and cost.
3. A `StateGraph` is a **directed graph** where nodes are agent functions and edges define flow. A `for` loop is sequential-only; it can't branch conditionally or loop back without messy `if/while` logic. LangGraph makes conditional branching first-class via `add_conditional_edges`.
4. It returns a **string** because `path_map` maps strings to node names. A boolean can only represent two states; strings can map to any number of branches (`"replan"`, `"done"`, `"escalate"`, etc.).
5. LangGraph automatically **merges** the partial dict with the existing state. Returning only changed keys is cleaner, avoids accidentally overwriting unchanged fields, and is the standard LangGraph pattern.

</details>

---
---

# 📘 Day 5 Part B: Advanced — Production-Ready StudyBuddy

> ⏱️ **Estimated Time:** 4–5 hours
> 🎯 **Goal:** Deploy as a web API, build a chat UI, understand production DB choices, and master the RAG vs fine-tuning decision
> 📌 **Prerequisites:** Complete Day 5 Part A first. Install dependencies from `requirements.txt` before starting.

---

## 🏗️ Day 5 Part B Architecture: StudyBuddy v4.0

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    STUDYBUDDY v4.0 — PRODUCTION STACK                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   BROWSER / MOBILE                                                           ║
║   ┌─────────────────────────────────────────┐                               ║
║   │   STREAMLIT UI  (Module L3.2)           │                               ║
║   │   Chat interface • Sidebar controls     │                               ║
║   └──────────────────┬──────────────────────┘                               ║
║                      │  HTTP POST /ask                                       ║
║                      ▼                                                       ║
║   ┌─────────────────────────────────────────┐                               ║
║   │   FASTAPI SERVICE  (Module L3.1)        │                               ║
║   │   /ask  /quiz  /health  /sessions       │                               ║
║   │   Session management • CORS             │                               ║
║   └──────────────────┬──────────────────────┘                               ║
║                      │                                                       ║
║                      ▼                                                       ║
║   ┌─────────────────────────────────────────┐                               ║
║   │   STUDYBUDDY AGENT  (Day 2 or LangGraph)│                               ║
║   └──────────────────┬──────────────────────┘                               ║
║                      │                                                       ║
║                      ▼                                                       ║
║   ┌───────────────────────────────────────────────────────────┐             ║
║   │  VECTORDB  (Module L3.3)                                  │             ║
║   │  Dev: ChromaDB (local SQLite, zero setup)                 │             ║
║   │  Prod: pgvector (PostgreSQL — scalable, SQL joins)        │             ║
║   └───────────────────────────────────────────────────────────┘             ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Module L3.1 — FastAPI Web Service (60 min)

### From Terminal Script to HTTP API

```
BEFORE (terminal script):              AFTER (FastAPI service):
python script.py                       curl -X POST http://localhost:8000/ask \
  → only you can use it                  -d '{"question":"Explain BST"}'
  → no network access                  → classmates on your Wi-Fi can use it
  → terminal only                      → works with Postman, React, mobile
                                       → deploy to cloud with one command
```

📄 **[View full example → `examples/day_05_level_3/level3_fastapi_service.py`](examples/day_05_level_3/level3_fastapi_service.py)**

> **What this builds:** `StudyBuddy v4.0` as a production REST API.
> Endpoints: `GET /health`, `POST /ask` (session memory), `POST /quiz`, `DELETE /session/{id}`.
>
> **Key patterns:** Pydantic models, in-memory session store, CORS middleware,
> automatic docs at `http://localhost:8000/docs`.
>
> **Run it:**
> ```bash
> pip install fastapi uvicorn
> uvicorn examples.day_05_level_3.level3_fastapi_service:app --reload --port 8000
> ```

> 💡 **How to run & test:**
> ```bash
> # Terminal 1 — start server (--reload auto-restarts on file save)
> uvicorn level3_fastapi_service:app --reload --port 8000
>
> # Terminal 2 — test endpoints
> curl http://localhost:8000/health
>
> curl -X POST http://localhost:8000/ask \
>      -H "Content-Type: application/json" \
>      -d '{"question": "What is a binary search tree?"}'
>
> # Visit interactive docs (try endpoints in browser!):
> # http://localhost:8000/docs
> ```

---

## Module L3.2 — Streamlit Chat UI (60 min)

📄 **[View full example → `examples/day_05_level_3/level3_streamlit_app.py`](examples/day_05_level_3/level3_streamlit_app.py)**

> **What this builds:** A browser chat UI for StudyBuddy using Streamlit.
> Sidebar controls topic, temperature, and mode (Chat / Quiz Me / Study Notes).
> Full conversation memory via `st.session_state`.
>
> **Key Streamlit pattern:** `@st.cache_resource` creates the Gemini client once
> and reuses it across all reruns — without it, a new client would form on every keystroke.
>
> **Run it:**
> ```bash
> pip install streamlit
> streamlit run examples/day_05_level_3/level3_streamlit_app.py
> ```

> 💡 **How to run:**
> ```bash
> streamlit run level3_streamlit_app.py
> # Browser opens automatically at http://localhost:8501
> ```

---

## Module L3.3 — Production VectorDB: ChromaDB vs pgvector (45 min)

```
╔══════════════════════════════════════════════════════════════════════════════╗
║               CHROMADB vs pgvector — WHEN TO USE WHICH                       ║
╠═════════════════════════╦════════════════════════════════════════════════════╣
║  ChromaDB               ║  pgvector (PostgreSQL extension)                   ║
╠═════════════════════════╬════════════════════════════════════════════════════╣
║ ✅ pip install, zero     ║ ⚙️  Needs PostgreSQL server (Docker or cloud)      ║
║    setup                ║ ✅ Runs on AWS RDS, Supabase, Railway, Heroku      ║
║ ✅ Works on your laptop  ║ ✅ Handles millions of vectors easily              ║
║ ✅ Python-native API     ║ ✅ Full SQL: JOIN vectors with other tables        ║
║ ❌ Single machine only   ║ ✅ Multi-user, ACID transactions, backups          ║
║ ❌ No SQL joins          ║ ❌ Requires Docker or cloud DB setup               ║
╠═════════════════════════╩════════════════════════════════════════════════════╣
║  USE ChromaDB:  learning, prototypes, local scripts, solo dev               ║
║  USE pgvector:  production, multi-user, SQL queries needed                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

**Start PostgreSQL + pgvector locally using Docker:**

```bash
# Pull and start the pgvector image (one command!)
docker run -d \
  --name studybuddy-pgvector \
  -e POSTGRES_USER=student \
  -e POSTGRES_PASSWORD=studybuddy \
  -e POSTGRES_DB=studybuddy_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Verify it started:
docker ps   # should show studybuddy-pgvector

# Stop / start later:
docker stop studybuddy-pgvector
docker start studybuddy-pgvector
```

📄 **[View full example → `examples/day_05_level_3/level3_pgvector_demo.py`](examples/day_05_level_3/level3_pgvector_demo.py)**

> **What this demonstrates:** Semantic search using PostgreSQL + pgvector.
> Same RAG concept as ChromaDB, but on a production-grade database that scales to millions
> of rows, supports SQL JOINs, and runs on AWS RDS / Supabase / Railway.
>
> **New SQL operator:** `<=>` = cosine distance between two vectors.
> `ORDER BY embedding <=> query_vec` = "find rows most similar to this query".
>
> **Requires:** Docker running pgvector (see Docker setup commands above).
>
> **Run it:**
> ```bash
> python examples/day_05_level_3/level3_pgvector_demo.py
> ```

---

## Module L3.4 — RAG vs Fine-tuning: When to Use Which (45 min)

```
╔══════════════════════════════════════════════════════════════════════════════╗
║              RAG vs FINE-TUNING — DECISION FLOWCHART                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  START: "I want the AI to answer questions about MY specific data"           ║
║                              │                                               ║
║                              ▼                                               ║
║  Q1: Does the data change frequently?                                        ║
║      YES → RAG ✅  (update VectorDB, no retraining)                         ║
║      NO  → continue ↓                                                       ║
║                                                                              ║
║  Q2: Budget < $1,000 OR no ML expertise?                                    ║
║      YES → RAG ✅  (fine-tuning is expensive and complex)                   ║
║      NO  → continue ↓                                                       ║
║                                                                              ║
║  Q3: Need a different STYLE or PERSONA (not just new knowledge)?            ║
║      YES → Fine-tuning ✅  (teaches tone/style, not just facts)             ║
║      NO  → continue ↓                                                       ║
║                                                                              ║
║  Q4: Data is confidential — must NEVER appear in a prompt?                  ║
║      YES → Fine-tuning ✅  (data baked into weights, never in context)      ║
║      NO  → RAG ✅  (almost always the right default choice)                 ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  RAG:          Cheap • Fast to update • No ML expertise • Use by default    ║
║  Fine-tuning:  Expensive • Static • Needs ML expertise • Rare cases only    ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

📄 **[View full example → `examples/day_05_level_3/level3_rag_vs_finetune_comparison.py`](examples/day_05_level_3/level3_rag_vs_finetune_comparison.py)**

> **What this demonstrates:** A side-by-side RAG vs plain Gemini comparison.
> Plain Gemini asked about StudyBuddy's grading policy → hallucinated answer.
> RAG Gemini with policy docs in ChromaDB → correct, specific answer.
>
> **The lesson:** RAG beats fine-tuning when data changes frequently, is company-specific,
> or the budget does not support a $10K+ fine-tuning run.
>
> **Run it:**
> ```bash
> python examples/day_05_level_3/level3_rag_vs_finetune_comparison.py
> ```

---

## 🧪 Level 3 Knowledge Check

1. What does CORS do? What would break if you removed it when Streamlit calls the FastAPI service?
2. What does `@st.cache_resource` do in Streamlit? What would happen without it?
3. In pgvector, what does the `<=>` operator do? How is it different from SQL `=`?
4. What is an `ivfflat` index? Why do we need it when we already have a vector column?
5. In the RAG vs fine-tuning flowchart, give a real example (not StudyBuddy) where fine-tuning IS the right choice.
6. Why does plain Gemini give a wrong answer about the 78% score? What does this prove about RAG?

<details>
<summary>💡 Click to reveal answers</summary>

1. **CORS** adds HTTP headers that tell the browser "it's safe for `localhost:8501` (Streamlit) to call `localhost:8000` (FastAPI)." Without CORS, the browser blocks the request for security — Streamlit UI would show a network error on every API call.
2. `@st.cache_resource` creates the Gemini client **once** and reuses it across all reruns. Without it, Streamlit creates a brand-new `genai.Client()` on every user keystroke — wasting connections, slowing responses, and potentially hitting initialisation rate limits.
3. `<=>` computes **cosine distance** between two vectors (0.0 = same direction/meaning, 2.0 = opposite). SQL `=` checks exact byte-level equality — two vectors are almost never exactly equal, so `=` would never return results from a semantic search.
4. `ivfflat` is an **Approximate Nearest Neighbour index** that partitions vectors into `lists` groups. Without it, PostgreSQL scans every row for every query (O(n) — slow at 1M+ rows). With it, the search narrows to the closest partitions first (much faster), at the cost of occasional near-misses (approximate, not exact).
5. Example: A legal firm's chatbot that must always respond in formal British English with specific liability disclaimer wording. Fine-tuning bakes **style and tone** permanently into the model — RAG can inject the disclaimer text but can't change HOW the model phrases sentences.
6. Gemini was never trained on StudyBuddy's internal grading policy. Without context, it either refuses or **hallucinates** a plausible-sounding but wrong answer. This proves that RAG is essential whenever you need answers grounded in **your specific data** — no amount of prompting will make the model know facts it was never trained on.

</details>

---

```
╔════════════════════════════════════════════════════════════════════════════╗
║             AGENTIC AI CHEAT SHEET                          ║
╠════════════════════════════════════════════════════════════════════════════╣
║  CORE CONCEPTS                                               ║
║  Token       = chunk of text processed by LLM               ║
║  Temperature = creativity level (0=safe, 1=creative)        ║
║  Context     = what the model can "see" right now           ║
║  Tool        = Python function the AI can call              ║
║  Agent Loop  = Perceive → Think → Act → Observe → Repeat    ║
║  ReAct       = Thought + Action + Observation pattern        ║
║  Orchestrator= Manager agent that coordinates others        ║
╠════════════════════════════════════════════════════════════════════════════╣
║  VECTORDB & RAG CONCEPTS                                     ║
║  Embedding   = text converted to a list of numbers (vector) ║
║  VectorDB    = database that stores & searches vectors       ║
║  ChromaDB    = free, local, open-source VectorDB            ║
║  RAG         = Retrieval → Augment prompt → Generate answer  ║
║  Similarity  = how close two vectors are (0=different,      ║
║               1=identical meaning)                           ║
╠════════════════════════════════════════════════════════════════════════════╣
║  WHEN TO USE WHAT                                           ║
║  Gen AI      → Single Q&A, content generation, summarizing  ║
║  AI Agent    → Tasks needing tools, memory, multi-turn      ║
║  Agentic AI  → Complex goals, parallel work, low supervision║
║  RAG + Agent → Agent needs to query YOUR documents/data     ║
╠════════════════════════════════════════════════════════════════════════════╣
║  ALWAYS DO                                                  ║
║  ✓ Set iteration limits on agent loops                      ║
║  ✓ Add try/except around API calls                          ║
║  ✓ Log every agent action                                   ║
║  ✓ Use Human-in-the-Loop for high-stakes tasks              ║
║  ✓ Never commit your API key to GitHub (.gitignore!)        ║
╚════════════════════════════════════════════════════════════════════════════╝
```

---

*🤖 This course was designed to give you a complete, practical foundation in Agentic AI. The skills you've learned — prompt engineering, tool calling, multi-agent orchestration — are among the most in-demand in the tech industry today. Keep building! The best way to learn is to add your own twists to StudyBuddy and see what you can create.*

*Happy coding! 🚀*

