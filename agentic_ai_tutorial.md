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
> ```bash
> # requirements.txt — paste these into a file named requirements.txt
> # then run:  pip install -r requirements.txt
>
> google-genai      # Gemini API — used in every file
> python-dotenv     # loads API keys from .env file
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

```python
# verify_setup.py — run this first!
#
# ─── INSTALL REQUIRED PACKAGES ───────────────────────────────────────────────
# pip install google-genai python-dotenv
# ─────────────────────────────────────────────────────────────────────────────

import os                        # ✅ stdlib — no install needed
from dotenv import load_dotenv   # 📦 python-dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    print(f"✅ API key found: {api_key[:8]}...{api_key[-4:]}")
else:
    print("❌ API key NOT found. Check your .env file!")

try:
    from google import genai
    print("✅ google-genai package installed correctly")
except ImportError:
    print("❌ Import failed. Run: pip install google-genai")
```

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

```python
# day1_hello_genai.py
#
# ─── INSTALL REQUIRED PACKAGES ───────────────────────────────────────────────
# pip install google-genai python-dotenv
# ─────────────────────────────────────────────────────────────────────────────

import os                        # ✅ stdlib — no install needed
from dotenv import load_dotenv   # 📦 python-dotenv
from google import genai         # 📦 google-genai

# Load API key from .env file
load_dotenv()

# Create the client (your connection to Gemini)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Make your first request!
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="What is a binary search tree? Explain in 3 sentences."
)

# response.text  ← This is where Gemini's reply lives.
# The full response object contains metadata (tokens used, safety ratings, etc.)
# but .text is a shortcut that gives you just the string content.
print(response.text)
```

> 💡 **How to run this file:**
> ```bash
> # In your terminal, from your project folder:
> python day1_hello_genai.py
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

```python
# day1_prompt_engineering.py
#
# ─── INSTALL REQUIRED PACKAGES ───────────────────────────────────────────────
# pip install google-genai python-dotenv
# ─────────────────────────────────────────────────────────────────────────────

import os                        # ✅ stdlib — no install needed
from dotenv import load_dotenv   # 📦 python-dotenv
from google import genai         # 📦 google-genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def ask_ai(prompt):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text

# ❌ BAD PROMPT — vague, no structure
bad_prompt = "explain trees"
print("=== BAD PROMPT RESULT ===")
print(ask_ai(bad_prompt))

print("\n" + "="*50 + "\n")

# ✅ GOOD PROMPT — role, task, format, context
good_prompt = """
You are a friendly and encouraging computer science tutor for college freshmen.

Your task: Explain binary search trees (BST).

Format your response as:
1. One-line definition (no jargon)
2. Real-world analogy (relatable to college students)
3. Key properties (3 bullet points)
4. Simple Python code example with comments
5. One common interview question about BSTs

Keep the tone friendly and conversational.
"""
print("=== GOOD PROMPT RESULT ===")
print(ask_ai(good_prompt))
```

### The System Prompt — Giving AI a Personality

A **system prompt** sets the AI's role and behavior for the entire conversation. Think of it as the job description you give to your AI intern at the start of the day.

```python
# day1_system_prompt.py
#
# ─── INSTALL REQUIRED PACKAGES ───────────────────────────────────────────────
# pip install google-genai python-dotenv
# ─────────────────────────────────────────────────────────────────────────────

import os                        # ✅ stdlib — no install needed
from dotenv import load_dotenv   # 📦 python-dotenv
from google import genai         # 📦 google-genai
from google.genai import types   # 📦 google-genai (same package, sub-module)

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# System prompt defines the AI's persona and rules
SYSTEM_PROMPT = """
You are "StudyBuddy", a friendly AI tutor specializing in computer science 
for college students. 

Your rules:
- Always use simple language. No unnecessary jargon.
- Give a real-world analogy for every concept.
- Include a short Python code snippet when explaining algorithms or data structures.
- End every response with one "Quick Quiz" question to test understanding.
- If a student seems confused, be extra encouraging.
"""

def study_buddy_ask(question: str) -> str:
    """Ask StudyBuddy a question with a consistent personality."""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,      # Slightly creative but mostly accurate
            max_output_tokens=800 # Keep answers concise
        ),
        contents=question
    )
    return response.text

# Test our StudyBuddy
print(study_buddy_ask("What is recursion?"))
print("\n" + "─"*60 + "\n")
print(study_buddy_ask("How does merge sort work?"))
```

### Temperature — Controlling Creativity

```python
# day1_temperature.py
#
# ─── INSTALL REQUIRED PACKAGES ───────────────────────────────────────────────
# pip install google-genai python-dotenv
# ─────────────────────────────────────────────────────────────────────────────

import os                        # ✅ stdlib — no install needed
from dotenv import load_dotenv   # 📦 python-dotenv
from google import genai         # 📦 google-genai
from google.genai import types   # 📦 google-genai (same package, sub-module)

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def ask_with_temperature(prompt: str, temperature: float) -> str:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(temperature=temperature),
        contents=prompt
    )
    return response.text

prompt = "Write a one-sentence analogy for a binary search tree."

print(f"Temperature 0.0 (Deterministic):")
print(ask_with_temperature(prompt, 0.0))

print(f"\nTemperature 0.5 (Balanced):")
print(ask_with_temperature(prompt, 0.5))

print(f"\nTemperature 1.5 (Very Creative):")
print(ask_with_temperature(prompt, 1.5))
```

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

```python
# day1_study_notes_generator.py
"""
DAY 1 — Study Notes Generator (Version 1.0)
The simplest version of our AI Study Buddy.
Limitation: One-shot. No memory. No tools. No real-time data.
"""
#
# ─── INSTALL REQUIRED PACKAGES ───────────────────────────────────────────────
# pip install google-genai python-dotenv
# ─────────────────────────────────────────────────────────────────────────────

import os                        # ✅ stdlib — no install needed
import pathlib                   # ✅ stdlib — no install needed
import datetime                  # ✅ stdlib — no install needed
from dotenv import load_dotenv   # 📦 python-dotenv
from google import genai         # 📦 google-genai
from google.genai import types   # 📦 google-genai (same package, sub-module)

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ─── System Prompt — StudyBuddy's Personality ───────────────────────────────
STUDY_BUDDY_SYSTEM_PROMPT = """
You are StudyBuddy, an expert computer science tutor for college students.

When generating study notes, ALWAYS follow this exact format:

# [Topic Name]

## 📝 What is it?
[One paragraph — simple definition with NO jargon. Use plain English.]

## 🌍 Real-World Analogy
[A relatable analogy that a college student would understand.]

## ⚙️ How It Works (Step-by-Step)
[Numbered steps explaining the concept]

## 💻 Python Code Example
~~~python
# Well-commented Python code example goes here
~~~

## 📊 Time & Space Complexity
| Operation | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| operation | O(...)          | O(...)           |

## ✅ Key Takeaways
- Key point 1
- Key point 2
- Key point 3

## ❓ Practice Questions
1. Beginner: [simple concept question]
2. Intermediate: [application question]
3. Advanced: [design/trade-off question]

---
StudyBuddy Note: [One encouraging sentence]
"""

def generate_study_notes(topic: str, save_to_file: bool = True) -> str:
    """
    Generate structured study notes for a given CS topic.

    Args:
        topic: The CS topic to generate notes for (e.g., "Binary Search Tree")
        save_to_file: Whether to save the notes to a markdown file

    Returns:
        The generated study notes as a string

    Sample output for topic="Binary Search Tree":
    ─────────────────────────────────────────────
    # Binary Search Tree

    ## 📝 What is it?
    A Binary Search Tree (BST) is a special kind of tree where every node has
    at most two children. The rule is simple: everything to the LEFT of a node
    is SMALLER, and everything to the RIGHT is LARGER.

    ## 🌍 Real-World Analogy
    Think of a BST like a phone book that's organized for fast searching.
    Each page tells you "go left for names before this" or "go right for
    names after this." You never have to read the whole book — just follow
    the signs!

    ## ⚙️ How It Works (Step-by-Step)
    1. Start at the root (top) node
    2. Compare your value with the current node
    3. If smaller → go LEFT. If larger → go RIGHT
    4. Repeat until you find the value or hit a None (not found)

    ## 💻 Python Code Example
    ~~~python
    class Node:
        def __init__(self, value):
            self.value = value
            self.left = None   # smaller values go here
            self.right = None  # larger values go here

    class BST:
        def __init__(self):
            self.root = None

        def insert(self, value):
            if not self.root:
                self.root = Node(value)
            else:
                self._insert(self.root, value)

        def _insert(self, node, value):
            if value < node.value:
                if node.left is None:
                    node.left = Node(value)  # found the spot!
                else:
                    self._insert(node.left, value)  # keep going left
            else:
                if node.right is None:
                    node.right = Node(value)
                else:
                    self._insert(node.right, value)

    # Usage
    tree = BST()
    for val in [5, 3, 7, 1, 4]:
        tree.insert(val)
    # Result:       5
    #              / \\
    #             3   7
    #            / \\
    #           1   4
    ~~~

    ## 📊 Time & Space Complexity
    | Operation | Average Case | Worst Case (unbalanced) | Space |
    |-----------|-------------|------------------------|-------|
    | Search    | O(log n)    | O(n)                   | O(1)  |
    | Insert    | O(log n)    | O(n)                   | O(1)  |
    | Delete    | O(log n)    | O(n)                   | O(1)  |
    | Traversal | O(n)        | O(n)                   | O(n)  |

    ## ✅ Key Takeaways
    - BST property: left < node < right (ALWAYS — even after insertions/deletions)
    - Balanced BST → O(log n). Unbalanced (like a line) → degrades to O(n)
    - In-order traversal of a BST gives you values in SORTED ORDER
    - AVL trees and Red-Black trees are self-balancing BSTs

    ## ❓ Practice Questions
    1. (Beginner) What happens if you insert values 1,2,3,4,5 in order into a BST?
       Draw it. Is it balanced?
    2. (Intermediate) Write a function to check if a given binary tree IS a valid BST.
    3. (Advanced) Why does deleting a node with two children require finding the
       "in-order successor"? What would happen if you just removed the node?
    ─────────────────────────────────────────────
    """
    print(f"📚 Generating study notes for: {topic}")
    print("🤔 Thinking...")

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=STUDY_BUDDY_SYSTEM_PROMPT,
            temperature=0.4,        # Low temp = consistent, accurate notes
            max_output_tokens=2000  # Enough for full study notes
        ),
        contents=f"Generate comprehensive study notes for: {topic}"
    )

    notes = response.text

    # Save to file for later reference
    if save_to_file:
        output_dir = pathlib.Path("study_notes")
        output_dir.mkdir(exist_ok=True)

        # Create a safe filename from the topic
        safe_filename = topic.lower().replace(" ", "_").replace("/", "_")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        filepath = output_dir / f"{safe_filename}_{timestamp}.md"

        filepath.write_text(notes, encoding="utf-8")
        print(f"✅ Notes saved to: {filepath}")

    return notes


def interactive_study_session():
    """Run an interactive study notes session."""
    print("=" * 60)
    print("       🤖 StudyBuddy v1.0 — Study Notes Generator       ")
    print("=" * 60)
    print("  Powered by: Gemini LLM (Generative AI)")
    print("  Limitation: No memory, no tools, no real-time data")
    print("=" * 60)

    while True:
        print("\nEnter a CS topic to study (or 'quit' to exit):")
        topic = input("📖 Topic: ").strip()

        if topic.lower() in ("quit", "exit", "q"):
            print("\n👋 Happy studying! See you next time!")
            break

        if not topic:
            print("⚠️ Please enter a topic!")
            continue

        notes = generate_study_notes(topic)
        print("\n" + notes)
        print("\n" + "─" * 60)

        # Quick limitation demonstration
        print("\n⚠️  NOTICE — Day 1 Limitation:")
        print("   StudyBuddy v1.0 can only answer from its training data.")
        print("   It cannot: search the web, remember past sessions,")  
        print("   give you real-time info, or ask follow-up questions.")
        print("   This is what we improve on Day 2! 🚀")


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Demo: Generate notes for Binary Search Tree
    # (We'll use this same topic across all 3 days)
    notes = generate_study_notes("Binary Search Tree")
    print(notes)

    # Uncomment to run interactive mode:
    # interactive_study_session()
```

### 🧪 Exercise: Extend the Generator

Try adding these features yourself:

```python
# EXERCISE 1: Add difficulty levels
def generate_notes_with_difficulty(topic: str, level: str = "beginner"):
    """
    level can be: "beginner", "intermediate", "advanced"
    Hint: Add the level to your prompt!
    """
    # YOUR CODE HERE
    pass

# EXERCISE 2: Generate a mini-quiz from notes
def generate_quiz_from_notes(notes: str, num_questions: int = 5):
    """
    Takes existing notes and generates an MCQ quiz.
    Hint: Pass the notes as context in your prompt!
    """
    # YOUR CODE HERE
    pass
```

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

```python
# day2_function_calling_basics.py
"""
How Function Calling works with Gemini:
1. You define a Python function
2. You describe it to the AI (name, what it does, parameters)
3. The AI decides IF and WHEN to call it
4. YOUR code actually runs the function
5. You give the result back to the AI
"""
#
# ─── INSTALL REQUIRED PACKAGES ───────────────────────────────────────────────
# pip install google-genai python-dotenv
# ─────────────────────────────────────────────────────────────────────────────

import os                        # ✅ stdlib — no install needed
import datetime                  # ✅ stdlib — no install needed
from dotenv import load_dotenv   # 📦 python-dotenv
from google import genai         # 📦 google-genai
from google.genai import types   # 📦 google-genai (same package, sub-module)

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ─── Step 1: Define your actual Python functions ─────────────────────────────

def get_current_date() -> str:
    """Returns today's date in a human-readable format."""
    return datetime.datetime.now().strftime("%A, %B %d, %Y")

def calculate_days_until_exam(exam_date_str: str) -> str:
    """
    Calculates how many days until an exam.
    exam_date_str: Date in format 'YYYY-MM-DD'
    """
    try:
        exam_date = datetime.datetime.strptime(exam_date_str, "%Y-%m-%d").date()
        today = datetime.date.today()
        days_left = (exam_date - today).days
        if days_left < 0:
            return f"That exam was {abs(days_left)} days ago."
        elif days_left == 0:
            return "The exam is TODAY! Good luck! 🍀"
        else:
            return f"You have {days_left} days until your exam."
    except ValueError:
        return "Invalid date format. Please use YYYY-MM-DD."

def get_cs_topic_difficulty(topic: str) -> str:
    """Returns the difficulty level and estimated study time for a CS topic."""
    # Simplified knowledge base
    topic_info = {
        "binary search": {"difficulty": "Easy", "hours": 2},
        "binary search tree": {"difficulty": "Medium", "hours": 4},
        "bst": {"difficulty": "Medium", "hours": 4},
        "dynamic programming": {"difficulty": "Hard", "hours": 10},
        "graph algorithms": {"difficulty": "Hard", "hours": 8},
        "linked list": {"difficulty": "Easy", "hours": 2},
        "sorting algorithms": {"difficulty": "Medium", "hours": 5},
        "recursion": {"difficulty": "Medium", "hours": 3},
    }
    
    topic_lower = topic.lower()
    for key, info in topic_info.items():
        if key in topic_lower:
            return (f"Topic: {topic}\n"
                   f"Difficulty: {info['difficulty']}\n"
                   f"Recommended study time: ~{info['hours']} hours")
    
    return f"Topic: {topic}\nDifficulty: Unknown\nRecommended study time: ~3-5 hours"

# ─── Step 2: Describe your functions to the AI ───────────────────────────────
# This tells the AI WHAT the function does and WHAT parameters it takes
#
# WHY do we need to "describe" functions the AI can already see?
# Because the AI doesn't read your Python code — it only sees text prompts!
# We must TELL it: "Hey, you can call a function named X. It does Y. 
# It takes these parameters of these types."
#
# Think of it like a job posting:
#   FunctionDeclaration = the job title + description
#   parameters (Schema) = the required qualifications
#
# Once described, Gemini can decide ON ITS OWN when to call the function.

tools = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="get_current_date",
            description="Gets today's current date. Use this when you need to know what day it is.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={})
        ),
        types.FunctionDeclaration(
            name="calculate_days_until_exam",
            description="Calculates how many days are left until a specific exam date.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "exam_date_str": types.Schema(
                        type=types.Type.STRING,
                        description="The exam date in YYYY-MM-DD format, e.g., '2026-06-15'"
                    )
                },
                required=["exam_date_str"]
            )
        ),
        types.FunctionDeclaration(
            name="get_cs_topic_difficulty",
            description="Returns difficulty level and estimated study hours for a CS topic.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "topic": types.Schema(
                        type=types.Type.STRING,
                        description="The CS topic name, e.g., 'Binary Search Tree', 'Dynamic Programming'"
                    )
                },
                required=["topic"]
            )
        )
    ])
]

# ─── Step 3: The tool executor — maps AI's choice to actual function ──────────

AVAILABLE_TOOLS = {
    "get_current_date": get_current_date,
    "calculate_days_until_exam": calculate_days_until_exam,
    "get_cs_topic_difficulty": get_cs_topic_difficulty,
}

def execute_tool_call(function_call) -> str:
    """Execute whatever function the AI decided to call."""
    function_name = function_call.name
    function_args = dict(function_call.args) if function_call.args else {}
    
    if function_name not in AVAILABLE_TOOLS:
        return f"Error: Unknown function '{function_name}'"
    
    print(f"  🔧 Calling tool: {function_name}({function_args})")
    result = AVAILABLE_TOOLS[function_name](**function_args)
    print(f"  📤 Tool result: {result}")
    return str(result)

# ─── Step 4: The Agent Loop ───────────────────────────────────────────────────

def run_agent(user_message: str) -> str:
    """
    Run the AI agent loop:
    1. Send message to AI
    2. If AI wants to call a tool → run the tool → send result back
    3. Repeat until AI gives a final answer
    """
    print(f"\n👤 User: {user_message}")
    print("🤖 Agent thinking...\n")
    
    messages = [{"role": "user", "parts": [{"text": user_message}]}]
    
    max_iterations = 5  # Safety limit
    for iteration in range(max_iterations):
        # Send to AI (with tools available)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                tools=tools,
                system_instruction=(
                    "You are StudyBuddy, a helpful CS tutor. "
                    "Use the available tools to give accurate, helpful answers."
                )
            ),
            contents=messages
        )
        
        # Check if AI wants to call a tool
        candidate = response.candidates[0]
        has_tool_call = any(
            part.function_call for part in candidate.content.parts
            if hasattr(part, 'function_call') and part.function_call
        )
        
        if has_tool_call:
            # AI requested a tool — run it and loop back
            tool_results = []
            for part in candidate.content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    result = execute_tool_call(part.function_call)
                    tool_results.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=part.function_call.name,
                                response={"result": result}
                            )
                        )
                    )
            
            # Add AI's tool request + our results to message history
            messages.append({"role": "model", "parts": candidate.content.parts})
            messages.append({"role": "user", "parts": tool_results})
        
        else:
            # AI gave a final text answer — we're done!
            final_answer = response.text
            print(f"🤖 StudyBuddy Agent: {final_answer}")
            return final_answer
    
    return "Agent reached maximum iterations without completing."


# ─── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # The AI will use tools automatically to answer these!
    run_agent("What's today's date?")
    run_agent("My DSA exam is on 2026-06-20. How much time do I have?")
    run_agent("How difficult is Binary Search Tree and how long should I study it?")
    run_agent(
        "My DSA exam is on 2026-06-20. I need to study Binary Search Trees. "
        "How many days do I have and is that enough time?"
    )  # This one will call MULTIPLE tools!
```

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

```python
# day2_memory.py
"""
Adding conversation memory to StudyBuddy.
We store the full chat history and pass it with every request.
"""
#
# ─── INSTALL REQUIRED PACKAGES ───────────────────────────────────────────────
# pip install google-genai python-dotenv
# ─────────────────────────────────────────────────────────────────────────────

import os                        # ✅ stdlib — no install needed
from dataclasses import dataclass, field  # ✅ stdlib — no install needed
from dotenv import load_dotenv   # 📦 python-dotenv
from google import genai         # 📦 google-genai
from google.genai import types   # 📦 google-genai (same package, sub-module)

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@dataclass
class StudyBuddyWithMemory:
    """
    StudyBuddy v1.5 — Now with conversation memory!
    Remembers everything you told it within a session.
    """
    model_name: str = "gemini-2.0-flash"
    history: list = field(default_factory=list)          # Chat history
    student_profile: dict = field(default_factory=dict)  # Persistent facts

    SYSTEM_PROMPT = """
    You are StudyBuddy, a friendly CS tutor with memory.
    You remember everything the student tells you in this session.
    Refer back to previous context naturally (like a real tutor would).
    Track: student name, exam dates, topics being studied, difficulty levels noted.
    """

    def chat(self, user_message: str) -> str:
        """Send a message, maintaining full conversation history."""
        # Add user's message to history
        self.history.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })

        # Send ENTIRE history to the AI — this is how memory works!
        response = client.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=self.SYSTEM_PROMPT,
                temperature=0.5
            ),
            contents=self.history  # ← The entire conversation history!
        )

        ai_response = response.text

        # Add AI's response to history
        self.history.append({
            "role": "model",
            "parts": [{"text": ai_response}]
        })

        return ai_response

    def reset(self):
        """Start a fresh conversation."""
        self.history = []
        self.student_profile = {}
        print("🔄 Memory cleared. Starting a new session.")


def demo_memory():
    """Show how memory makes conversations contextual."""
    buddy = StudyBuddyWithMemory()

    conversations = [
        "Hi! I'm Alex. I'm studying for my Data Structures exam on June 20.",
        "I'm struggling with Binary Search Trees.",
        "Can you give me a quick BST recap?",
        "Now, what was my exam topic I said I'm struggling with?",  # Tests memory!
        "And when is my exam again?",  # Tests memory again!
    ]

    print("=" * 60)
    print("  📝 StudyBuddy with Memory Demo")
    print("=" * 60)

    for message in conversations:
        print(f"\n👤 Alex: {message}")
        response = buddy.chat(message)
        print(f"🤖 StudyBuddy: {response}")
        print(f"   [Memory size: {len(buddy.history)} message(s) stored]")

if __name__ == "__main__":
    demo_memory()
```

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

```python
# day2_research_quiz_agent.py
"""
DAY 2 — StudyBuddy v2.0: Research & Quiz Agent
Upgrades from Day 1:
  + Uses tools (search simulation, quiz generation, answer checking)
  + Has conversation memory
  + Tracks student progress
Still a limitation: Human must drive each step manually.
"""
#
# ─── INSTALL REQUIRED PACKAGES ───────────────────────────────────────────────
# pip install google-genai python-dotenv
# ─────────────────────────────────────────────────────────────────────────────

import os                        # ✅ stdlib — no install needed
import json                      # ✅ stdlib — no install needed
import random                    # ✅ stdlib — no install needed
import datetime                  # ✅ stdlib — no install needed
from dataclasses import dataclass, field  # ✅ stdlib — no install needed
from typing import Optional      # ✅ stdlib — no install needed
# ⬆️ Python 3.10+ alternative: replace Optional[X] with X | None directly
#    e.g.  def foo(x: Optional[str]) -> None:
#          becomes: def foo(x: str | None) -> None:
from dotenv import load_dotenv   # 📦 python-dotenv
from google import genai         # 📦 google-genai
from google.genai import types   # 📦 google-genai (same package, sub-module)

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ─── Shared State: All agents read from and write to this ─────────────────────

@dataclass
class StudentProfile:
    """Shared memory that all agents can access and update."""
    name: str
    exam_date: str
    topics: list[str]
    study_hours_per_day: int = 2
    
    # Populated during the session
    study_plan: dict = field(default_factory=dict)
    topic_research: dict = field(default_factory=dict)  # topic → research data
    quiz_history: list[dict] = field(default_factory=list)
    weak_areas: list[str] = field(default_factory=list)
    strong_areas: list[str] = field(default_factory=list)
    daily_reports: list[dict] = field(default_factory=list)
    
    def add_quiz_result(self, topic: str, question: str, 
                         student_answer: str, is_correct: bool) -> None:
        """Record a quiz result and update weak/strong areas."""
        self.quiz_history.append({
            "date": datetime.datetime.now().isoformat(),
            "topic": topic,
            "question": question,
            "student_answer": student_answer,
            "is_correct": is_correct
        })
        
        # Update weak/strong areas
        topic_results = [r for r in self.quiz_history if r["topic"] == topic]
        correct = sum(1 for r in topic_results if r["is_correct"])
        accuracy = correct / len(topic_results) if topic_results else 0
        
        if accuracy < 0.6 and topic not in self.weak_areas:
            self.weak_areas.append(topic)
            if topic in self.strong_areas:
                self.strong_areas.remove(topic)
        elif accuracy >= 0.8 and topic not in self.strong_areas:
            self.strong_areas.append(topic)
            if topic in self.weak_areas:
                self.weak_areas.remove(topic)

    def get_overall_accuracy(self) -> float:
        if not self.quiz_history:
            return 0.0
        correct = sum(1 for r in self.quiz_history if r["is_correct"])
        return correct / len(self.quiz_history)

    def days_until_exam(self) -> int:
        exam = datetime.datetime.strptime(self.exam_date, "%Y-%m-%d").date()
        return (exam - datetime.date.today()).days


# ─── Specialized Agent Classes ────────────────────────────────────────────────

def _call_gemini(system_prompt: str, user_prompt: str,
                  as_json: bool = True, temperature: float = 0.3) -> str:
    """Helper to call Gemini. All agents use this.

    Args:
        system_prompt: The agent's role and instructions
        user_prompt:   The specific task for this call
        as_json:       If True, forces Gemini to return raw JSON (no markdown fences)
        temperature:   Creativity level (0.3 = accurate, deterministic)
    """
    # Build config dict — include response_mime_type only when JSON is needed.
    # Passing it in the constructor (not as a post-assignment) is the correct
    # pattern for google-genai SDK objects.
    config_kwargs: dict = {
        "system_instruction": system_prompt,
        "temperature": temperature,
    }
    if as_json:
        config_kwargs["response_mime_type"] = "application/json"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(**config_kwargs),
        contents=user_prompt
    )
    return response.text


class PlannerAgent:
    """
    AGENT 1: The Planner
    Responsibility: Creates/updates study schedules based on time and weak areas.
    Runs: At the start, and whenever the Orchestrator decides a re-plan is needed.
    """
    
    SYSTEM_PROMPT = """
    You are a CS exam study planner. You create realistic, achievable study plans.
    
    Consider:
    - Student's available days and hours per day
    - Topic complexity (DP > Graphs > BST > LinkedList)
    - Weak areas get MORE time
    - Build in review days (don't just cram new topics)
    - Day before exam: only light review, no new topics
    
    Return JSON with this structure:
    {
        "daily_schedule": [
            {
                "day": 1,
                "date": "YYYY-MM-DD",
                "primary_topic": "...",
                "secondary_topic": "..." or null,
                "goal": "What the student should be able to do",
                "hours_allocated": 2,
                "special_notes": "..." or null
            }
        ],
        "exam_readiness_strategy": "...",
        "risk_areas": ["topics likely to cause difficulty"]
    }
    """

    def create_or_update_plan(self, profile: StudentProfile) -> dict:
        print("  📅 [PlannerAgent] Creating study plan...")
        
        prompt = f"""
        Student: {profile.name}
        Exam date: {profile.exam_date}
        Days until exam: {profile.days_until_exam()}
        Topics to cover: {', '.join(profile.topics)}
        Study hours per day: {profile.study_hours_per_day}
        Current weak areas: {profile.weak_areas or 'None identified yet'}
        Current strong areas: {profile.strong_areas or 'None identified yet'}
        Today: {datetime.date.today().isoformat()}
        
        Create an optimal study plan. Weak areas need extra attention.
        """
        
        result = json.loads(_call_gemini(self.SYSTEM_PROMPT, prompt))
        profile.study_plan = result
        print(f"  ✅ [PlannerAgent] Plan created: {len(result.get('daily_schedule', []))} days scheduled")
        return result


class ResearchAgent:
    """
    AGENT 2: The Researcher
    Responsibility: Fetches and structures learning content for each topic.
    Runs: For each topic in the plan, before the student studies it.
    """
    
    SYSTEM_PROMPT = """
    You are a CS educator. You create comprehensive, beginner-friendly topic summaries.
    
    Return JSON with this structure:
    {
        "topic": "...",
        "one_line_definition": "...",
        "analogy": "...",
        "how_it_works": ["step 1", "step 2", ...],
        "key_concepts": {"concept_name": "explanation", ...},
        "python_example": "# Python code with comments\\ncode here",
        "time_complexity": {"operation": "O(...)", ...},
        "common_interview_questions": ["q1", "q2", "q3"],
        "pro_tips": ["...", "..."],
        "estimated_study_hours": 3
    }
    """
    
    def research_topic(self, topic: str, profile: StudentProfile) -> dict:
        print(f"  🔬 [ResearchAgent] Researching '{topic}'...")
        
        prompt = f"""
        Create a comprehensive study guide for: {topic}
        Target audience: College CS student (beginner/intermediate level)
        Student context: Studying for exam in {profile.days_until_exam()} days
        """
        
        result = json.loads(_call_gemini(self.SYSTEM_PROMPT, prompt))
        profile.topic_research[topic] = result
        print(f"  ✅ [ResearchAgent] '{topic}' researched: {len(result.get('key_concepts', {}))} concepts")
        return result


class QuizAgent:
    """
    AGENT 3: The Quiz Master
    Responsibility: Generates targeted quiz questions, especially for weak areas.
    Runs: Daily, after the student has studied.
    """
    
    SYSTEM_PROMPT = """
    You are a CS quiz creator for college exams. Create realistic exam-style questions.
    
    Return JSON with this structure:
    {
        "questions": [
            {
                "id": 1,
                "topic": "...",
                "question": "...",
                "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
                "correct_answer": "A",
                "explanation": "...",
                "difficulty": "easy|medium|hard",
                "concept_tested": "..."
            }
        ]
    }
    """

    def generate_quiz(self, topics: list[str], profile: StudentProfile, 
                       num_questions: int = 5) -> list[dict]:
        print(f"  ❓ [QuizAgent] Generating {num_questions} quiz questions...")
        
        # Bias toward weak areas
        weak = profile.weak_areas
        if weak:
            topics = weak + [t for t in topics if t not in weak]
        
        prompt = f"""
        Generate {num_questions} quiz questions covering these topics: {', '.join(topics[:3])}
        Weak areas (focus more here): {weak or 'None'}
        Student accuracy so far: {profile.get_overall_accuracy():.0%}
        Mix difficulty levels appropriately.
        """
        
        result = json.loads(_call_gemini(self.SYSTEM_PROMPT, prompt))
        questions = result.get("questions", [])
        print(f"  ✅ [QuizAgent] {len(questions)} questions ready")
        return questions


class EvaluatorAgent:
    """
    AGENT 4: The Evaluator
    Responsibility: Grades quiz answers, identifies weak spots, provides feedback.
    Runs: After the student completes a quiz.
    """
    
    SYSTEM_PROMPT = """
    You are a CS exam grader and coach. You give honest but encouraging feedback.
    
    Return JSON with this structure:
    {
        "overall_feedback": "...",
        "score": "X/Y",
        "percentage": 80,
        "question_feedback": [
            {
                "question_id": 1,
                "is_correct": true,
                "feedback": "...",
                "concept_to_review": "..." or null
            }
        ],
        "weak_concepts": ["..."],
        "strong_concepts": ["..."],
        "recommended_focus": "What to study next based on results"
    }
    """

    def evaluate_quiz(self, questions: list[dict], answers: dict, 
                       profile: StudentProfile) -> dict:
        print("  📊 [EvaluatorAgent] Grading quiz...")
        
        # Build evaluation context
        qa_pairs = []
        for q in questions:
            qid = q.get("id")
            student_ans = answers.get(str(qid), answers.get(qid, "Not answered"))
            qa_pairs.append({
                "id": qid,
                "question": q["question"],
                "correct_answer": q["correct_answer"],
                "student_answer": str(student_ans),
                "explanation": q["explanation"]
            })
        
        prompt = f"""
        Evaluate these quiz answers for student {profile.name}:
        {json.dumps(qa_pairs, indent=2)}
        
        Current weak areas: {profile.weak_areas}
        Provide detailed, encouraging feedback.
        """
        
        result = json.loads(_call_gemini(self.SYSTEM_PROMPT, prompt))
        
        # Update student profile with results
        for q in questions:
            qid = q.get("id")
            student_ans = str(answers.get(str(qid), answers.get(qid, "")))
            is_correct = student_ans.upper() == q["correct_answer"].upper()
            profile.add_quiz_result(
                topic=q.get("topic", "Unknown"),
                question=q["question"],
                student_answer=student_ans,
                is_correct=is_correct
            )
        
        print(f"  ✅ [EvaluatorAgent] Grade: {result.get('score', 'N/A')} ({result.get('percentage', 0)}%)")
        return result


class ReportAgent:
    """
    AGENT 5: The Daily Reporter
    Responsibility: Generates human-readable daily progress reports.
    Runs: At end of each study session. Student reads it at their leisure.
    """
    
    def generate_daily_report(self, profile: StudentProfile, 
                               eval_result: dict, day_number: int) -> str:
        print("  📝 [ReportAgent] Writing daily report...")
        
        report_date = datetime.date.today().strftime("%B %d, %Y")
        accuracy = profile.get_overall_accuracy()
        days_left = profile.days_until_exam()
        
        report = f"""
# 📊 StudyBuddy Daily Report — Day {day_number}
**Student:** {profile.name} | **Date:** {report_date} | **Days Until Exam:** {days_left}

---

## 🎯 Today's Quiz Results
- **Score:** {eval_result.get('score', 'N/A')}
- **Overall Accuracy (All Time):** {accuracy:.0%}

### Feedback
> {eval_result.get('overall_feedback', 'Good effort today!')}

---

## 💪 Your Strengths
{chr(10).join(f'- ✅ {s}' for s in profile.strong_areas) or '- Keep going — strengths are forming!'}

## ⚠️ Areas Needing More Practice
{chr(10).join(f'- 🔄 {w}' for w in profile.weak_areas) or '- None identified — great job!'}

---

## 🗓️ Tomorrow's Plan
{eval_result.get('recommended_focus', 'Continue with the current plan.')}

---

## 📈 Overall Progress
- Topics researched: {len(profile.topic_research)}/{len(profile.topics)}
- Total quiz questions answered: {len(profile.quiz_history)}
- Study plan: {len(profile.study_plan.get('daily_schedule', []))} days scheduled

---
*Report generated automatically by StudyBuddy v3.0 🤖*
"""
        
        # Save report to file
        reports_dir = pathlib.Path("study_reports")
        reports_dir.mkdir(exist_ok=True)
        report_path = reports_dir / f"day_{day_number:02d}_{profile.name.lower()}_report.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"  ✅ [ReportAgent] Report saved: {report_path}")
        
        # Add to profile
        profile.daily_reports.append({
            "day": day_number,
            "date": report_date,
            "score": eval_result.get('score'),
            "accuracy": f"{accuracy:.0%}",
            "report_file": str(report_path)
        })
        
        return report


# ─── THE ORCHESTRATOR ─────────────────────────────────────────────────────────

class AutonomousExamCoach:
    """
    StudyBuddy v3.0 — The Autonomous Exam Coach
    
    This is the AGENTIC AI system. The student sets a high-level goal,
    and the Orchestrator runs everything autonomously.
    
    Architecture:
    - Orchestrator decides WHAT to do and WHEN to do it
    - 5 specialized agents handle their specific domains
    - All agents share a StudentProfile for state management
    - Runs for multiple sessions with minimal human input
    """

    def __init__(self):
        print("🚀 Initializing AutonomousExamCoach (StudyBuddy v3.0)...")
        self.planner = PlannerAgent()
        self.researcher = ResearchAgent()
        self.quiz_master = QuizAgent()
        self.evaluator = EvaluatorAgent()
        self.reporter = ReportAgent()
        print("✅ All 5 agents initialized and connected.")

    def onboard_student(
        self,
        name: str,
        exam_date: str,
        topics: list[str],
        hours_per_day: int = 2
    ) -> StudentProfile:
        """
        STEP 1 (one-time): Student provides their goal and context.
        From here, the system runs autonomously.
        """
        print(f"\n{'='*60}")
        print(f"  🎓 ONBOARDING: {name}")
        print(f"{'='*60}")
        
        profile = StudentProfile(
            name=name,
            exam_date=exam_date,
            topics=topics,
            study_hours_per_day=hours_per_day
        )
        
        days_left = profile.days_until_exam()
        print(f"  📅 Exam: {exam_date} ({days_left} days away)")
        print(f"  📚 Topics: {', '.join(topics)}")
        print(f"  ⏱️  Study time: {hours_per_day} hrs/day")
        
        return profile

    def run_autonomous_session(
        self,
        profile: StudentProfile,
        day_number: int = 1,
        simulate_answers: bool = False  # Set True for demo/testing
    ) -> dict:
        """
        Run a full autonomous study session for one day.
        
        The Orchestrator:
        1. Determines today's focus from the plan
        2. Triggers research if topic is new
        3. Generates targeted quiz
        4. Collects answers (or simulates them)
        5. Evaluates performance
        6. Adapts the plan based on performance
        7. Generates daily report
        
        All AUTONOMOUSLY — no step-by-step user prompting!
        """
        print(f"\n{'='*60}")
        print(f"  🤖 ORCHESTRATOR: Starting Day {day_number} session")
        print(f"{'='*60}")
        
        # ── AUTONOMOUS DECISION 1: Create or retrieve study plan ──
        if not profile.study_plan:
            print("\n📋 [Orchestrator] No plan exists. Delegating to PlannerAgent...")
            self.planner.create_or_update_plan(profile)
        
        # ── AUTONOMOUS DECISION 2: Determine today's focus ────────
        schedule = profile.study_plan.get("daily_schedule", [])
        today_schedule = schedule[day_number - 1] if day_number <= len(schedule) else None
        
        if today_schedule:
            today_topic = today_schedule["primary_topic"]
            print(f"\n🎯 [Orchestrator] Today's focus: {today_topic}")
            print(f"   Goal: {today_schedule.get('goal', 'Master the basics')}")
        else:
            # Orchestrator adapts — focuses on weakest areas if no schedule
            today_topic = profile.weak_areas[0] if profile.weak_areas else profile.topics[0]
            print(f"\n🎯 [Orchestrator] Adaptive focus on weak area: {today_topic}")
        
        # ── AUTONOMOUS DECISION 3: Research if not yet done ───────
        if today_topic not in profile.topic_research:
            print(f"\n🔬 [Orchestrator] '{today_topic}' not researched. Delegating to ResearchAgent...")
            self.researcher.research_topic(today_topic, profile)
        else:
            print(f"\n📚 [Orchestrator] '{today_topic}' already in knowledge base. Skipping research.")
        
        # ── AUTONOMOUS DECISION 4: Generate targeted quiz ─────────
        print("\n❓ [Orchestrator] Delegating to QuizAgent for daily quiz...")
        quiz_topics = [today_topic] + profile.weak_areas[:2]
        questions = self.quiz_master.generate_quiz(quiz_topics, profile, num_questions=3)
        
        # ── INTERACT: Present quiz to student (or simulate) ───────
        answers = {}
        if simulate_answers:
            # For demo purposes — simulate student answers
            print("\n📝 [DEMO] Simulating student answering quiz...")
            for q in questions:
                # Simulate ~70% accuracy
                import random
                if random.random() < 0.7:
                    answers[str(q["id"])] = q["correct_answer"]
                else:
                    # Pick a wrong answer
                    wrong_options = [k for k in ["A", "B", "C", "D"] 
                                   if k != q["correct_answer"]]
                    answers[str(q["id"])] = random.choice(wrong_options)
                print(f"   Q{q['id']}: Student answered '{answers[str(q['id'])]}'")
        else:
            # Real interactive mode
            print("\n📝 QUIZ TIME! Answer the following questions:")
            for q in questions:
                print(f"\n  Question {q['id']}: {q['question']}")
                for opt, text in q.get("options", {}).items():
                    print(f"    {opt}) {text}")
                answer = input("  Your answer (A/B/C/D): ").strip().upper()
                answers[str(q["id"])] = answer
        
        # ── AUTONOMOUS DECISION 5: Evaluate & adapt ───────────────
        print("\n📊 [Orchestrator] Delegating to EvaluatorAgent...")
        eval_result = self.evaluator.evaluate_quiz(questions, answers, profile)
        
        # ── AUTONOMOUS DECISION 6: Re-plan if needed ──────────────
        if profile.weak_areas and day_number % 2 == 0:
            print("\n🔄 [Orchestrator] Weak areas detected. Triggering re-plan...")
            self.planner.create_or_update_plan(profile)
        
        # ── AUTONOMOUS DECISION 7: Generate & save daily report ───
        print("\n📝 [Orchestrator] Delegating to ReportAgent...")
        report = self.reporter.generate_daily_report(profile, eval_result, day_number)
        
        print(f"\n✅ [Orchestrator] Day {day_number} complete!")
        print(f"   Overall accuracy: {profile.get_overall_accuracy():.0%}")
        print(f"   Weak areas: {profile.weak_areas or 'None'}")
        print(f"   Days until exam: {profile.days_until_exam()}")
        
        return {
            "day": day_number,
            "topic_studied": today_topic,
            "quiz_score": eval_result.get("score"),
            "accuracy": profile.get_overall_accuracy(),
            "report_preview": report[:300] + "..."
        }

    def run_exam_prep_program(
        self,
        profile: StudentProfile,
        num_sessions: int = 3,
        simulate: bool = True
    ) -> None:
        """
        Run the full multi-day exam prep program AUTONOMOUSLY.
        
        The student just calls this once and the system handles everything.
        This is the key difference from Day 1 and Day 2!
        """
        print(f"\n{'='*60}")
        print(f"  🚀 AUTONOMOUS EXAM PREP STARTING")
        print(f"  Student: {profile.name}")
        print(f"  Sessions to run: {num_sessions}")
        print(f"  Mode: {'Simulation' if simulate else 'Interactive'}")
        print(f"{'='*60}")
        
        results = []
        for day in range(1, num_sessions + 1):
            session_result = self.run_autonomous_session(
                profile=profile,
                day_number=day,
                simulate_answers=simulate
            )
            results.append(session_result)

            if day < num_sessions:
                # ⚠️ Rate limit protection: the Gemini free tier allows ~15 req/min.
                # Each session makes several API calls (plan + research + quiz + eval + report).
                # A 10-second pause between full sessions keeps us well within limits.
                print(f"\n⏳ [Orchestrator] Session {day} done. Pausing 10s before next session...")
                import time
                time.sleep(10)
        
        # Final autonomously-generated summary
        print(f"\n{'='*60}")
        print(f"  🎓 PROGRAM COMPLETE: {profile.name}'s Exam Prep Summary")
        print(f"{'='*60}")
        print(f"  Sessions completed: {len(results)}")
        print(f"  Final accuracy: {profile.get_overall_accuracy():.0%}")
        print(f"  Topics mastered: {profile.strong_areas or 'Still building...'}")
        print(f"  Areas to review: {profile.weak_areas or 'All looking good!'}")
        print(f"  Reports saved to: ./study_reports/")
        print(f"\n  📬 StudyBuddy will continue sending reports until your exam.")
        print(f"  Good luck on {profile.exam_date}! 🍀")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Create the autonomous coach
    coach = AutonomousExamCoach()

    # 2. Student provides their one-time goal
    profile = coach.onboard_student(
        name="Alex",
        exam_date="2026-06-20",
        topics=["Binary Search Tree", "Dynamic Programming", "Graph Algorithms", "Sorting"],
        hours_per_day=2
    )

    # 3. The system runs AUTONOMOUSLY from here!
    #    Student doesn't need to ask for anything else.
    coach.run_exam_prep_program(
        profile=profile,
        num_sessions=3,    # Run 3 study sessions
        simulate=True      # Set False for real interactive mode
    )
```

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

```bash
# ─── INSTALL REQUIRED PACKAGES ────────────────────────────────────────────
# pip install chromadb google-genai python-dotenv
#
# chromadb  → open-source local vector database (runs on your machine)
# google-genai  → Gemini API for generation
# ─────────────────────────────────────────────────────────────────────────

# Run this demo with:
# python bonus_vectordb_rag_demo.py
```

```python
# bonus_vectordb_rag_demo.py
#
# ─── INSTALL REQUIRED PACKAGES ────────────────────────────────────────────
# pip install chromadb google-genai python-dotenv
# ─────────────────────────────────────────────────────────────────────────
#
# What this demo shows:
#   PART 1: Store CS study notes in ChromaDB (a local VectorDB)
#   PART 2: Simple semantic search — find relevant notes for any query
#   PART 3: RAG — use retrieved notes as context for Gemini answers
#   PART 4: Compare RAG answer vs. plain Gemini answer

import os                        # ✅ stdlib — no install needed
import json                      # ✅ stdlib — no install needed
from dotenv import load_dotenv   # 📦 python-dotenv
from google import genai         # 📦 google-genai
import chromadb                  # 📦 chromadb (local vector DB — no cloud needed!)
from chromadb.utils import embedding_functions  # 📦 chromadb

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ─── PART 1: Build the Knowledge Base in ChromaDB ────────────────────────────

# ChromaDB stores data in a "collection" (like a table in SQL).
# PersistentClient saves data to disk — it survives script restarts.
# Client() (no args) is in-memory only — faster for demos.

chroma_client = chromadb.Client()   # in-memory for this demo
# For persistence across runs, use:
# chroma_client = chromadb.PersistentClient(path="./chroma_study_db")

# ChromaDB can use its own built-in embedding model (all-MiniLM-L6-v2)
# This converts text to vectors automatically — no extra API calls needed!
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.create_collection(
    name="cs_study_notes",
    embedding_function=embedding_fn,
    metadata={"description": "CS study materials for StudyBuddy RAG demo"}
)

# ─── Our knowledge base: bite-sized study notes about CS topics ───────────
# In a real system these would come from PDFs, textbooks, or lecture slides.
CS_STUDY_CHUNKS = [
    # Binary Search Tree
    {
        "id": "bst_definition",
        "text": "A Binary Search Tree (BST) is a tree data structure where each node has "
                "at most two children. For any node, all values in its left subtree are "
                "smaller than the node's value, and all values in its right subtree are larger.",
        "topic": "BST",
        "type": "definition"
    },
    {
        "id": "bst_complexity",
        "text": "BST time complexity: Search, Insert, and Delete are O(log n) on average "
                "for a balanced tree. In the worst case (completely unbalanced, like a linked list), "
                "all operations degrade to O(n). Space complexity is O(n) for storing n nodes.",
        "topic": "BST",
        "type": "complexity"
    },
    {
        "id": "bst_insertion",
        "text": "To insert into a BST: start at the root. If the new value is less than "
                "the current node, go left; if greater, go right. Repeat until you find "
                "an empty slot (None), then place the new node there.",
        "topic": "BST",
        "type": "algorithm"
    },
    {
        "id": "bst_traversal",
        "text": "BST traversal types: In-order (left → root → right) gives sorted output. "
                "Pre-order (root → left → right) is useful for copying trees. "
                "Post-order (left → right → root) is useful for deleting trees.",
        "topic": "BST",
        "type": "algorithm"
    },
    # Dynamic Programming
    {
        "id": "dp_definition",
        "text": "Dynamic Programming (DP) is a technique for solving problems by breaking "
                "them into overlapping subproblems and storing results to avoid recomputation. "
                "It requires two properties: optimal substructure and overlapping subproblems.",
        "topic": "Dynamic Programming",
        "type": "definition"
    },
    {
        "id": "dp_vs_recursion",
        "text": "The difference between recursion and dynamic programming: pure recursion "
                "recomputes the same subproblems repeatedly (exponential time). DP stores "
                "results in a memo table (memoization) or builds bottom-up (tabulation), "
                "reducing time to O(n) or O(n²) in most cases.",
        "topic": "Dynamic Programming",
        "type": "comparison"
    },
    # Sorting
    {
        "id": "merge_sort_complexity",
        "text": "Merge sort time complexity is O(n log n) in all cases (best, average, worst). "
                "It is a stable, divide-and-conquer algorithm. Space complexity is O(n) "
                "because it requires auxiliary space for merging. Preferred when stability matters.",
        "topic": "Sorting",
        "type": "complexity"
    },
    {
        "id": "quick_sort_complexity",
        "text": "Quick sort average time complexity is O(n log n) but worst case is O(n²) "
                "when pivot is always the smallest or largest element. In practice it is "
                "faster than merge sort because of better cache performance and O(log n) "
                "space (in-place). Not stable.",
        "topic": "Sorting",
        "type": "complexity"
    },
]

# Add all chunks to ChromaDB — it automatically generates embeddings!
print("📚 Loading study materials into ChromaDB...")
collection.add(
    documents=[chunk["text"] for chunk in CS_STUDY_CHUNKS],
    ids=[chunk["id"] for chunk in CS_STUDY_CHUNKS],
    metadatas=[
        {"topic": chunk["topic"], "type": chunk["type"]}
        for chunk in CS_STUDY_CHUNKS
    ]
)
print(f"✅ {len(CS_STUDY_CHUNKS)} study chunks stored in VectorDB\n")


# ─── PART 2: Semantic Search — Find Relevant Notes ───────────────────────────

def search_knowledge_base(query: str, n_results: int = 3, topic_filter: str | None = None) -> list[dict]:
    """
    Search ChromaDB for the most relevant study chunks.

    ChromaDB converts the query to a vector and finds the closest chunks
    using cosine similarity — this is SEMANTIC search, not keyword search!

    Args:
        query: The question or topic to search for
        n_results: How many chunks to retrieve
        topic_filter: Optional — restrict search to one topic (e.g., "BST")

    Returns:
        List of dicts with 'id', 'text', 'metadata', 'distance'
    """
    where_filter = {"topic": topic_filter} if topic_filter else None

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )

    # Reformat into clean list of results
    retrieved = []
    for i, doc_id in enumerate(results["ids"][0]):
        # results["distances"][0][i] is the "distance" in vector space.
        # ChromaDB uses cosine distance by default:
        #   distance = 0.0  → identical meaning (perfect match)
        #   distance = 1.0  → completely unrelated
        # We convert to similarity (1 - distance) so higher = more relevant.
        retrieved.append({
            "id": doc_id,
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "similarity": 1 - results["distances"][0][i],  # higher = more similar
        })

    return retrieved


# Demo: semantic search
print("=" * 60)
print("  🔍 DEMO 1: Semantic Search")
print("=" * 60)

test_queries = [
    "how fast is tree lookup?",                          # Should find bst_complexity
    "what is the difference between dp and recursion?",  # Should find dp_vs_recursion
    "which sort is better for nearly sorted data?",      # Should find sort complexities
]

for query in test_queries:
    print(f"\n🔎 Query: '{query}'")
    results = search_knowledge_base(query, n_results=2)
    for r in results:
        print(f"   ✅ [{r['id']}] (similarity={r['similarity']:.2f})")
        print(f"      {r['text'][:100]}...")


# ─── PART 3: RAG — Use Retrieved Context in Gemini Prompt ────────────────────

def answer_with_rag(question: str, n_context_chunks: int = 3) -> str:
    """
    RAG pipeline: Retrieve → Augment → Generate.

    Steps:
    1. Search VectorDB for the most relevant study chunks
    2. Build an augmented prompt: [context chunks] + [question]
    3. Gemini answers using the retrieved context — not just training memory

    This dramatically reduces hallucinations and gives fresh, specific answers.
    """
    print(f"\n❓ Question: {question}")
    print("   Step 1: Searching VectorDB for relevant context...")

    # RETRIEVE: find relevant chunks
    retrieved_chunks = search_knowledge_base(question, n_results=n_context_chunks)

    if not retrieved_chunks:
        return "No relevant context found in the knowledge base."

    # Show what was retrieved
    for i, chunk in enumerate(retrieved_chunks):
        print(f"   📄 Retrieved chunk {i+1}: [{chunk['id']}] (similarity={chunk['similarity']:.2f})")

    # AUGMENT: build the context block
    context_text = "\n\n".join(
        f"Source [{chunk['id']}]:\n{chunk['text']}"
        for chunk in retrieved_chunks
    )

    # Build the RAG prompt — ONLY the retrieved chunks, not the full knowledge base!
    rag_prompt = f"""You are StudyBuddy, a CS tutor. Answer the question using ONLY 
the provided study notes. If the answer isn't in the notes, say so clearly.

=== STUDY NOTES (from VectorDB) ===
{context_text}

=== STUDENT QUESTION ===
{question}

=== YOUR ANSWER ===
Be concise, accurate, and cite which source(s) you used."""

    print("   Step 2: Generating answer using retrieved context...")

    # GENERATE: Gemini uses the context to answer
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=rag_prompt
    )

    return response.text


# ─── PART 4: Compare RAG answer vs. Plain Gemini ────────────────────────────────────

def answer_without_rag(question: str) -> str:
    """Plain Gemini call — no retrieval, no custom context."""
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=(
            f"You are a CS tutor. Answer this question: {question}\n"
            "Keep your answer to 3-4 sentences."
        )
    )
    return response.text


def compare_rag_vs_plain(question: str) -> None:
    """
    Side-by-side comparison showing how RAG improves answers.
    
    Key difference:
    - Without RAG: Gemini uses general training knowledge (may hallucinate or be generic)
    - With RAG:    Gemini uses YOUR specific study notes (accurate, grounded)
    """
    print("\n" + "=" * 70)
    print(f"  📊 COMPARISON: RAG vs Plain Gemini")
    print(f"  Question: {question}")
    print("=" * 70)

    print("\n❌ WITHOUT RAG (plain Gemini):")
    print("-" * 40)
    plain_answer = answer_without_rag(question)
    print(plain_answer)

    print("\n✅ WITH RAG (using our VectorDB knowledge base):")
    print("-" * 40)
    rag_answer = answer_with_rag(question)
    print(rag_answer)

    print("\n💡 Notice: The RAG answer cites specific sources and is grounded")
    print("   in YOUR study materials, not just generic AI knowledge.")


# ─── Advanced: Metadata Filtering ────────────────────────────────────────────

def search_by_type(topic: str, chunk_type: str) -> None:
    """
    ChromaDB supports metadata filtering — find chunks by type.
    e.g., get only 'complexity' chunks for BST (not definitions or algorithms)
    This is useful when you want targeted retrieval.
    """
    print(f"\n🔍 Filtered search: topic='{topic}', type='{chunk_type}'")
    results = collection.query(
        query_texts=[f"{topic} {chunk_type}"],
        n_results=3,
        where={"$and": [{"topic": topic}, {"type": chunk_type}]}
    )
    for i, doc in enumerate(results["documents"][0]):
        print(f"  [{i+1}] {doc[:120]}...")


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Demo 1: Semantic search
    # (already shown above during setup)

    # Demo 2: RAG answer vs plain Gemini
    compare_rag_vs_plain(
        "What is the time complexity of BST operations and why does it matter?"
    )

    # Demo 3: Metadata filtering
    search_by_type("BST", "algorithm")
    search_by_type("Sorting", "complexity")

    # Try your own questions!
    print("\n\n" + "=" * 60)
    print("  💬 Ask your own question:")
    print("=" * 60)
    user_q = input("Your question: ").strip()
    if user_q:
        rag_answer = answer_with_rag(user_q)
        print(f"\n🤖 StudyBuddy (RAG answer):\n{rag_answer}")
```

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

```python
# level2_web_search_agent.py
#
# ─── INSTALL REQUIRED PACKAGES ───────────────────────────────────────────────
# pip install duckduckgo-search google-genai python-dotenv
#
# duckduckgo-search  → real web search, FREE, no API key needed
# ─────────────────────────────────────────────────────────────────────────────
#
# Run: python level2_web_search_agent.py

import os                             # ✅ stdlib
import datetime                       # ✅ stdlib
from dotenv import load_dotenv        # 📦 python-dotenv
from google import genai              # 📦 google-genai
from google.genai import types        # 📦 google-genai
from duckduckgo_search import DDGS    # 📦 duckduckgo-search

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ─── Tool 1: Real Web Search ─────────────────────────────────────────────────

def web_search(query: str, max_results: int = 3) -> str:
    """
    Search the live web using DuckDuckGo — completely free, no API key needed.

    Args:
        query:       The search query (e.g., "binary search tree tutorial 2026")
        max_results: How many results to return (default 3 to keep prompt small)

    Returns:
        Formatted string of results with titles, URLs, and summaries

    Example:
        web_search("BST insertion Python") →
        "Result 1:
           Title: Binary Search Tree - GeeksForGeeks
           URL: https://www.geeksforgeeks.org/...
           Summary: A BST is a node-based binary tree..."
    """
    try:
        with DDGS() as ddgs:
            # text() returns list of dicts: {title, href, body}
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"No web results found for: '{query}'. Try a different query."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"Result {i}:\n"
                f"  Title: {r.get('title', 'Untitled')}\n"
                f"  URL: {r.get('href', 'No URL')}\n"
                f"  Summary: {r.get('body', 'No summary')[:250]}..."
            )
        return "\n\n".join(formatted)

    except Exception as e:
        # Graceful failure — agent still works if search is temporarily unavailable
        return f"Web search unavailable ({e}). Using AI training knowledge instead."


# ─── Tool 2: Get Today's Date ─────────────────────────────────────────────────

def get_current_date() -> str:
    """Returns today's date — useful for exam countdown."""
    return datetime.datetime.now().strftime("%A, %B %d, %Y")


# ─── Tool 3: Days until exam ──────────────────────────────────────────────────

def days_until_exam(exam_date: str) -> str:
    """
    Calculate days remaining until exam.
    exam_date: 'YYYY-MM-DD' format
    """
    try:
        exam = datetime.datetime.strptime(exam_date, "%Y-%m-%d").date()
        remaining = (exam - datetime.date.today()).days
        if remaining < 0:
            return f"That exam was {abs(remaining)} days ago."
        if remaining == 0:
            return "The exam is TODAY! Good luck! 🍀"
        return f"You have {remaining} days until your exam on {exam_date}."
    except ValueError:
        return "Invalid date format. Please use YYYY-MM-DD (e.g. 2026-06-20)."


# ─── Register tools with Gemini ──────────────────────────────────────────────

TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="web_search",
        description=(
            "Search the live web for current information on any topic. "
            "Use when the student asks for tutorials, resources, recent examples, "
            "or anything that benefits from up-to-date information."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(
                    type=types.Type.STRING,
                    description="Specific search query. E.g. 'BST insertion Python tutorial 2026'"
                ),
                "max_results": types.Schema(
                    type=types.Type.INTEGER,
                    description="Number of results (1-5). Default 3."
                )
            },
            required=["query"]
        )
    ),
    types.FunctionDeclaration(
        name="get_current_date",
        description="Get today's date. Use when student asks about schedules or timing.",
        parameters=types.Schema(type=types.Type.OBJECT, properties={})
    ),
    types.FunctionDeclaration(
        name="days_until_exam",
        description="Calculate days remaining until an exam date.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "exam_date": types.Schema(
                    type=types.Type.STRING,
                    description="Exam date in YYYY-MM-DD format"
                )
            },
            required=["exam_date"]
        )
    )
])

TOOL_MAP = {
    "web_search": web_search,
    "get_current_date": get_current_date,
    "days_until_exam": days_until_exam,
}

SYSTEM_PROMPT = """
You are StudyBuddy v3.6, a CS tutor with live web search capability.

When a student asks about a CS topic or wants learning resources:
1. ALWAYS call web_search() first to get fresh, real-world resources
2. Summarize the results clearly with source URLs
3. Add your own explanation to supplement the web results

For time/exam questions, use the date tools for accurate answers.
Be encouraging and specific — give actionable, sourced advice.
"""


def run_web_search_agent(user_message: str, history: list | None = None) -> tuple[str, list]:
    """
    Run the web-search-enabled StudyBuddy agent.

    Loop: send message → if tool call → execute → send result → repeat → final answer

    Returns: (answer_text, updated_history)
    """
    if history is None:
        history = []

    history.append({"role": "user", "parts": [{"text": user_message}]})
    print(f"\n👤 Student: {user_message}")

    for _ in range(6):  # max 6 tool calls per message
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[TOOLS],
                temperature=0.4
            ),
            contents=history
        )

        candidate = response.candidates[0]
        tool_calls = [
            p for p in candidate.content.parts
            if hasattr(p, "function_call") and p.function_call
        ]

        if not tool_calls:
            # Gemini gave a final text answer — done!
            answer = response.text
            history.append({"role": "model", "parts": [{"text": answer}]})
            return answer, history

        # Execute each tool and collect results
        history.append({"role": "model", "parts": candidate.content.parts})
        tool_results = []
        for tc in tool_calls:
            fn_name = tc.function_call.name
            fn_args = dict(tc.function_call.args) if tc.function_call.args else {}
            print(f"  🔧 Calling: {fn_name}({fn_args})")
            result = TOOL_MAP[fn_name](**fn_args)
            print(f"  📤 Result: {str(result)[:100]}...")
            tool_results.append(
                types.Part(function_response=types.FunctionResponse(
                    name=fn_name, response={"result": result}
                ))
            )
        history.append({"role": "user", "parts": tool_results})

    return "Agent exceeded maximum iterations.", history


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 64)
    print("  🌐 StudyBuddy v3.6 — With Live Web Search")
    print("=" * 64)

    history: list = []
    demo_questions = [
        "Find me the best beginner tutorials for Binary Search Trees.",
        "My exam is 2026-06-20. How many days do I have to prepare?",
        "Search for common BST interview questions asked at top tech companies.",
    ]

    for question in demo_questions:
        answer, history = run_web_search_agent(question, history)
        print(f"\n🤖 StudyBuddy: {answer[:300]}...")
        print("─" * 60)
```

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

```python
# level2_studybuddy_with_rag.py
#
# ─── INSTALL REQUIRED PACKAGES ───────────────────────────────────────────────
# pip install chromadb google-genai python-dotenv
# ─────────────────────────────────────────────────────────────────────────────
#
# Run: python level2_studybuddy_with_rag.py

import os                        # ✅ stdlib
import datetime                  # ✅ stdlib
from dotenv import load_dotenv   # 📦 python-dotenv
from google import genai         # 📦 google-genai
from google.genai import types   # 📦 google-genai
import chromadb                  # 📦 chromadb
from chromadb.utils import embedding_functions  # 📦 chromadb

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ─── Persistent ChromaDB (survives restarts) ──────────────────────────────────
chroma = chromadb.PersistentClient(path="./studybuddy_knowledge_db")
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma.get_or_create_collection("study_knowledge", embedding_function=embedding_fn)
print(f"📚 Knowledge base loaded: {collection.count()} existing chunks")

# ─── Seed starter knowledge ───────────────────────────────────────────────────
SEED_CHUNKS = [
    {"id": "bst_core", "text": "A BST is a binary tree where left < node < right. "
     "O(log n) avg for search/insert/delete. O(n) worst case when unbalanced. "
     "In-order traversal gives sorted output.", "topic": "BST"},
    {"id": "dp_core", "text": "Dynamic Programming solves problems by caching overlapping "
     "subproblem results. Approaches: top-down memoization or bottom-up tabulation. "
     "Requires: optimal substructure + overlapping subproblems.", "topic": "DP"},
]

existing_ids = set(collection.get()["ids"])
new_chunks = [c for c in SEED_CHUNKS if c["id"] not in existing_ids]
if new_chunks:
    collection.add(
        documents=[c["text"] for c in new_chunks],
        ids=[c["id"] for c in new_chunks],
        metadatas=[{"topic": c["topic"]} for c in new_chunks]
    )
    print(f"✅ Seeded {len(new_chunks)} starter chunks")


# ─── Upgraded ResearchAgent: VectorDB first, Gemini fallback, auto-store ──────

def research_topic_with_rag(topic: str) -> dict:
    """
    3-step RAG research loop:
    ① Search ChromaDB for existing knowledge
    ② Generate via Gemini (with DB context if found, from scratch if not)
    ③ Store new content back in ChromaDB for next time

    The knowledge base GROWS automatically with each new topic query!
    """
    print(f"\n🔬 Researching: '{topic}'")

    # ── Step ①: Query VectorDB ────────────────────────────────────────────────
    results = collection.query(query_texts=[f"explain {topic}"], n_results=2)
    has_hit = bool(results["ids"][0])

    # Cosine distance: 0.0 = identical, 2.0 = unrelated. Threshold 0.8 = relevant.
    is_relevant = has_hit and results["distances"][0][0] < 0.8
    context = "\n\n".join(results["documents"][0]) if is_relevant else ""

    if is_relevant:
        sim = 1 - results["distances"][0][0]
        print(f"  📚 VectorDB hit (similarity={sim:.2f}) — using cached knowledge")
    else:
        print(f"  🌐 No relevant cache — generating via Gemini")

    # ── Step ②: Generate content ──────────────────────────────────────────────
    if context:
        prompt = (f"Using this existing knowledge as reference:\n{context}\n\n"
                  f"Expand into a complete study guide for: {topic} (max 250 words)")
    else:
        prompt = (f"Create a concise study guide for college CS students on: {topic}\n"
                  f"Include: definition, analogy, key operations, time complexity. Max 250 words.")

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=500)
    )
    content = response.text

    # ── Step ③: Store result if new ───────────────────────────────────────────
    if not is_relevant:
        new_id = f"{topic.lower().replace(' ', '_')}_{datetime.datetime.now().strftime('%H%M%S')}"
        collection.add(
            documents=[content], ids=[new_id],
            metadatas=[{"topic": topic, "source": "gemini"}]
        )
        print(f"  💾 Stored in ChromaDB (id={new_id}, total={collection.count()} chunks)")

    return {"topic": topic, "content": content, "db_size": collection.count()}


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🧠 ResearchAgent + RAG Demo")
    print(f"  Starting DB size: {collection.count()} chunks")

    for topic in ["Binary Search Tree", "Heap Data Structure", "Binary Search Tree"]:
        result = research_topic_with_rag(topic)
        print(f"\n  📄 Preview: {result['content'][:100]}...")
        print(f"  📊 DB size: {result['db_size']} chunks")

    print(f"\n✅ Restart the script — BST queries now use the cached DB result!")
```

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

```python
# level2_langgraph_studybuddy.py
#
# ─── INSTALL REQUIRED PACKAGES ───────────────────────────────────────────────
# pip install langgraph google-genai python-dotenv
#
# langgraph  → stateful multi-agent graph framework
# ─────────────────────────────────────────────────────────────────────────────
#
# Run: python level2_langgraph_studybuddy.py

import os                        # ✅ stdlib
import json                      # ✅ stdlib
import random                    # ✅ stdlib
from typing import TypedDict     # ✅ stdlib
from dotenv import load_dotenv   # 📦 python-dotenv
from google import genai         # 📦 google-genai
from google.genai import types   # 📦 google-genai
from langgraph.graph import StateGraph, END  # 📦 langgraph

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ─── Shared state TypedDict ───────────────────────────────────────────────────
# Every node receives the full state and returns ONLY what it changed.

class StudyState(TypedDict):
    student_name: str
    topic: str
    exam_date: str
    max_sessions: int
    # Updated by nodes:
    study_plan: str
    research_content: str
    quiz_questions: list
    student_answers: dict
    score: int
    weak_areas: list[str]
    feedback: str
    session_count: int


# ─── Node functions (one per agent) ──────────────────────────────────────────

def plan_node(state: StudyState) -> dict:
    """Creates study plan. Only returns {'study_plan': ...}"""
    print(f"  📅 [PlannerNode] Planning for '{state['topic']}'...")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(temperature=0.3),
        contents=(
            f"Create a 3-bullet study plan for {state['student_name']} to master "
            f"'{state['topic']}' by {state['exam_date']}. "
            f"Weak areas to prioritise: {state.get('weak_areas', [])}."
        )
    )
    print(f"  ✅ Plan ready")
    return {"study_plan": response.text}


def research_node(state: StudyState) -> dict:
    """Generates study content. Only returns {'research_content': ...}"""
    print(f"  🔬 [ResearchNode] Researching '{state['topic']}'...")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=400),
        contents=(
            f"Concise study guide for college students on: {state['topic']}\n"
            f"Include: definition, analogy, operations, time complexity. Max 200 words."
        )
    )
    print(f"  ✅ Research done")
    return {"research_content": response.text}


def quiz_node(state: StudyState) -> dict:
    """Generates quiz questions + simulates answers. Returns {'quiz_questions':..., 'student_answers':...}"""
    print(f"  ❓ [QuizNode] Generating quiz for '{state['topic']}'...")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            temperature=0.4,
            response_mime_type="application/json"
        ),
        contents=(
            f"Generate 3 MCQ questions on {state['topic']}. "
            f'JSON array: [{{"id":1,"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},'
            f'"correct":"A","concept":"..."}}]'
        )
    )
    try:
        questions = json.loads(response.text)
    except json.JSONDecodeError:
        questions = []

    # Simulate 70% accuracy for demo (replace with input() for real use)
    answers = {}
    for q in questions:
        qid = str(q.get("id", 1))
        answers[qid] = (
            q.get("correct", "A") if random.random() < 0.7
            else random.choice([k for k in "ABCD" if k != q.get("correct", "A")])
        )

    print(f"  ✅ {len(questions)} questions, answers simulated")
    return {"quiz_questions": questions, "student_answers": answers}


def evaluate_node(state: StudyState) -> dict:
    """Grades answers, identifies weak spots. Returns score, weak_areas, feedback, session_count."""
    print(f"  📊 [EvaluateNode] Grading...")
    questions = state.get("quiz_questions", [])
    answers = state.get("student_answers", {})

    correct = sum(
        1 for q in questions
        if answers.get(str(q.get("id")), "").upper() == q.get("correct", "").upper()
    )
    score = int(correct / max(len(questions), 1) * 100)
    weak = [
        q.get("concept", state["topic"]) for q in questions
        if answers.get(str(q.get("id")), "").upper() != q.get("correct", "").upper()
    ]
    session = state.get("session_count", 0) + 1

    print(f"  ✅ Score: {score}% | Weak: {weak} | Session #{session}")
    return {
        "score": score, "weak_areas": weak, "session_count": session,
        "feedback": f"Session {session}: {score}% on {state['topic']}. "
                    + (f"Review: {', '.join(weak)}." if weak else "All concepts solid! 🎉")
    }


# ─── Conditional edge: decide next step after evaluation ─────────────────────

def should_replan(state: StudyState) -> str:
    """
    Called by LangGraph after evaluate_node to choose the next node.
    Returns 'replan' → loops back to plan_node
    Returns 'done'   → goes to END
    """
    if state.get("score", 100) < 60 and state.get("session_count", 0) < state.get("max_sessions", 3):
        print(f"  🔄 Score {state['score']}% < 60% — triggering replan")
        return "replan"
    print(f"  ✅ {'Score sufficient' if state.get('score',0)>=60 else 'Max sessions reached'} — done")
    return "done"


# ─── Build the graph ──────────────────────────────────────────────────────────

def build_study_graph():
    builder = StateGraph(StudyState)
    builder.add_node("planner", plan_node)
    builder.add_node("researcher", research_node)
    builder.add_node("quiz", quiz_node)
    builder.add_node("evaluator", evaluate_node)
    builder.set_entry_point("planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "quiz")
    builder.add_edge("quiz", "evaluator")
    # Conditional: after evaluator, call should_replan() to choose next node
    builder.add_conditional_edges(
        "evaluator", should_replan,
        {"replan": "planner", "done": END}
    )
    return builder.compile()


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  🔗 StudyBuddy + LangGraph")
    print("  Flow: plan → research → quiz → evaluate → (replan or done)")
    print("=" * 60)

    graph = build_study_graph()

    final = graph.invoke({
        "student_name": "Alex", "topic": "Binary Search Tree",
        "exam_date": "2026-06-20", "max_sessions": 3,
        # Initialise all mutable fields
        "study_plan": "", "research_content": "",
        "quiz_questions": [], "student_answers": {},
        "score": 0, "weak_areas": [], "feedback": "", "session_count": 0,
    })

    print("\n" + "=" * 60)
    print("  📊 FINAL STATE")
    print(f"  Score: {final['score']}% | Sessions: {final['session_count']}")
    print(f"  Weak areas: {final['weak_areas'] or 'None!'}")
    print(f"  Feedback: {final['feedback']}")
```

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

```python
# level3_fastapi_service.py
#
# ─── INSTALL REQUIRED PACKAGES ───────────────────────────────────────────────
# pip install fastapi uvicorn google-genai python-dotenv
#
# fastapi   → web framework
# uvicorn   → ASGI server that runs FastAPI
# ─────────────────────────────────────────────────────────────────────────────
#
# Run:  uvicorn level3_fastapi_service:app --reload --port 8000
# Docs: http://localhost:8000/docs  (auto-generated interactive API docs!)
# Test: curl http://localhost:8000/health

import os                        # ✅ stdlib
import uuid                      # ✅ stdlib — generates unique session IDs
import json                      # ✅ stdlib
import datetime                  # ✅ stdlib
from dotenv import load_dotenv   # 📦 python-dotenv
from google import genai         # 📦 google-genai
from google.genai import types   # 📦 google-genai
from fastapi import FastAPI, HTTPException   # 📦 fastapi
from fastapi.middleware.cors import CORSMiddleware  # 📦 fastapi (built-in)
from pydantic import BaseModel   # 📦 fastapi (pydantic for request validation)

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="StudyBuddy API",
    description="AI-powered CS exam preparation assistant",
    version="4.0"
)

# CORS: allows Streamlit (port 8501) to call this API (port 8000).
# Browsers block cross-origin requests without CORS headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # In production: restrict to your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request/Response schemas (Pydantic validates automatically) ──────────────

class AskRequest(BaseModel):
    question: str
    session_id: str | None = None   # Omit for new session, provide to continue

class AskResponse(BaseModel):
    answer: str
    session_id: str     # Always returned — client stores for follow-up calls

class QuizRequest(BaseModel):
    topic: str
    num_questions: int = 3

class QuizResponse(BaseModel):
    topic: str
    questions: list[dict]

# ─── Session store (in-memory — use Redis for production) ─────────────────────
# session_id → list of conversation turns (for memory)

sessions: dict[str, list] = {}

SYSTEM_PROMPT = """
You are StudyBuddy, a friendly AI CS tutor for college students.
Give clear, concise answers. Use simple examples. Be encouraging.
Respond in 3-5 sentences unless the student asks for more detail.
"""


def get_or_create_session(session_id: str | None) -> tuple[str, list]:
    """Get existing session or start a new one. Returns (session_id, history)."""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]
    new_id = str(uuid.uuid4())[:8]   # Short 8-char UUID
    sessions[new_id] = []
    return new_id, sessions[new_id]


def chat_with_gemini(message: str, history: list) -> str:
    """Single Gemini call that maintains full conversation history for memory."""
    history.append({"role": "user", "parts": [{"text": message}]})
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.5
        ),
        contents=history
    )
    answer = response.text
    history.append({"role": "model", "parts": [{"text": answer}]})
    return answer


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """
    Quick server health check.
    Test: curl http://localhost:8000/health
    """
    return {
        "status": "healthy",
        "active_sessions": len(sessions),
        "timestamp": datetime.datetime.now().isoformat()
    }


@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """
    Ask StudyBuddy a question with conversation memory.

    First call (new session):
      curl -X POST http://localhost:8000/ask \\
           -H "Content-Type: application/json" \\
           -d '{"question": "What is a BST?"}'

    Follow-up (continue same session — use session_id from first response):
      curl -X POST http://localhost:8000/ask \\
           -H "Content-Type: application/json" \\
           -d '{"question": "What is its time complexity?", "session_id": "abc12345"}'
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id, history = get_or_create_session(req.session_id)

    try:
        answer = chat_with_gemini(req.question, history)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"AI service error: {e}")

    return AskResponse(answer=answer, session_id=session_id)


@app.post("/quiz", response_model=QuizResponse)
async def generate_quiz(req: QuizRequest):
    """
    Generate MCQ quiz on any CS topic.

    Test:
    curl -X POST http://localhost:8000/quiz \\
         -H "Content-Type: application/json" \\
         -d '{"topic": "Binary Search Tree", "num_questions": 3}'
    """
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.4
            ),
            contents=(
                f"Generate {req.num_questions} MCQ questions about {req.topic}. "
                f'JSON array: [{{"id":1,"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},'
                f'"correct":"A","explanation":"..."}}]'
            )
        )
        questions = json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Quiz generation failed: {e}")

    return QuizResponse(topic=req.topic, questions=questions)


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a session's history to start fresh."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    sessions.pop(session_id)
    return {"message": f"Session {session_id} cleared."}


@app.get("/sessions")
async def list_sessions():
    """List all active sessions (useful for debugging)."""
    return {
        "total": len(sessions),
        "sessions": [{"id": sid, "turns": len(h)//2} for sid, h in sessions.items()]
    }
```

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

```python
# level3_streamlit_app.py
#
# ─── INSTALL REQUIRED PACKAGES ───────────────────────────────────────────────
# pip install streamlit google-genai python-dotenv
# ─────────────────────────────────────────────────────────────────────────────
#
# Run:  streamlit run level3_streamlit_app.py
# Opens http://localhost:8501 in your browser automatically!

import os                        # ✅ stdlib
from dotenv import load_dotenv   # 📦 python-dotenv
from google import genai         # 📦 google-genai
from google.genai import types   # 📦 google-genai
import streamlit as st           # 📦 streamlit

load_dotenv()

# ─── MUST be first Streamlit call ─────────────────────────────────────────────
st.set_page_config(page_title="StudyBuddy 🤖", page_icon="🤖", layout="wide")


# ─── Cache the Gemini client (created once, reused across reruns) ─────────────
# Without @st.cache_resource, a new client is created on EVERY user interaction.
@st.cache_resource
def get_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        st.error("❌ GEMINI_API_KEY not found. Check your .env file.")
        st.stop()
    return genai.Client(api_key=key)

gemini = get_client()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    topic = st.selectbox(
        "📚 Study Topic",
        ["Binary Search Tree", "Dynamic Programming", "Graph Algorithms",
         "Sorting Algorithms", "Linked Lists"]
    )
    temperature = st.slider("🎨 Creativity", 0.0, 1.0, 0.4, 0.1)
    mode = st.radio("🎯 Mode", ["💬 Chat", "❓ Quiz Me", "📝 Study Notes"])
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()


# ─── Session state (persists between Streamlit reruns) ────────────────────────
# Without session_state, all variables reset every time the user types.

if "messages" not in st.session_state:
    st.session_state.messages = []   # displayed in UI
if "history" not in st.session_state:
    st.session_state.history = []    # sent to Gemini (for memory)


# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🤖 StudyBuddy v4.0")
st.caption(f"Topic: **{topic}** | Mode: **{mode}** | Temp: **{temperature}**")
st.divider()

# Build system prompt from sidebar selections
SYSTEM = (
    f"You are StudyBuddy, a CS tutor specialising in {topic}. "
    f"Mode: {mode}. "
    + ("Ask MCQ quiz questions when the student says 'ready'." if "Quiz" in mode
       else "Generate structured notes when asked." if "Notes" in mode
       else "Have a natural tutoring conversation.")
)

# ─── Display existing messages ────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Welcome on first load
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            f"Hi! I'm StudyBuddy 👋 studying **{topic}** in **{mode}** mode.\n\n"
            + ("Say **ready** for your first quiz question!" if "Quiz" in mode
               else "What would you like to study first?")
        )


# ─── Chat input ───────────────────────────────────────────────────────────────
# st.chat_input() returns None until the user presses Enter

if prompt := st.chat_input(f"Ask about {topic}..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add to Gemini history
    st.session_state.history.append({"role": "user", "parts": [{"text": prompt}]})

    # Get AI response with a loading spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = gemini.models.generate_content(
                    model="gemini-2.0-flash",
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM,
                        temperature=temperature,
                        max_output_tokens=600
                    ),
                    contents=st.session_state.history
                )
                answer = resp.text
            except Exception as e:
                answer = f"⚠️ Error: {e}"
        st.markdown(answer)

    # Store response
    st.session_state.history.append({"role": "model", "parts": [{"text": answer}]})
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Footer stats
st.divider()
st.caption(
    f"Session turns: {len(st.session_state.messages)//2} | "
    f"Memory: {len(st.session_state.history)} msgs | Gemini 2.0-Flash"
)
```

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

```python
# level3_pgvector_demo.py
#
# ─── INSTALL REQUIRED PACKAGES ───────────────────────────────────────────────
# pip install psycopg2-binary pgvector sentence-transformers python-dotenv
#
# psycopg2-binary      → PostgreSQL Python driver
# pgvector             → pgvector Python client (handles vector types)
# sentence-transformers→ local embedding model (~80MB, no API needed for embeddings)
#
# Prereq: Docker with pgvector running (see bash commands above)
# Run: python level3_pgvector_demo.py
# ─────────────────────────────────────────────────────────────────────────────

import os                        # ✅ stdlib
import psycopg2                  # 📦 psycopg2-binary
from pgvector.psycopg2 import register_vector   # 📦 pgvector
from sentence_transformers import SentenceTransformer  # 📦 sentence-transformers

# ─── Load local embedding model (no API key needed!) ─────────────────────────
# all-MiniLM-L6-v2: 384-dimensional vectors, ~80MB, runs on CPU
# Downloads automatically on first use
print("🔄 Loading embedding model (~80MB download on first use)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_DIM = 384
print("✅ Embedding model ready")

# ─── Connect to PostgreSQL ─────────────────────────────────────────────────
conn = psycopg2.connect(
    host="localhost", port=5432,
    dbname="studybuddy_db", user="student", password="studybuddy"
)
register_vector(conn)   # Teaches psycopg2 how to handle the vector type
print("✅ Connected to PostgreSQL + pgvector")

# ─── Create table and index ───────────────────────────────────────────────────

with conn.cursor() as cur:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS study_chunks (
            id         SERIAL PRIMARY KEY,
            content    TEXT NOT NULL,
            topic      VARCHAR(100),
            embedding  vector({VECTOR_DIM}),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    # ivfflat index: approximate nearest-neighbour for fast similarity search
    # Without index: PostgreSQL scans every row (slow at scale)
    # With index: narrows to candidate partitions first (fast!)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx
        ON study_chunks USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 10);
    """)
    conn.commit()
print("✅ Table and ivfflat index ready")

# ─── Insert study chunks ──────────────────────────────────────────────────────

def add_chunk(content: str, topic: str) -> None:
    """Convert text to vector and insert into PostgreSQL."""
    # embedder.encode() runs locally — no internet required
    vector = embedder.encode(content).tolist()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO study_chunks (content, topic, embedding) VALUES (%s, %s, %s)",
            (content, topic, vector)
        )
    conn.commit()

STUDY_DATA = [
    ("A BST is a binary tree where left < parent < right. "
     "O(log n) avg for search/insert/delete, O(n) worst case.", "BST"),
    ("Dynamic Programming caches subproblem results to avoid recomputation. "
     "Top-down: memoization. Bottom-up: tabulation.", "DP"),
    ("Merge sort is O(n log n) all cases, stable, uses O(n) space. "
     "Better than quicksort when stability matters.", "Sorting"),
]

print("\n📥 Inserting chunks with vector embeddings...")
for content, topic in STUDY_DATA:
    add_chunk(content, topic)
print(f"✅ {len(STUDY_DATA)} chunks inserted")

# ─── Semantic search using pgvector ──────────────────────────────────────────

def search_similar(query: str, limit: int = 2) -> list[dict]:
    """
    Find the most semantically similar chunks.

    The <=> operator computes COSINE DISTANCE between vectors.
    Lower distance = more similar. ORDER BY distance ASC = most similar first.
    This is very different from SQL '=' which requires exact matches!
    """
    query_vec = embedder.encode(query).tolist()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT content, topic,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM study_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (query_vec, query_vec, limit)
        )
        rows = cur.fetchall()
    return [{"content": r[0], "topic": r[1], "similarity": round(r[2], 3)} for r in rows]


# ─── Demo ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  🔍 PGVECTOR SEMANTIC SEARCH")
print("=" * 60)

for query in ["how fast is BST lookup?", "overlapping subproblems cache", "stable sort algorithm"]:
    print(f"\n🔎 Query: '{query}'")
    for r in search_similar(query):
        print(f"  [{r['topic']}] similarity={r['similarity']}: {r['content'][:80]}...")

conn.close()
print("\n✅ pgvector demo complete!")
```

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

```python
# level3_rag_vs_finetune_comparison.py
#
# ─── INSTALL REQUIRED PACKAGES ───────────────────────────────────────────────
# pip install chromadb google-genai python-dotenv
# ─────────────────────────────────────────────────────────────────────────────
#
# Demonstrates WHY RAG is better than "fine-tuning" for domain-specific knowledge.
# Method A: plain Gemini (simulates model trained without your data)
# Method B: RAG Gemini (retrieved context from VectorDB)
#
# Run: python level3_rag_vs_finetune_comparison.py

import os                        # ✅ stdlib
from dotenv import load_dotenv   # 📦 python-dotenv
from google import genai         # 📦 google-genai
import chromadb                  # 📦 chromadb
from chromadb.utils import embedding_functions  # 📦 chromadb

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# VectorDB with company-specific knowledge (Gemini wouldn't know this)
chroma = chromadb.Client()
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma.create_collection("company_docs", embedding_function=embedding_fn)

PROPRIETARY_DOCS = [
    "StudyBuddy grading: scores ≥85% = Gold Star badge. 70-84% = Silver Star. "
    "<70% triggers a mandatory review session before next quiz.",
    "StudyBuddy pricing: Free (5 sessions/month), Pro $9.99/month (unlimited), "
    "Team $29.99/month (up to 10 students, shared dashboards).",
    "StudyBuddy study schedule policy: 45-minute focused sessions, 15-minute breaks "
    "(Pomodoro technique). Maximum 3 cycles per day to avoid burnout.",
]

collection.add(
    documents=PROPRIETARY_DOCS,
    ids=[f"doc_{i}" for i in range(len(PROPRIETARY_DOCS))]
)

question = "What happens if a student scores 78% on a StudyBuddy quiz?"

print("=" * 64)
print(f"  Question: '{question}'")
print("=" * 64)

# METHOD A: Plain Gemini — has no proprietary knowledge
print("\n❌ METHOD A: Plain Gemini (no retrieval)")
print("   (Simulates a model that wasn't trained on our company docs)")
print("-" * 50)
resp_a = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=f"You are a StudyBuddy support AI. Answer: {question}"
)
print(resp_a.text)
print("\n⚠️  Gemini makes up a generic answer — it doesn't know our grading policy!")

# METHOD B: RAG — retrieves company-specific policy first
print("\n✅ METHOD B: RAG Gemini (with retrieval)")
print("   (Retrieves the actual company policy from VectorDB)")
print("-" * 50)
results = collection.query(query_texts=[question], n_results=2)
context = "\n".join(results["documents"][0])
rag_prompt = (
    f"You are a StudyBuddy support AI.\n\n"
    f"Company policy (answer ONLY from this):\n{context}\n\n"
    f"Question: {question}"
)
resp_b = client.models.generate_content(
    model="gemini-2.0-flash", contents=rag_prompt
)
print(resp_b.text)
print("\n✅ RAG gives the exact correct answer by retrieving the policy document!")

print("\n" + "=" * 64)
print("  KEY INSIGHT: RAG vs Fine-tuning for this use case")
print("=" * 64)
print("  RAG wins because:")
print("  • Pricing/policy changes monthly — RAG updates are instant (DB insert)")
print("  • Fine-tuning would need a new training run for every policy change")
print("  • Budget: RAG = free DB storage. Fine-tuning = $$$")
print("  • Fine-tuning only makes sense for teaching STYLE, not facts")
```

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

