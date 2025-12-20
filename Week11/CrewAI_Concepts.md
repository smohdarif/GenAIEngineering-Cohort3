# CrewAI Concepts - Week 11 Overview

## What is CrewAI?

**CrewAI = Team of AI agents working together like employees**

```
┌─────────────────────────────────────────────┐
│                   CREW                      │
│  (Your AI Team)                             │
│                                             │
│   Agent 1 ──→ Task 1 ──→ Output            │
│   Agent 2 ──→ Task 2 ──→ Output            │
│   Agent 3 ──→ Task 3 ──→ Final Result      │
└─────────────────────────────────────────────┘
```

---

## Core Concepts

| Concept | Description | Analogy |
|---------|-------------|---------|
| 🤖 **Agent** | Worker with role, goal & skills | Employee |
| 📋 **Task** | Work assignment with description | Job ticket |
| 👥 **Crew** | Team that runs tasks | Department |
| 🔧 **Tools** | Capabilities (search, read files) | Skills/Equipment |
| 🧠 **Memory** | Context sharing between agents | Team knowledge |

---

## How CrewAI Works (Simplest Form)

```python
from crewai import Agent, Task, Crew

# 1. Create an Agent (the worker)
agent = Agent(
    role="Researcher",
    goal="Find information",
    backstory="Expert analyst with 10 years experience"
)

# 2. Create a Task (the work)
task = Task(
    description="Research latest AI developments",
    agent=agent,
    expected_output="A summary report"
)

# 3. Create a Crew (the team)
crew = Crew(
    agents=[agent],
    tasks=[task]
)

# 4. Run it!
result = crew.kickoff()
print(result)
```

---

## Week 11 - Day 1 (Learning CrewAI Basics)

12 scripts progressively teaching CrewAI:

| Script | Concept Covered |
|--------|-----------------|
| `1_crewai.py` | Basic Agent + Task + Crew |
| `2_crewai.py` | Multiple agents collaboration |
| `3_crewai.py` | Agent delegation |
| `4_crewai.py` | Sequential task execution |
| `5_crewai.py` | Adding Tools (web search) |
| `6_crewai.py` | Custom tools |
| `7_crewai.py` | File operations |
| `8_crewai.py` | Memory & context |
| `9_crewai.py` | Advanced workflows |
| `10_crewai.py` | PDF analysis |
| `11_crewai.py` | Output formatting |
| `12_crewai.py` | Complete pipeline |

---

## Week 11 - Day 2 (Real-World Applications)

4 practical projects applying CrewAI:

| Project | Description | Key Features |
|---------|-------------|--------------|
| `1_annual_reports_analysis/` | Analyze financial PDFs | RAG, PDF parsing, FAISS |
| `2_news_aggregator/` | Collect & summarize news | Multi-agent, API integration |
| `3_sdlc_plan/` | Generate software docs | Requirements, architecture |
| `4_code_assist/` | AI coding assistant | Code generation, review |

---

## Agent Parameters

```python
Agent(
    role='Senior Research Analyst',      # Job title
    goal='Find breakthrough innovations', # What to achieve
    backstory='8 years experience...',    # Personality/expertise
    tools=[search_tool, file_tool],       # Available tools
    llm=my_llm,                           # Which AI model to use
    verbose=True                          # Show detailed logs
)
```

---

## Task Parameters

```python
Task(
    description='Analyze market trends',  # What to do
    agent=research_agent,                 # Who does it
    expected_output='A detailed report',  # What to deliver
    tools=[web_search],                   # Tools for this task
    context=[previous_task]               # Dependencies
)
```

---

## Crew Parameters

```python
Crew(
    agents=[agent1, agent2],              # Team members
    tasks=[task1, task2],                 # Work to do
    process=Process.sequential,           # How to run (sequential/hierarchical)
    verbose=True,                         # Show progress
    memory=True                           # Enable memory
)
```

---

## Flow Diagram

```
User Request
     │
     ▼
┌─────────┐
│  CREW   │ ◄── Orchestrates everything
└────┬────┘
     │
     ▼
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Agent 1 │ ──► │ Agent 2 │ ──► │ Agent 3 │
│Research │     │ Writer  │     │ Editor  │
└────┬────┘     └────┬────┘     └────┬────┘
     │               │               │
     ▼               ▼               ▼
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Task 1  │     │ Task 2  │     │ Task 3  │
│ Gather  │     │ Draft   │     │ Polish  │
│  Data   │     │ Report  │     │ Output  │
└─────────┘     └─────────┘     └─────────┘
                                     │
                                     ▼
                              Final Result
```

---

## Using OpenRouter (Alternative to OpenAI)

```python
import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'
```

This redirects API calls through OpenRouter, giving access to multiple models (GPT-4, Claude, Llama, etc.) with one API key.

---

## Quick Reference

| Action | Code |
|--------|------|
| Run crew | `crew.kickoff()` |
| Add tool | `agent = Agent(..., tools=[tool])` |
| Chain tasks | `task2 = Task(..., context=[task1])` |
| Save output | `with open('out.md', 'w') as f: f.write(str(result))` |

