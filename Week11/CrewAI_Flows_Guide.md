# CrewAI Flows - Simple Guide

## 🎯 What is CrewAI?

CrewAI is a framework for creating **teams of AI agents** that work together to complete complex tasks. Think of it like a virtual company where each "employee" (agent) has a specific role.

---

## 🎀 What are Decorators?

**Decorators** are a Python feature that lets you "wrap" a function to add extra behavior. They start with `@` symbol.

### Simple Example

```python
# Without decorator
def say_hello():
    return "Hello"

# With decorator - adds extra behavior
@make_fancy
def say_hello():
    return "Hello"
```

### How Decorators Work (Visual)

```
┌─────────────────────────────────────────────────────────────┐
│                    DECORATOR FLOW                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   @decorator          Your Function         Enhanced Result  │
│   ┌─────────┐         ┌───────────┐         ┌────────────┐  │
│   │ Wrapper │ ──────▶ │  Original │ ──────▶ │   Output   │  │
│   │  Code   │         │  Function │         │  + Extra   │  │
│   └─────────┘         └───────────┘         └────────────┘  │
│       ▲                                           │          │
│       └───────────────────────────────────────────┘          │
│                    Adds behavior before/after                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Real-World Analogy

Think of a decorator like a **gift wrapper**:
- Your function is the **gift** (the actual thing)
- The decorator is the **wrapping paper** (adds presentation)
- The result is a **wrapped gift** (function + extra features)

### CrewAI Decorators Explained

| Decorator | What it Does | Real-World Analogy |
|-----------|-------------|-------------------|
| `@CrewBase` | Marks a class as a CrewAI crew | "This is a company" |
| `@agent` | Registers a method as an agent | "This person is an employee" |
| `@task` | Registers a method as a task | "This is a job to be done" |
| `@crew` | Creates the final crew | "This is the team" |
| `@before_kickoff` | Runs before crew starts | "Prepare before work" |
| `@after_kickoff` | Runs after crew finishes | "Clean up after work" |

### CrewAI Decorator Example

```python
@CrewBase  # "This class is a CrewAI crew"
class MyCompany:
    
    @agent  # "This method returns an employee"
    def researcher(self) -> Agent:
        return Agent(role='Researcher', goal='Find info')
    
    @task  # "This method returns a job"
    def research_task(self) -> Task:
        return Task(description='Do research')
    
    @crew  # "This method creates the team"
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks)
```

### With vs Without Decorator (Full Example)

**❌ WITHOUT Decorator (Manual Way):**

```python
def say_hello(name):
    return f"Hello, {name}!"

# Manually wrap it to add behavior
def add_greeting(func):
    def wrapper(name):
        print("🎉 Starting...")
        result = func(name)
        print("✅ Done!")
        return result
    return wrapper

# Manually apply
say_hello = add_greeting(say_hello)

say_hello("Arif")
# Output:
# 🎉 Starting...
# ✅ Done!
# "Hello, Arif!"
```

**✅ WITH Decorator (Clean Way):**

```python
def add_greeting(func):
    def wrapper(name):
        print("🎉 Starting...")
        result = func(name)
        print("✅ Done!")
        return result
    return wrapper

@add_greeting  # ← Same thing, but cleaner!
def say_hello(name):
    return f"Hello, {name}!"

say_hello("Arif")
# Output:
# 🎉 Starting...
# ✅ Done!
# "Hello, Arif!"
```

**🔑 Key Point - These are EXACTLY the same:**

```python
# Manual way:
say_hello = add_greeting(say_hello)

# Decorator way:
@add_greeting
def say_hello(name):
    ...
```

The `@` is just syntactic sugar (shortcut) for wrapping a function! 🍬

---

### How Decorators Auto-Call Functions

When you use `@some_function`, Python automatically:
1. Takes your function
2. Passes it to `some_function`
3. Replaces your function with whatever `some_function` returns

**Proof Example:**

```python
def my_decorator(func):
    print(f"🔥 I received: {func.__name__}")  # This runs automatically!
    return func

@my_decorator  # ← This CALLS my_decorator and passes greet to it
def greet():
    print("Hello!")

# Output (happens immediately when Python reads the code):
# 🔥 I received: greet
```

**What Python Does Behind the Scenes:**

```python
@my_decorator
def greet():
    print("Hello!")

# Python converts this to:
greet = my_decorator(greet)  # ← Auto-calls my_decorator!
```

**Visual Flow:**

```
┌────────────────────────────────────────────────┐
│  @my_decorator                                  │
│  def greet():                                   │
│      print("Hello!")                            │
└────────────────────────────────────────────────┘
                    ↓
        Python automatically does:
                    ↓
┌────────────────────────────────────────────────┐
│  greet = my_decorator(greet)                   │
│                 ↑                              │
│         Calls this function                    │
│         with greet as argument                 │
└────────────────────────────────────────────────┘
```

**So in CrewAI:**
- `@agent` → Calls `agent()` function automatically
- `@task` → Calls `task()` function automatically  
- `@before_kickoff` → Calls `before_kickoff()` function automatically

**The decorator function runs when Python loads the code, not when you call the decorated function!**

---

### Why Use Decorators?

1. **Cleaner Code**: Less boilerplate, more readable
2. **Auto-Registration**: CrewAI automatically finds all agents/tasks
3. **Lifecycle Hooks**: Easy setup and cleanup
4. **Separation**: Keep configuration separate from logic

---

## 📦 The 3 Core Building Blocks

### 1. **Agent** 🤖 = The Worker

```python
research_agent = Agent(
    role='Senior Research Analyst',        # Job title
    goal='Find cutting-edge AI developments',  # What they aim to do
    backstory='You have 8 years experience...'  # Their expertise
)
```

**Key Properties:**
| Property | What it does |
|----------|-------------|
| `role` | Job title (who they are) |
| `goal` | What they're trying to achieve |
| `backstory` | Their expertise & personality |
| `tools` | External capabilities (search, files, etc.) |

---

### 2. **Task** 📋 = The Assignment

```python
research_task = Task(
    description='Research LLM developments in 2024...',  # What to do
    expected_output='A 2000-word research report...',    # Expected result
    agent=research_agent                                  # Who does it
)
```

**Key Properties:**
| Property | What it does |
|----------|-------------|
| `description` | Detailed instructions |
| `expected_output` | Format of the result |
| `agent` | Which agent does this task |

---

### 3. **Crew** 👥 = The Team

```python
research_crew = Crew(
    agents=[research_agent],     # List of workers
    tasks=[research_task],       # List of work to do
    process=Process.sequential,  # How to execute
    verbose=True                 # Show progress
)

result = research_crew.kickoff()  # 🚀 Start the work!
```

---

## 🔄 How Tasks Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    SEQUENTIAL FLOW                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Task 1          Task 2          Task 3          Task 4    │
│   ┌─────┐         ┌─────┐         ┌─────┐         ┌─────┐  │
│   │Agent│ ──────▶ │Agent│ ──────▶ │Agent│ ──────▶ │Agent│  │
│   │  A  │         │  B  │         │  C  │         │  D  │  │
│   └─────┘         └─────┘         └─────┘         └─────┘  │
│      │               │               │               │      │
│      ▼               ▼               ▼               ▼      │
│   Output 1 ────▶ Output 2 ────▶ Output 3 ────▶ Final      │
│   (passed to     (passed to     (passed to      Result     │
│    Task 2)        Task 3)        Task 4)                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Day 1 vs Day 2 Comparison

### **Day 1: Simple Crew** (1_crewai.py)
- ✅ 1 Agent (Research Analyst)
- ✅ 1 Task (LLM Research Report)
- ✅ Code-based configuration
- ✅ Good for simple use cases

### **Day 2: Advanced Crew** (SDLC, News, etc.)
- ✅ Multiple Agents (7 specialists)
- ✅ Multiple Tasks (8 documents)
- ✅ YAML-based configuration
- ✅ Decorators (`@agent`, `@task`, `@crew`)
- ✅ Lifecycle hooks (`@before_kickoff`, `@after_kickoff`)

---

## 🏗️ Day 2 Advanced Pattern (SDLC Example)

```
┌─────────────────────────────────────────────────────────────┐
│                    SDLC CREW FLOW                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Product        Business        Software        API          │
│  Manager   ──▶  Analyst    ──▶  Architect  ──▶  Designer    │
│     │              │               │              │          │
│     ▼              ▼               ▼              ▼          │
│  product_     business_       software_      api_specs.     │
│  reqs.md      reqs.md         arch.md         yaml          │
│                                                              │
│  Technical       UI/UX           QA                          │
│  Lead       ──▶  Designer   ──▶  Lead                       │
│     │              │               │                         │
│     ▼              ▼               ▼                         │
│  high_level_   low_level_     roadmap.md                    │
│  design.md     design.md                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎛️ Key Decorators in Action (Day 2)

```python
@CrewBase
class SDLCDevelopmentCrew:
    
    @before_kickoff  # Runs before crew starts
    def prepare_inputs(self, inputs):
        inputs['tech_stack'] = {...}
        return inputs
    
    @agent  # Defines an agent
    def product_manager(self) -> Agent:
        return Agent(config=self.agents_config['product_manager'])
    
    @task  # Defines a task
    def analyze_requirements(self) -> Task:
        return Task(config=self.tasks_config['analyze_requirements'])
    
    @crew  # Creates the crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks)
    
    @after_kickoff  # Runs after crew finishes
    def process_output(self, output):
        print("🎉 Done!")
```

---

## 📂 YAML Configuration (Cleaner Code)

**agents.yaml:**
```yaml
product_manager:
  role: "Senior Product Manager"
  goal: "Define product strategy"
  backstory: "8+ years experience..."
```

**tasks.yaml:**
```yaml
analyze_requirements:
  description: "Analyze business requirements..."
  expected_output: "A comprehensive document..."
  agent: business_analyst
```

---

## 🔑 Important Things to Remember

| Concept | Purpose |
|---------|---------|
| **Sequential Process** | Tasks run one after another, output passes to next |
| **Hierarchical Process** | Manager agent delegates to worker agents |
| **Memory** | Crew remembers context across tasks |
| **Delegation** | Agents can ask other agents for help |
| **Tools** | External capabilities (search, files, APIs) |
| **Verbose** | Shows detailed execution logs |

---

## 🎯 Week 11 Projects Summary

### Day 1: Basic CrewAI
| File | What it does |
|------|-------------|
| `1_crewai.py` | Simple research agent that writes LLM reports |

### Day 2: Advanced Projects
| Project | Agents | Purpose |
|---------|--------|---------|
| `1_annual_reports_analysis/` | Financial analysts | Analyze company PDFs |
| `2_news_aggregator/` | News curators | Aggregate and summarize news |
| `3_sdlc_plan/` | PM, Architect, Dev, QA | Generate full software specs |
| `4_code_assist/` | Code assistant | Help with coding tasks |

---

## 🚀 Quick Start Template

```python
from crewai import Agent, Task, Crew

# 1. Create Agent
agent = Agent(
    role='Expert',
    goal='Do great work',
    backstory='You are an expert...'
)

# 2. Create Task
task = Task(
    description='Complete this work...',
    expected_output='A detailed report',
    agent=agent
)

# 3. Create Crew & Run
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
print(result)
```

---

## 📚 Key Takeaways

1. **Agent** = Worker with a role, goal, and backstory
2. **Task** = Job assignment with description and expected output
3. **Crew** = Team that runs all tasks
4. **Decorators** = `@` symbols that add extra behavior to functions
5. **Sequential** = Tasks run one after another
6. **YAML Config** = Cleaner way to define agents and tasks

