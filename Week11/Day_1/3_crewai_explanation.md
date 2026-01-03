# 3_crewai.py Explained

This script demonstrates **how to create a CUSTOM TOOL** for a CrewAI agent!

---

## 🔧 Key Concept: Custom Tool Creation

```python
def calculate_compound_interest_func(principal: float, rate: float, time: int, n: int = 12) -> str:
    """
    Calculate compound interest.
    ...
    """
    amount = principal * (1 + rate/n) ** (n * time)
    interest = amount - principal
    return f"""
    Principal: ${principal:,.2f}
    ...
    """

# Convert function to a CrewAI tool
calculate_compound_interest = tool("Calculate Compound Interest")(calculate_compound_interest_func)
```

---

## 📊 Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      3_crewai.py Flow                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1️⃣ DEFINE CUSTOM FUNCTION                                      │
│     ┌─────────────────────────────────────────┐                │
│     │  calculate_compound_interest_func()     │                │
│     │  - Takes: principal, rate, time         │                │
│     │  - Returns: Formatted interest result   │                │
│     └─────────────────────────────────────────┘                │
│                          │                                      │
│                          ▼                                      │
│  2️⃣ WRAP AS CREWAI TOOL                                         │
│     ┌─────────────────────────────────────────┐                │
│     │  tool("Calculate Compound Interest")    │                │
│     │  (calculate_compound_interest_func)     │                │
│     └─────────────────────────────────────────┘                │
│                          │                                      │
│                          ▼                                      │
│  3️⃣ CREATE AGENT WITH TOOL                                      │
│     ┌─────────────────────────────────────────┐                │
│     │  Agent(                                 │                │
│     │    role='interest rate calculator'      │                │
│     │    tools=[calculate_compound_interest]  │ ◄── Tool given │
│     │  )                                      │                │
│     └─────────────────────────────────────────┘                │
│                          │                                      │
│                          ▼                                      │
│  4️⃣ CREATE TASK                                                 │
│     ┌─────────────────────────────────────────┐                │
│     │  Task(                                  │                │
│     │    description="calculate compound      │                │
│     │    interest {principal: 1000, ...}"     │                │
│     │  )                                      │                │
│     └─────────────────────────────────────────┘                │
│                          │                                      │
│                          ▼                                      │
│  5️⃣ AGENT DECIDES TO USE TOOL                                   │
│     ┌─────────────────────────────────────────┐                │
│     │  LLM sees task → "I need to calculate"  │                │
│     │  → Calls calculate_compound_interest()  │                │
│     │  → Returns formatted result             │                │
│     └─────────────────────────────────────────┘                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🆚 Comparison: Scripts 1, 2, 3

| Feature | 1_crewai.py | 2_crewai.py | 3_crewai.py |
|---------|-------------|-------------|-------------|
| **Agent** | Research Analyst | Research Analyst | Calculator |
| **Tools** | ❌ None | ✅ SerperDevTool (web) | ✅ Custom Tool |
| **Tool Type** | - | Built-in | **User-defined** |
| **Purpose** | LLM knowledge only | Web search | Math calculation |

---

## 💡 Key Takeaway

**3_crewai.py teaches you how to create YOUR OWN tools!**

```python
# Step 1: Write a normal Python function
def my_function(param1, param2):
    return result

# Step 2: Convert to CrewAI tool
my_tool = tool("Tool Name")(my_function)

# Step 3: Give to agent
agent = Agent(tools=[my_tool])
```

This is powerful because you can give agents **any capability** - database queries, API calls, file operations, calculations, etc.!

---

## 📝 Code Breakdown

### 1. Import the tool decorator
```python
from crewai.tools import tool
```

### 2. Define your function with type hints and docstring
```python
def calculate_compound_interest_func(principal: float, rate: float, time: int, n: int = 12) -> str:
    """
    Calculate compound interest.
    
    Args:
        principal: Initial amount
        rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
        time: Time period in years
        n: Number of times interest compounds per year
    """
    amount = principal * (1 + rate/n) ** (n * time)
    interest = amount - principal
    return f"Principal: ${principal:,.2f}, Final Amount: ${amount:,.2f}"
```

### 3. Wrap as CrewAI tool
```python
calculate_compound_interest = tool("Calculate Compound Interest")(calculate_compound_interest_func)
```

### 4. Give tool to agent
```python
calculator = Agent(
    role='interest rate calculator',
    goal='calculate compound interest based on user input',
    backstory="""You're a banker.""",
    tools=[calculate_compound_interest],  # ← Tool attached here
)
```

### 5. Agent automatically uses the tool when needed
The LLM sees the task description and decides to call the tool with the right parameters!

---

## 🚀 Why Custom Tools Matter

| Use Case | Custom Tool Example |
|----------|---------------------|
| Database | `query_database(sql)` |
| API Calls | `call_weather_api(city)` |
| File Ops | `read_csv(filename)` |
| Math | `calculate_interest(...)` |
| External Services | `send_email(to, subject, body)` |

Custom tools let you extend agents with **any Python functionality**!

