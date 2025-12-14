# Week 10 - Day 1: AI Agents

This folder contains examples of AI agents that can autonomously perform tasks.

---

## 1. Simple AI Agent with 2 Tools (`simple-ai-agent-with-2tools.py`)

A simple AI agent that uses OpenAI's function calling to decide which tool to use based on the user's query.

### Tools Available:
- **Calculator**: Evaluates mathematical expressions
- **Web Search**: Searches the web using Tavily API

### How It Works:

```
User Query → LLM decides which tool(s) to use
                    ↓
        ┌──────────┴──────────┐
        ▼                     ▼
   Calculator            Web Search
   (math ops)           (Tavily API)
        ↓                     ↓
   Result: number       News articles
        └──────────┬──────────┘
                   ↓
         LLM combines results
                   ↓
         Final formatted answer
```

### Sample Output:

```
User Query: What is 156 multiplied by 23? and What are the latest developments in AI this week?
============================================================

LLM decided to use: calculator
Arguments: {'expression': '156 * 23'}

Tool Response:
Result: 3588

LLM decided to use: web_search
Arguments: {'query': 'latest developments in AI this week'}

Tool Response:
Title: AI Magazine: Home of AI and Artificial Intelligence News
Content: This Week's Top 5 Stories in AI...
URL: https://aimagazine.com/

Final Answer:
156 multiplied by 23 equals **3588**.

### Latest Developments in AI This Week:
1. Amazon has made a significant investment of $50 billion in AI infrastructure
2. Elon Musk's xAI is nearing a $15 billion funding round
3. Trump signed an order aimed at regulating state-level AI laws
============================================================
```

### Requirements:
```bash
pip install openai tavily-python python-dotenv
```

### Environment Variables:
```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

## 2. Browser Automation Agent (`flight.py`)

An AI-powered browser automation agent that can autonomously browse the web and perform tasks using natural language instructions.

### How It Works:

```
┌─────────────────────────────────────────┐
│              Your Task                  │
│  "Go to YouTube and play a video"       │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│              Agent (AI)                 │
│  • Understands the task                 │
│  • Plans: Navigate → Search → Click     │
│  • Uses Gemini to decide actions        │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│          BrowserSession                 │
│  • Actual Chrome/Chromium browser       │
│  • Executes clicks, typing, navigation  │
│  • Handles page loading                 │
└─────────────────────────────────────────┘
```

### Sample Output:

```
INFO [Agent] 🎯 Task: Go to Youtube and play me a Virat Kohli Cover drive video against Australia

INFO [Agent] 📍 Step 1:
INFO [Agent]   🎯 Next goal: Navigate to YouTube's homepage.
INFO [tools] 🔗 Navigated to https://www.youtube.com

INFO [Agent] 📍 Step 2:
INFO [Agent]   🎯 Next goal: Input search query into the search bar.
INFO [BrowserSession] ⌨️ Typed "Virat Kohli Cover drive video against Australia"

INFO [Agent] 📍 Step 3:
INFO [tools] 🔍 Scrolled down to view search results

INFO [Agent] 📍 Step 4:
INFO [tools] 🖱️ Clicked on video

INFO [Agent] 📍 Step 5:
INFO [Agent]   ▶️ done: Successfully played the video

📄 Final Result: 
I have successfully navigated to YouTube and played the video 
'The VIRAT KOHLI cover drive! He scores a stunning boundary off Pat Cummins!'

INFO [Agent] ✅ Task completed successfully
```

### Requirements:
```bash
pip install browser-use playwright python-dotenv
playwright install
```

### Environment Variables:
```
GEMINI_KEY=your_google_gemini_api_key
```

---

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install browser-use playwright openai tavily-python python-dotenv
playwright install
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_key
GEMINI_KEY=your_gemini_key
TAVILY_API_KEY=your_tavily_key
```

4. Run the scripts:
```bash
python simple-ai-agent-with-2tools.py
python flight.py
```

