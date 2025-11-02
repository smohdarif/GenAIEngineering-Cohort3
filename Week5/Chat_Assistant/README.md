# 🚀 Knowledge Assistant - RAG-Based Chat Application

A comprehensive Retrieval-Augmented Generation (RAG) chat assistant that can capture knowledge from web URLs and answer questions based on that knowledge. Built with FastAPI, Streamlit, and advanced RAG techniques including query transformation and RAG fusion.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Complete Flow](#complete-flow)
- [Components](#components)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [How It Works](#how-it-works)

---

## 🎯 Overview

The Knowledge Assistant is a RAG (Retrieval-Augmented Generation) application that:
1. **Captures Knowledge**: Scrapes and processes web content from URLs
2. **Stores in Vector Database**: Converts text into embeddings and stores in LanceDB
3. **Retrieves Context**: Uses advanced retrieval techniques (RAG Fusion) to find relevant information
4. **Generates Answers**: Uses LLMs (Google Gemini) to generate context-aware responses

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Streamlit UI   │  (Frontend)
│  Chat_FrontEnd  │
└────────┬────────┘
         │ HTTP Requests
         ▼
┌─────────────────┐
│   FastAPI       │  (Backend API)
│  Chat_BackEnd   │
└────────┬────────┘
         │
         ├───► Assistant.py (Core Logic)
         │      ├── Capture_Knowledge()
         │      ├── Retrieve_Context()
         │      └── Ask_Assistant()
         │
         ├───► Web_Scraper.py
         │      └── fetch_main_content()
         │
         └───► LanceDB (Vector Database)
                └── Quick_Ref (Database)
```

---

## 🔄 Complete Flow

### Phase 1: Knowledge Capture Flow

```
User Input (URL + Table Name)
        │
        ▼
┌───────────────────────┐
│  POST /Capture_       │
│      Knowledge        │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Web_Scraper.py       │
│  - fetch_main_content │
│  - Remove nav/footer  │
│  - Extract main HTML  │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Build_Chunks()       │
│  - HTML Header Split  │
│  - Text Split (300ch) │
│  - Add Metadata       │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Create Embeddings    │
│  - all-MiniLM-L6-v2   │
│  - Convert to vectors │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  LanceDB              │
│  - Store vectors      │
│  - Store metadata     │
│  - Table created      │
└───────────────────────┘
```

### Phase 2: Query & Response Flow

```
User Query
    │
    ▼
┌───────────────────────┐
│  GET /Ask_Assistant   │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Retrieve_Context()   │
│                       │
│  1. Query Transform   │
│     - transform_query │
│     - Precise reform  │
│     - 3 paraphrases   │
│                       │
│  2. Query Expansion   │
│     - expand_query    │
│     - 3 alternates    │
│                       │
│  3. Multi-Query Search│
│     - 7 total queries │
│     - Vector search   │
│     - Top 5 per query │
│     - Deduplicate     │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Ask_Assistant()      │
│                       │
│  Input:               │
│  - Chat History       │
│  - Context (retrieved)│
│  - User Query         │
│                       │
│  LLM: Gemini 2.0      │
│  - System instruction │
│  - Generate response  │
└───────────┬───────────┘
            │
            ▼
     Response to User
```

---

## 🧩 Components

### 1. **Chat_FrontEnd.py** (Streamlit UI)
- **Purpose**: User interface for the chat application
- **Features**:
  - Sidebar for knowledge capture (URL + table name)
  - Chat interface with history
  - Real-time response display
  - Timestamp tracking

### 2. **Chat_BackEnd.py** (FastAPI Server)
- **Purpose**: REST API server
- **Endpoints**:
  - `GET /`: API information
  - `POST /Capture_Knowledge`: Capture knowledge from URL
  - `GET /Ask_Assistant`: Ask questions to the assistant
- **Features**: CORS enabled for frontend integration

### 3. **Assistant.py** (Core Logic)
- **Main Functions**:
  - `Capture_Knowledge()`: Orchestrates knowledge capture
  - `Build_Chunks()`: Splits content into chunks with metadata
  - `Retrieve_Context()`: Implements RAG Fusion retrieval
  - `transform_query()`: Query transformation using Groq
  - `expand_query()`: Query expansion using Groq
  - `Ask_Assistant()`: Generates responses using Gemini

### 4. **Web_Scraper.py** (Content Extraction)
- **Purpose**: Extract main content from web pages
- **Features**:
  - Removes navigation, headers, footers
  - Removes cookie banners/popups
  - Handles both static and dynamic content
  - Supports requests and Selenium methods

---

## ✨ Features

### Advanced RAG Techniques

1. **RAG Fusion**
   - Query transformation (precise + paraphrases)
   - Query expansion (diverse variations)
   - Multi-query retrieval
   - Result aggregation and deduplication

2. **Intelligent Chunking**
   - HTML header-based splitting
   - Recursive text splitting
   - Metadata preservation
   - Optimal chunk sizes (300 chars)

3. **Context-Aware Responses**
   - Chat history integration
   - Context-based generation
   - Natural conversation flow

4. **Robust Web Scraping**
   - Cookie banner removal
   - Content extraction
   - Fallback methods

---

## 🛠️ Technology Stack

### Backend
- **FastAPI**: REST API framework
- **Uvicorn**: ASGI server
- **LanceDB**: Vector database
- **Sentence Transformers**: Embedding models
  - `all-MiniLM-L6-v2` (default)
  - `all-mpnet-base-v2` (alternative)

### AI/LLM Services
- **Google Gemini 2.0**: Response generation
- **Groq**: Query transformation and expansion
  - Model: `llama-3.3-70b-versatile`

### Frontend
- **Streamlit**: Web UI framework

### Text Processing
- **LangChain**: Text splitters (HTML, Recursive)
- **BeautifulSoup**: HTML parsing
- **Readability**: Content extraction

### Data Processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations

---

## 📦 Setup Instructions

### Prerequisites
- Python 3.9+
- Virtual environment
- API Keys:
  - Groq API Key ([Get it here](https://console.groq.com/))
  - Google Gemini API Key ([Get it here](https://aistudio.google.com/app/apikey))

### Step 1: Clone Repository
```bash
cd Week5/Chat_Assistant
```

### Step 2: Create Virtual Environment
```bash
cd ../..
python3 -m venv Week5/venv
source Week5/venv/bin/activate  # On Windows: Week5\venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
cd Week5
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in `Week5/Chat_Assistant/`:

```bash
GROQ_API_KEY="your_groq_api_key_here"
GOOGLE_API_KEY="your_google_api_key_here"
```

You can use the provided `.env.example` as a template:
```bash
cp Chat_Assistant/.env.example Chat_Assistant/.env
# Then edit .env and add your keys
```

### Step 5: Run the Application

**Terminal 1 - Start Backend:**
```bash
cd Week5/Chat_Assistant
source ../venv/bin/activate
python Chat_BackEnd.py
```

The backend will run on `http://localhost:4567`

**Terminal 2 - Start Frontend:**
```bash
cd Week5/Chat_Assistant
source ../venv/bin/activate
streamlit run Chat_FrontEnd.py
```

The frontend will open in your browser (usually `http://localhost:8501`)

---

## 🎮 Usage

### 1. Capture Knowledge

1. Open the Streamlit app in your browser
2. In the sidebar, enter:
   - **URL**: The webpage URL you want to capture (e.g., `https://example.com/article`)
   - **Table Name**: A name for your knowledge base table (e.g., `my_knowledge`)
3. Click **"Capture Knowledge"**
4. Wait for processing (this may take 30-60 seconds)
5. You'll see confirmation with number of chunks captured

### 2. Ask Questions

1. Once knowledge is captured, type your question in the chat input
2. The assistant will:
   - Transform your query into multiple variations
   - Search the vector database
   - Retrieve relevant context
   - Generate an answer based on the captured knowledge
3. Continue the conversation - chat history is maintained

### 3. Example Workflow

```
1. Capture: https://en.wikipedia.org/wiki/Artificial_intelligence
   Table: ai_knowledge

2. Query: "What is artificial intelligence?"
   → Assistant retrieves relevant chunks and answers

3. Follow-up: "What are its main applications?"
   → Assistant uses chat history + context for answer
```

---

## 🔌 API Endpoints

### Base URL
```
http://localhost:4567
```

### 1. Root Endpoint
```http
GET /
```
Returns API information and available endpoints.

**Response:**
```json
{
  "message": "Knowledge Assistant API is running!",
  "docs": "Visit /docs for interactive API documentation",
  "endpoints": {
    "POST /Capture_Knowledge": "Capture knowledge from a URL",
    "GET /Ask_Assistant": "Ask questions to the assistant"
  }
}
```

### 2. Capture Knowledge
```http
POST /Capture_Knowledge
Content-Type: application/json

{
  "url": "https://example.com/article",
  "table_name": "my_knowledge"
}
```

**Response:**
```json
{
  "Status": "Knowledge Captured",
  "Num_Chunks": "45",
  "Source": "example.com"
}
```

### 3. Ask Assistant
```http
GET /Ask_Assistant?Chat_Hist=[]&Query=What is AI?&table_name=my_knowledge
```

**Response:**
```json
"Artificial Intelligence (AI) refers to..."
```

**Parameters:**
- `Chat_Hist` (required): Chat history as string representation
- `Query` (required): User's question
- `table_name` (required): Name of the knowledge base table

---

## 🔍 How It Works

### Step-by-Step Process

#### Knowledge Capture Process

1. **URL Processing**
   - User provides URL and table name
   - Web scraper fetches the webpage
   - Removes navigation, headers, footers, cookie banners
   - Extracts main content

2. **Chunking Strategy**
   - First split by HTML headers (h1-h4) to preserve document structure
   - Then recursively split long chunks by sentences (300 chars)
   - Preserves metadata (source, topic/header)
   - Each chunk gets metadata about its section

3. **Embedding Generation**
   - Each chunk is converted to a vector using `all-MiniLM-L6-v2`
   - Embeddings capture semantic meaning
   - Vectors stored alongside metadata in LanceDB

4. **Storage**
   - All chunks stored in LanceDB table
   - Table named as specified by user
   - Ready for retrieval

#### Query Processing Process

1. **Query Transformation** (using Groq LLM)
   - Input: User query
   - Output: 4 queries total
     - 1 precise reformulation
     - 3 paraphrases

2. **Query Expansion** (using Groq LLM)
   - Input: User query
   - Output: 3 diverse variations
   - Expands search horizon while preserving intent

3. **Multi-Query Retrieval**
   - Total: 7 queries (4 transformed + 3 expanded)
   - Each query searched in vector database
   - Top 5 results per query (cosine similarity, threshold 0.6)
   - Results deduplicated

4. **Context Aggregation**
   - All retrieved chunks combined
   - Duplicates removed
   - Context sent to LLM

5. **Response Generation** (using Gemini 2.0)
   - Input:
     - System instruction (guidelines)
     - Chat history (last 2 exchanges)
     - Retrieved context
     - User query
   - Output: Natural language response in Markdown format

### Why RAG Fusion?

Traditional RAG uses a single query. RAG Fusion improves retrieval by:
- **Covering different angles**: Paraphrases catch information phrased differently
- **Expanding scope**: Alternate queries find related information
- **Better recall**: More queries = more relevant chunks found
- **Deduplication**: Same information from multiple queries is consolidated

---

## 📊 Data Flow Diagram

```
┌──────────────┐
│  User Query  │
└──────┬───────┘
       │
       ├─────────────────────────────┐
       │                             │
       ▼                             ▼
┌──────────────┐            ┌──────────────┐
│ Transform    │            │   Expand     │
│ Query        │            │   Query      │
│ (Groq)       │            │   (Groq)     │
└──────┬───────┘            └──────┬───────┘
       │                           │
       │ 4 queries                 │ 3 queries
       │                           │
       └───────────┬───────────────┘
                   │
                   │ 7 total queries
                   ▼
        ┌──────────────────────┐
        │  Vector Search       │
        │  (LanceDB)           │
        │  - Cosine similarity │
        │  - Top 5 per query   │
        └──────────┬───────────┘
                   │
                   │ Context chunks
                   ▼
        ┌──────────────────────┐
        │  Deduplicate         │
        └──────────┬───────────┘
                   │
                   │ Final context
                   ▼
        ┌──────────────────────┐
        │  Generate Response   │
        │  (Gemini 2.0)        │
        │  + Chat History      │
        └──────────┬───────────┘
                   │
                   ▼
            User receives answer
```

---

## 🎯 Key Features Explained

### 1. Intelligent Chunking
- **Header-based splitting**: Preserves document structure
- **Recursive splitting**: Handles long paragraphs
- **Metadata preservation**: Each chunk knows its topic/section
- **Optimal size**: 300 characters for balance between context and precision

### 2. RAG Fusion Retrieval
- **Multiple query perspectives**: 7 different ways to ask the same question
- **Comprehensive coverage**: Finds information that might be missed with single query
- **Quality filtering**: Cosine similarity threshold (0.6) ensures relevance
- **Deduplication**: Prevents redundant context

### 3. Context-Aware Generation
- **Chat history**: Maintains conversation context
- **System instructions**: Guides LLM behavior
- **Natural responses**: Doesn't sound like it's referencing a context

### 4. Robust Web Scraping
- **Content extraction**: Focuses on main content
- **Noise removal**: Eliminates navigation, ads, cookie banners
- **Fallback methods**: Handles various HTML structures

---

## 📝 Notes

- **Vector Database Location**: LanceDB creates a `Quick_Ref` directory locally
- **Chat History**: Last 2 exchanges (4 messages) are sent to LLM
- **Embedding Model**: Uses `all-MiniLM-L6-v2` (384 dimensions, fast and efficient)
- **Chunk Size**: 300 characters with 0 overlap
- **Retrieval**: Top 5 chunks per query, maximum distance 0.6 (cosine)

---

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError
**Solution**: Make sure virtual environment is activated and dependencies are installed
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: API Key Errors
**Solution**: Check `.env` file has correct keys
```bash
cat Chat_Assistant/.env
```

### Issue: Backend Connection Failed
**Solution**: Ensure backend is running on port 4567
```bash
python Chat_BackEnd.py
```

### Issue: No chunks captured
**Solution**: 
- Check if URL is accessible
- Verify web scraper can access the content
- Check for JavaScript-heavy sites (may need Selenium)

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
- [Sentence Transformers](https://www.sbert.net/)
- [RAG Fusion Paper](https://arxiv.org/abs/2307.15217)

---

## 👥 Contributing

Feel free to improve this project by:
- Adding support for more document types (PDF, DOCX)
- Implementing different embedding models
- Adding response evaluation metrics
- Improving the UI/UX

---

## 📄 License

This project is part of the GenAI Engineering Cohort 3 coursework.

---

**Built with ❤️ for GenAI Engineering Cohort 3**

