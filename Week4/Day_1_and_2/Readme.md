# Week 4 - Day 1 & 2: LLM Fundamentals and RAG Pipeline

This directory contains notebooks and applications demonstrating LLM capabilities, limitations, vector embeddings, databases, and RAG (Retrieval-Augmented Generation) techniques.

## 📚 Notebooks Overview

### 1. `1_llm_capabilities.ipynb` - LLM Capabilities Demonstration

**Purpose**: Comprehensive tutorial demonstrating various capabilities of Large Language Models using Mistral's API.

**What it covers**:
- **Prompt Engineering Techniques**
  - Zero-shot prompting (direct instructions without examples)
  - Few-shot prompting (providing examples to guide the model)
  - Chain-of-thought prompting (step-by-step reasoning)
  - Role-based prompting (assigning specific personas)

- **Text Generation**
  - Creative writing
  - Technical documentation generation
  - Summarization

- **Code Generation**
  - Code writing and generation
  - Code debugging
  - Code documentation

- **Language Tasks**
  - Translation between languages
  - Question answering
  - Information extraction

- **Structured Output**
  - JSON generation
  - Data extraction and formatting

**Setup Required**:
- Install: `pip install mistralai python-dotenv`
- Create `.env` file with: `MISTRAL_API_KEY=your_api_key_here`

**In Short**: A hands-on guide showing how to use Mistral AI for various LLM tasks with working code examples.

---

### 2. `2_llm_limitations.ipynb` - LLM Limitations

Demonstrates the limitations and challenges of Large Language Models, including:
- Hallucinations
- Context window limitations
- Token limits
- Bias and ethical concerns
- Reasoning limitations

---

### 3. `3_vector_embeddings.ipynb` - Vector Embeddings

Explores vector embeddings for semantic search:
- What are embeddings?
- How to generate embeddings
- Embedding models comparison
- Similarity calculations
- Use cases for embeddings

---

### 4. `4_vector_databases.ipynb` - Vector Databases

Introduction to vector databases:
- FAISS (Facebook AI Similarity Search)
- LanceDB
- Storing and retrieving vectors
- Similarity search operations
- Performance considerations

---

### 5. `5_search_engine_app.py` - Search Engine Application

A search engine application that:
- Indexes documents using vector embeddings
- Performs semantic search
- Returns relevant results based on query similarity

**Related**: `5_search_techniques.md` - Documentation on search techniques

---

### 6. `6_chunking_strategies.ipynb` - Chunking Strategies

Explores different text chunking strategies for RAG:
- Fixed-size chunking
- Sentence-based chunking
- Semantic chunking
- Header-based chunking
- Overlap strategies

**Related**: 
- `6_retrieval_strategies.ipynb` - Different retrieval strategies
- `6_RAG Pipeline Visualization.html` - Visual representation of RAG pipeline

---

### 7. `7_chat_app.py` - Chat Application

A complete chat application demonstrating RAG implementation with:
- Document ingestion
- Vector storage
- Context retrieval
- LLM-based response generation

---

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install streamlit
pip install mistralai python-dotenv
pip install sentence-transformers
pip install lancedb faiss-cpu
```

### Setup

1. Create a `.env` file in this directory with your API keys:
```
MISTRAL_API_KEY=your_mistral_api_key
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

2. Run notebooks in order (1 → 2 → 3 → 4 → 6 → 7) for a complete learning path

3. For applications:
   - `python 5_search_engine_app.py` - Run search engine
   - `streamlit run 7_chat_app.py` - Run chat application

---

## 📁 Additional Files

- **PDF Documents**: Sample documents for testing (IndianBudget2025.pdf, NASDAQ reports, etc.)
- **sample_data/**: Sample text files for experimentation
- **vector_dbs/**: Pre-built vector database files (FAISS and LanceDB)

---

## 🎯 Learning Path

1. **Start with** `1_llm_capabilities.ipynb` - Understand what LLMs can do
2. **Then** `2_llm_limitations.ipynb` - Understand their limitations
3. **Learn** `3_vector_embeddings.ipynb` - How to represent text as vectors
4. **Explore** `4_vector_databases.ipynb` - How to store and search vectors
5. **Study** `6_chunking_strategies.ipynb` - How to prepare documents for RAG
6. **Build** `7_chat_app.py` - Complete RAG application

---

## 📝 Notes

- All notebooks use Mistral AI by default, but can be adapted for other LLM providers
- Vector databases are stored locally in the `vector_dbs/` directory
- Sample documents are provided for testing purposes

---

**Part of GenAI Engineering Cohort 3**
