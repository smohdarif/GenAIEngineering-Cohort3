# RAG (Retrieval Augmented Generation) Concepts

This document explains how RAG works in the Week12 phidata examples.

## Overview

RAG enhances LLM responses by retrieving relevant information from a knowledge base before generating answers. This allows the AI to answer questions based on your custom data.

## Code Breakdown

```python
# Create a knowledge base for the Agent
knowledge_base = AgentKnowledge(vector_db=LanceDb(
        table_name="custom_knowledge",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=SentenceTransformerEmbedder(model="all-MiniLM-L6-v2") 
    ),)

# Add information to the knowledge base
knowledge_base.load_text("The sky is green")

# Add the knowledge base to the Agent
agent = Agent(knowledge=knowledge_base, search_knowledge=True)
```

## How It Works

### 1. Where Data is Saved

```python
uri="tmp/lancedb"
table_name="custom_knowledge"
```

- Data is stored in a **LanceDB** vector database at `tmp/lancedb` folder
- This is a local file-based database (fast and lightweight)
- The table name is `custom_knowledge`

### 2. How Data is Stored (Embeddings)

```python
embedder=SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
```

When you call `knowledge_base.load_text("The sky is green")`:

1. The text is converted to **embeddings** (vectors of ~384 numbers)
2. The `all-MiniLM-L6-v2` model creates these embeddings
3. Both the text and its vector are stored in LanceDB

### 3. How Data is Retrieved (Vector Search)

```python
search_type=SearchType.vector
search_knowledge=True
```

When the agent gets a question:

1. The question is converted to a vector (same embedder)
2. **Vector similarity search** finds the closest stored vectors
3. The matching text is retrieved and given to the LLM as context

## Visual Flow

### Storing Data

```
┌─────────────────────────────────────────────────────────────┐
│                    STORING DATA                              │
├─────────────────────────────────────────────────────────────┤
│  "The sky is green"                                          │
│         ↓                                                    │
│  SentenceTransformer (all-MiniLM-L6-v2)                     │
│         ↓                                                    │
│  [0.12, -0.34, 0.56, ...] (384 dim vector)                  │
│         ↓                                                    │
│  Saved to: tmp/lancedb/custom_knowledge                     │
└─────────────────────────────────────────────────────────────┘
```

### Retrieving Data

```
┌─────────────────────────────────────────────────────────────┐
│                   RETRIEVING DATA                            │
├─────────────────────────────────────────────────────────────┤
│  User asks: "What color is the sky?"                         │
│         ↓                                                    │
│  Question → Vector [0.15, -0.32, 0.58, ...]                 │
│         ↓                                                    │
│  Vector similarity search in LanceDB                        │
│         ↓                                                    │
│  Finds: "The sky is green" (closest match)                  │
│         ↓                                                    │
│  LLM answers using this context                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

| Component | Purpose |
|-----------|---------|
| `AgentKnowledge` | Container for the knowledge base |
| `LanceDb` | Vector database for storing embeddings |
| `SentenceTransformerEmbedder` | Converts text to vectors (embeddings) |
| `SearchType.vector` | Uses cosine similarity for retrieval |
| `search_knowledge=True` | Enables the agent to search the knowledge base |

## Why Use RAG?

1. **Custom Knowledge**: Add your own documents, facts, or data
2. **Up-to-date Info**: Knowledge base can be updated without retraining the LLM
3. **Accurate Answers**: Reduces hallucinations by grounding responses in real data
4. **Cost Effective**: No need to fine-tune expensive models

## Example Usage

```python
# Load a PDF document
knowledge_base.load_pdf("company_handbook.pdf")

# Load from URL
knowledge_base.load_url("https://example.com/docs")

# Load text directly
knowledge_base.load_text("Important fact: Our office is in Mumbai")

# Now the agent can answer questions about this knowledge
agent.print_response("Where is the office located?")
```

## Vector Database Options

The phidata framework supports multiple vector databases:

- **LanceDB** (local, file-based) - Used in examples
- **ChromaDB** (local or server)
- **Pinecone** (cloud)
- **Qdrant** (local or cloud)
- **Weaviate** (local or cloud)

## Embedding Models

Common embedding models:

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good |
| `all-mpnet-base-v2` | 768 | Medium | Better |
| `text-embedding-3-small` (OpenAI) | 1536 | API | Excellent |
| `text-embedding-3-large` (OpenAI) | 3072 | API | Best |

