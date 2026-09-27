# 🤖 Agentic AI RAG Assistant

An **Agentic AI and Generative AI application** that uses **Retrieval-Augmented Generation (RAG)** to answer questions from a document-based knowledge base.

The system combines **Large Language Models (LLMs), AI agents, tool calling, semantic search, keyword search, and hybrid retrieval** to retrieve relevant information before generating a response.

---

## 📌 Project Overview

The **Agentic AI RAG Assistant** is a document-based question-answering system designed to demonstrate the integration of **Agentic AI, Generative AI, RAG, LLMs, embeddings, vector search, and keyword retrieval**.

Instead of relying only on the language model's pretrained knowledge, the system first searches a collection of documents and provides relevant document chunks to the language model.

### Core Workflow

```text
User Question
      ↓
Agentic AI Agent
      ↓
Knowledge Base Search Tool
      ↓
Hybrid Retrieval
      ↓
┌──────────────────────┐
│    Vector Search     │
│          +           │
│    Keyword Search    │
└──────────────────────┘
      ↓
Relevant Document Chunks
      ↓
Qwen2.5-1.5B-Instruct
      ↓
Grounded Response
      ↓
Web Interface
```

---

# 🧠 AI Project Classification

This project combines multiple areas of Artificial Intelligence:

- **Agentic AI**
- **Generative AI**
- **Retrieval-Augmented Generation (RAG)**
- **Large Language Models (LLMs)**
- **Natural Language Processing (NLP)**
- **Information Retrieval**
- **Semantic Search**
- **Vector Search**
- **Hybrid Search**

---

# ✨ Key Features

## 📚 Document-Based Knowledge Base

The system uses PDF documents as its external knowledge source.

Current knowledge-base documents include:

- `ai_agents.pdf`
- `embeddings.pdf`
- `large_language_models.pdf`
- `retrieval_augmented_generation.pdf`
- `vector_search.pdf`

These documents are processed and indexed so that users can ask questions about their content.

---

## 📄 PDF Document Processing

The document processing pipeline extracts text from PDF documents and prepares the content for retrieval.

```text
PDF Document
      ↓
Document Loader
      ↓
Text Extraction
      ↓
Text Chunking
      ↓
Document Chunks
```

---

## ✂️ Document Chunking

Large documents are divided into smaller text chunks.

Chunking allows the retrieval system to search specific portions of documents instead of processing the complete document for every question.

```text
Large Document
      ↓
Text
      ↓
Smaller Chunks
      ↓
Embedding + Indexing
```

---

## 🧮 Text Embeddings

The project uses the Sentence Transformers model:

```text
all-MiniLM-L6-v2
```

The model converts document chunks and user queries into numerical vector representations.

The generated embeddings contain:

```text
384 dimensions
```

The embeddings are then used for semantic similarity search.

```text
Text
 ↓
Sentence Transformer
 ↓
384-Dimensional Embedding
 ↓
Vector Search
```

---

## 🔎 Vector Search

The project uses **FAISS** for vector similarity search.

Vector search retrieves information based on the semantic similarity between the user's question and document chunks.

```text
User Question
      ↓
Query Embedding
      ↓
FAISS
      ↓
Similar Document Chunks
```

---

## 🔤 Keyword Search

The project also performs keyword-based retrieval.

Keyword search helps retrieve document chunks containing important terms from the user's query.

This provides a complementary retrieval method to semantic vector search.

---

## 🔀 Hybrid Search

The system combines vector search and keyword search.

```text
             User Query
                 ↓
        ┌────────┴────────┐
        ↓                 ↓
 Vector Search       Keyword Search
     FAISS                 BM25
        ↓                 ↓
        └────────┬────────┘
                 ↓
          Hybrid Retrieval
                 ↓
       Relevant Document Chunks
```

Current retrieval weighting:

```text
Vector Search  = 70%
Keyword Search = 30%
```

The hybrid approach combines semantic similarity with keyword relevance.

---

# 🤖 Agentic AI

The project uses **Smolagents CodeAgent** as the agent layer.

The CodeAgent acts as the orchestrator between the user question, knowledge-base search tool, retrieved context, and language model.

```text
User
 ↓
CodeAgent
 ↓
Knowledge Base Tool
 ↓
Retriever
 ↓
Relevant Context
 ↓
LLM
 ↓
Final Answer
```

---

# 🛠️ Knowledge Base Search Tool

A custom tool named:

```python
knowledge_base_search(query)
```

is implemented using Smolagents.

The tool connects the agent to the project's retrieval system.

### Responsibilities

The tool:

1. Receives the user's query.
2. Sends the query to the retriever.
3. Performs hybrid retrieval.
4. Retrieves relevant document chunks.
5. Returns the retrieved context to the agent.

```text
CodeAgent
    ↓
knowledge_base_search()
    ↓
Retriever
    ↓
Hybrid Search
    ↓
Document Context
    ↓
CodeAgent
```

---

# 🔍 Retriever

The retriever acts as the connection between the agent tool and the underlying search systems.

```text
User Query
     ↓
Retriever
     ↓
Hybrid Search
   ↙       ↘
FAISS     BM25
   ↘       ↙
Relevant Results
     ↓
Context
```

The retriever combines the results from vector and keyword search using their chunk indexes and hybrid scores.

---

# 🧠 Large Language Model

The project uses:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

through the Hugging Face Transformers library.

The language model is responsible for generating the final response using the user's question and retrieved knowledge-base context.

```text
User Question
      +
Retrieved Context
      ↓
Qwen2.5-1.5B-Instruct
      ↓
Generated Response
```

---

# 🏗️ Complete System Architecture

```text
                         USER
                           │
                           ▼
                  ┌─────────────────┐
                  │  Web Interface  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │    CodeAgent    │
                  │   Smolagents    │
                  └────────┬────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ knowledge_base_search() │
              └────────────┬────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │    Retriever    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Hybrid Search  │
                  └────────┬────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
          ┌─────────────┐     ┌─────────────┐
          │ Vector      │     │ Keyword     │
          │ Search      │     │ Search      │
          │ FAISS       │     │ BM25        │
          └──────┬──────┘     └──────┬──────┘
                 │                   │
                 └─────────┬─────────┘
                           ▼
                 Relevant Document
                      Chunks
                           │
                           ▼
                ┌────────────────────┐
                │ Qwen2.5-1.5B       │
                │    Instruct        │
                └─────────┬──────────┘
                          │
                          ▼
                  Final Response
```

---

# 📂 Project Structure

```text
Agentic_Knowledge_Base/
│
├── .env
├── .gitignore
├── agent.py
├── app.py
├── backend.py
├── chunking.py
├── document_loader.py
├── embeddings.py
├── hybrid_search.py
├── keyword_search.py
├── knowledge_base_tool.py
├── retriever.py
├── vector_search.py
├── requirements.txt
│
├── frontend/
│   ├── index.html
│   └── README.md
│
└── data/
    └── documents/
        ├── ai_agents.pdf
        ├── embeddings.pdf
        ├── large_language_models.pdf
        ├── retrieval_augmented_generation.pdf
        └── vector_search.pdf
```

---

# 📁 Main Project Files

## `agent.py`

Contains the agent and LLM integration.

Responsibilities include:

- Loading the language model
- Creating the Smolagents CodeAgent
- Connecting the knowledge-base tool
- Sending prompts to the LLM
- Generating final responses
- Providing the `ask_question()` function for application use

---

## `knowledge_base_tool.py`

Contains the custom knowledge-base tool:

```python
knowledge_base_search(query)
```

This provides the agent with access to the retrieval system.

---

## `retriever.py`

Acts as the main retrieval interface.

It receives a query and retrieves relevant information from the hybrid search system.

---

## `hybrid_search.py`

Combines:

- Vector search
- Keyword search

The results are combined using weighted retrieval.

```text
Vector Search
     +
Keyword Search
     ↓
Hybrid Search
```

---

## `vector_search.py`

Handles semantic vector retrieval using FAISS.

The user query is converted into an embedding and compared with stored document embeddings.

---

## `keyword_search.py`

Handles keyword-based retrieval using BM25.

This provides lexical matching in addition to semantic retrieval.

---

## `embeddings.py`

Responsible for generating document embeddings using:

```text
all-MiniLM-L6-v2
```

The project also uses embedding caching to avoid unnecessarily regenerating existing embeddings.

---

## `document_loader.py`

Responsible for loading and extracting content from documents.

---

## `chunking.py`

Responsible for splitting extracted document text into smaller chunks for retrieval.

---

## `app.py`

Contains the web application interface used to interact with the Agentic RAG Assistant.

---

## `backend.py`

Provides backend application functionality for connecting application requests with the AI system.

---

# 🔄 End-to-End Workflow

The complete process works as follows.

### Step 1 — User asks a question

Example:

```text
What are the benefits of RAG?
```

### Step 2 — Agent receives the question

The Smolagents CodeAgent receives and processes the user's question.

### Step 3 — Agent calls the knowledge-base tool

```python
knowledge_base_search(query)
```

### Step 4 — Retriever processes the query

The retriever sends the query through the hybrid retrieval pipeline.

### Step 5 — Vector search

The query is converted into an embedding and searched against the FAISS index.

### Step 6 — Keyword search

The query is also processed through BM25 keyword retrieval.

### Step 7 — Hybrid retrieval

The vector and keyword results are combined.

### Step 8 — Relevant context is returned

The most relevant document chunks are provided to the agent.

### Step 9 — LLM generates the answer

Qwen2.5-1.5B-Instruct receives the question and retrieved context.

### Step 10 — Final response

The generated response is returned to the user through the application interface.

---

# 🔁 Complete Data Flow

```text
PDF Documents
      ↓
Document Loader
      ↓
Text Extraction
      ↓
Chunking
      ↓
Document Chunks
      ↓
Sentence Transformer
      ↓
Embeddings
      ↓
┌───────────────────┐
│   FAISS Index     │
│        +          │
│   BM25 Index      │
└─────────┬─────────┘
          │
          ▼
     Hybrid Search
          │
          ▼
   Relevant Chunks
          │
          ▼
 Knowledge Base Tool
          │
          ▼
      CodeAgent
          │
          ▼
 Qwen2.5-1.5B-Instruct
          │
          ▼
    Final Response
```

---

# 🖥️ Frontend

The project includes a web-based interface for interacting with the AI assistant.

The interface is designed around a document-aware AI assistant experience.

### Main interface components

- Chat interface
- Question input
- Recent questions
- Knowledge-base information
- Document information
- Example questions
- Retrieved source information
- AI-generated responses

---

# 📚 Knowledge Base

The current knowledge base contains five sample PDF documents:

| Document | Purpose |
|---|---|
| `ai_agents.pdf` | Information about AI agents |
| `embeddings.pdf` | Information about embeddings |
| `large_language_models.pdf` | Information about LLMs |
| `retrieval_augmented_generation.pdf` | Information about RAG |
| `vector_search.pdf` | Information about vector search |

---

# 🧮 Embedding Pipeline

```text
Document Text
     ↓
Chunking
     ↓
all-MiniLM-L6-v2
     ↓
384-Dimensional Embedding
     ↓
Embedding Cache
     ↓
FAISS
```

The embedding cache is stored locally under the project's cache directory.

---

# 🔎 Vector Search Pipeline

```text
User Query
     ↓
Query Embedding
     ↓
FAISS Similarity Search
     ↓
Top Relevant Chunks
     ↓
Chunk Index + Score
```

---

# 🔤 Keyword Search Pipeline

```text
User Query
     ↓
BM25
     ↓
Keyword Relevance
     ↓
Relevant Chunks
     ↓
Chunk Index + Score
```

---

# 🔀 Hybrid Retrieval Pipeline

```text
                 User Query
                     │
            ┌────────┴────────┐
            ▼                 ▼
      Vector Search      Keyword Search
          FAISS               BM25
            │                 │
            └────────┬────────┘
                     ▼
              Combine Results
                     ↓
              Weighted Scoring
                     ↓
             Relevant Chunks
```

Current weighting:

```text
Vector Search  → 0.7
Keyword Search → 0.3
```

---

# 🛠️ Technologies Used

| Category | Technology |
|---|---|
| Programming Language | Python |
| Agent Framework | Smolagents |
| Agent | CodeAgent |
| LLM | Qwen2.5-1.5B-Instruct |
| LLM Platform | Hugging Face |
| LLM Library | Transformers |
| RAG | Retrieval-Augmented Generation |
| Embedding Framework | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Search | FAISS |
| Keyword Search | BM25 |
| Retrieval | Hybrid Search |
| Frontend | Streamlit / Web Interface |
| Environment Variables | python-dotenv |
| Version Control | Git |
| Repository Hosting | GitHub |

---

# 📦 Requirements

The main Python dependencies are managed through:

```text
requirements.txt
```

The project requires Python and the libraries needed for:

- LLM inference
- Embeddings
- Vector search
- Keyword search
- Agent orchestration
- Web interface

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/jiss05/Agentic-AI-RAG-Assistant.git
```

## 2. Enter the Project Directory

```bash
cd Agentic-AI-RAG-Assistant
```

## 3. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

The project uses environment variables for sensitive credentials.

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token
```

The Hugging Face token is used for Hugging Face model access.

### Security

Never:

- Hardcode API tokens
- Commit `.env`
- Publish API keys
- Share private credentials

The project `.gitignore` excludes the `.env` file.

---

# 🧑‍💻 Running the Agent Directly

The agent can also be accessed programmatically through:

```python
from agent import ask_question

answer = ask_question("What is RAG?")

print(answer)
```

The `ask_question()` function sends the question through the Agentic RAG pipeline and returns the final result.

---

# 🧠 Agent Architecture

The agent layer can be understood using the following components:

```text
CodeAgent
    │
    ├── User Question
    │
    ├── Tool Selection
    │
    ├── knowledge_base_search()
    │
    ├── Retrieved Context
    │
    └── Final Response
```

### Component Roles

| Component | Responsibility |
|---|---|
| CodeAgent | Orchestrates the workflow |
| Knowledge Base Tool | Provides retrieval access |
| Retriever | Retrieves relevant chunks |
| FAISS | Semantic vector search |
| BM25 | Keyword search |
| Qwen | Generates response |
| Web Interface | User interaction |

---

## 📊 Retrieval Evaluation

The retrieval system can be evaluated using metrics such as:

- Retrieval accuracy
- Precision
- Recall
- Context relevance
- Answer relevance
- Grounding quality

---

# 🎯 Project Objective

The main objective of this project is to demonstrate how **Agentic AI and Retrieval-Augmented Generation can be combined with Large Language Models to create a document-aware AI assistant**.

The project brings together:

```text
LLM
+
Agent
+
Tool Calling
+
RAG
+
Embeddings
+
Vector Search
+
Keyword Search
+
Hybrid Retrieval
+
Document Knowledge Base
```

---

# 🎓 Learning Outcomes

This project provides practical experience with:

- Agentic AI
- Generative AI
- Retrieval-Augmented Generation
- Large Language Models
- Prompt Engineering
- Tool Calling
- Semantic Search
- Vector Embeddings
- FAISS
- BM25
- Hybrid Retrieval
- Document Processing
- Context Retrieval
- LLM Grounding
- Python AI Development
- AI Application Architecture

---
