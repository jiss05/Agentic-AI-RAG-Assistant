# 🤖 Agentic AI RAG Assistant

An **Agentic AI and Generative AI application** that uses **Retrieval-Augmented Generation (RAG)** to answer questions from a document-based knowledge base.

The system combines an **LLM, agent-based tool execution, semantic search, keyword search, and hybrid retrieval** to retrieve relevant information before generating responses.

---

## 📌 Project Overview

The **Agentic AI RAG Assistant** allows users to ask questions about a collection of documents and receive answers based on information retrieved from the knowledge base.

Instead of relying only on the language model's pretrained knowledge, the system first searches the available documents and provides the retrieved context to the LLM.

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
┌───────────────────┐
│ Vector Search     │
│       +           │
│ Keyword Search    │
└───────────────────┘
      ↓
Relevant Document Chunks
      ↓
Qwen2.5-1.5B-Instruct
      ↓
Grounded Response
      ↓
Web Interface
