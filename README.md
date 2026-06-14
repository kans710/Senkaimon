# Senkaimon - RAG Based PDF Interaction System

## Overview

Senkaimon is a Retrieval-Augmented Generation (RAG) application that enables users to interact with multiple PDF documents through natural language queries. The system leverages Large Language Models (LLMs), vector embeddings, and semantic search to retrieve contextually relevant information from uploaded documents and generate accurate responses.

Unlike traditional keyword-based document search systems, Senkaimon utilizes vector embeddings and Retrieval-Augmented Generation to understand the semantic meaning of user queries, resulting in more precise and context-aware answers.

The application is designed to run entirely on local hardware, ensuring privacy, reduced operational costs, and offline accessibility without relying on external cloud-based AI services.

---

## Features

* Upload and process multiple PDF documents simultaneously
* Natural language question-answering over document collections
* Semantic search using vector embeddings
* Context-aware response generation through RAG architecture
* Local deployment with no dependency on paid APIs
* Interactive web interface built with Streamlit
* Efficient document indexing using ChromaDB
* Privacy-focused architecture with local inference

---

## Technology Stack

### Frontend

* Streamlit

### Backend & AI Framework

* Python
* LangChain

### Language Model

* Llama 3.2 (via Ollama)

### Embedding Model

* Nomic-Embed-Text

### Vector Database

* ChromaDB

### Optimization

* ONNX Runtime

---

## System Architecture

```text
PDF Documents
      │
      ▼
Document Loader
      │
      ▼
Text Chunking
      │
      ▼
Embedding Generation
(Nomic-Embed-Text)
      │
      ▼
Vector Storage
(ChromaDB)
      │
      ▼
User Query
      │
      ▼
Similarity Search
      │
      ▼
Relevant Chunks Retrieved
      │
      ▼
Llama 3.2 (Ollama)
      │
      ▼
Generated Response
      │
      ▼
Streamlit Interface
```

---

## How It Works

### 1. Document Processing

Uploaded PDF documents are parsed and converted into machine-readable text.

The extracted content is divided into smaller chunks to improve retrieval efficiency and maintain contextual relevance during semantic search.

---

### 2. Embedding Generation

Each text chunk is transformed into a dense vector representation using the Nomic-Embed-Text embedding model.

These embeddings capture semantic meaning rather than simple keyword matching.

---

### 3. Vector Storage

Generated embeddings are stored in ChromaDB, a vector database optimized for similarity search operations.

This allows efficient retrieval of relevant document sections based on user intent.

---

### 4. Query Processing

When a user submits a question:

* The query is converted into an embedding.
* ChromaDB performs similarity search.
* The most relevant document chunks are retrieved.

---

### 5. Response Generation

Retrieved document chunks are provided as contextual information to Llama 3.2 running locally through Ollama.

The model generates responses grounded in the retrieved content, reducing hallucinations and improving factual accuracy.

---

## Why Retrieval-Augmented Generation (RAG)?

Traditional language models rely solely on their pre-trained knowledge and may generate inaccurate or outdated responses.

RAG improves reliability by retrieving relevant information from source documents before generating an answer.

Benefits include:

* Improved factual accuracy
* Reduced hallucinations
* Ability to work with private documents
* No model retraining required
* Dynamic knowledge updates

---

## Key Challenges Addressed

### Semantic Document Search

Implemented vector embeddings to enable meaning-based retrieval instead of traditional keyword matching.

### Local AI Deployment

Configured Llama 3.2 and embedding models through Ollama to eliminate reliance on cloud APIs and preserve user privacy.

### Multi-PDF Context Management

Designed the retrieval pipeline to work across multiple documents simultaneously while maintaining response relevance.

### Efficient Inference

Integrated ONNX Runtime optimizations to improve execution efficiency on consumer hardware.

---

## Future Improvements

* Chat history and conversational memory
* Source citations for generated responses
* Support for DOCX and TXT files
* OCR support for scanned PDFs
* User authentication and document management
* Hybrid search combining semantic and keyword retrieval
* Document summarization capabilities

---

## Learning Outcomes

Through this project, I gained hands-on experience with:

* Retrieval-Augmented Generation (RAG)
* Large Language Model integration
* LangChain pipelines
* Vector databases and semantic search
* Embedding models
* Local AI deployment with Ollama
* Streamlit application development
* ONNX optimization techniques

---

## Author

Karan Singh

B.Tech Computer Science (AI & ML)

Dronacharya Group of Institutions
