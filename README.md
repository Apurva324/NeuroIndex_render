---
title: NeuroIndex
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---
# NeuroIndex - Lightweight Vector Database & RAG Engine

This document explains **everything** about NeuroIndex - from vector
embeddings and similarity search to the complete code architecture,
retrieval pipeline, evaluation system, and interactive visualization.

After reading this, you should understand exactly how a query moves
through NeuroIndex, how vectors are stored and searched, how BM25 and
hybrid retrieval work, and how the RAG assistant generates answers from
retrieved documents.

------------------------------------------------------------------------

## Table of Contents

1.  [What is NeuroIndex?](#1-what-is-neuroindex)
2.  [Vector Search Background](#2-vector-search-background)
3.  [Project Overview](#3-project-overview)
4.  [File Structure](#4-file-structure)
5.  [The Journey of a Query](#5-the-journey-of-a-query)
6.  [Deep Dive: Each Component](#6-deep-dive-each-component)
7.  [How HNSW Search Works](#7-how-hnsw-search-works)
8.  [How KD-Tree Search Works](#8-how-kd-tree-search-works)
9.  [How BM25 and Hybrid Search
    Work](#9-how-bm25-and-hybrid-search-work)
10. [How RAG and Ask AI Work](#10-how-rag-and-ask-ai-work)
11. [Interactive Vector
    Visualization](#11-interactive-vector-visualization)
12. [Evaluation and Benchmarking](#12-evaluation-and-benchmarking)
13. [Persistence and Storage](#13-persistence-and-storage)
14. [Building and Running](#14-building-and-running)
15. [Understanding the Output](#15-understanding-the-output)
16. [Extending the Project](#16-extending-the-project)

------------------------------------------------------------------------

## 1. What is NeuroIndex?

**NeuroIndex** is a lightweight vector database and retrieval-augmented
generation (RAG) system built in Python.

Instead of relying entirely on an external vector database, NeuroIndex
implements the core retrieval infrastructure itself:

-   Vector storage
-   Cosine similarity
-   Brute-force nearest-neighbor search
-   KD-Tree search
-   HNSW approximate nearest-neighbor search
-   BM25 lexical retrieval
-   Hybrid retrieval
-   Reciprocal Rank Fusion
-   SQLite persistence
-   Document storage
-   Ollama embeddings
-   Ollama LLM generation
-   Retrieval evaluation
-   Interactive 2D PCA visualization

### What NeuroIndex Does

``` text
User Query
    │
    ▼
┌─────────────────────┐
│ Query Embedding     │
│ Ollama              │
│ nomic-embed-text    │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Retrieval                        │
│                                  │
│ Vector Search                    │
│ BM25 Search                      │
│ Hybrid Search                    │
│ HNSW / KD-Tree / Brute Force     │
└──────────┬───────────────────────┘
           │
           ▼
┌─────────────────────┐
│ Top-K Documents     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Llama 3.2           │
│ RAG Generation      │
└──────────┬──────────┘
           │
           ▼
       Final Answer
```

------------------------------------------------------------------------

## 2. Vector Search Background

### What is an Embedding?

An embedding converts text into a numerical vector.

For example:

``` text
"binary search tree"
            │
            ▼
      Embedding Model
            │
            ▼
[0.012, -0.381, 0.104, ...]
            │
            ▼
        768 dimensions
```

NeuroIndex uses:

``` text
nomic-embed-text
```

The embedding dimension is:

``` text
768
```

The important idea is that semantically similar text should produce
vectors that are close together in vector space.

### Cosine Similarity

NeuroIndex supports cosine similarity/distance.

Conceptually:

``` text
              A · B
cosine = ─────────────
          ||A|| ||B||
```

A value closer to `1` means the vectors point in similar directions.

This allows NeuroIndex to answer semantic questions such as:

``` text
Query:
"tree based searching"

      ↓

Closest documents:

Binary Search Tree
KD-Tree
Vector Search
...
```

even when the query does not contain the exact same words.

### Vector Search vs Keyword Search

Keyword search:

``` text
"binary tree"
      ↓
Find documents containing:
"binary" and "tree"
```

Vector search:

``` text
"binary tree"
      ↓
Embedding
      ↓
Find semantically similar concepts
```

NeuroIndex supports both approaches and can combine them.

------------------------------------------------------------------------

## 3. Project Overview

The system has three major layers:

``` text
┌────────────────────────────────────────────────────┐
│                    Web Interface                    │
│                  index.html                         │
└─────────────────────────┬──────────────────────────┘
                          │ HTTP
                          ▼
┌────────────────────────────────────────────────────┐
│                    API Server                       │
│                  server.py                          │
└─────────────────────────┬──────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────┐
│                 Retrieval Engine                    │
│                  vectordb.py                        │
│                                                    │
│ HNSW │ KD-Tree │ Brute Force │ BM25 │ Hybrid      │
└─────────────────────────┬──────────────────────────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
      ┌──────────────┐         ┌──────────────┐
      │ SQLite       │         │ Ollama       │
      │ Persistence  │         │ Embeddings   │
      │ + Documents  │         │ + LLM        │
      └──────────────┘         └──────────────┘
```

### Current Demo Dataset

The demo environment contains:

``` text
20 vectors
31 documents
768 dimensions
Cosine metric
```

The vector demo includes topics such as:

-   Linked Lists
-   Binary Search Trees
-   Dynamic Programming
-   Graph BFS and DFS
-   Hash Tables
-   Calculus
-   Linear Algebra
-   Probability
-   Machine Learning
-   KD-Trees
-   BM25
-   Vector Search
-   Hybrid Search
-   Reciprocal Rank Fusion

------------------------------------------------------------------------

## 4. File Structure

``` text
NeuroIndex/
│
├── index.html                  # Interactive web UI
├── server.py                   # HTTP API server
├── vectordb.py                 # Main vector/document database
├── config.py                   # Configuration
├── ollama_client.py            # Ollama API client
│
├── hnsw.py                     # HNSW index
├── kdtree.py                   # KD-Tree index
├── bruteforce.py               # Exact vector search
├── bm25.py                     # BM25 lexical search
├── distances.py                # Distance functions
├── storage.py                  # SQLite persistence
├── text_utils.py               # Text processing helpers
│
├── demo_data.py                # Demo vectors and data
│
├── evaluation.py               # Retrieval evaluation
├── eval_queries.json           # Evaluation queries
├── build_eval_queries.py       # Build evaluation dataset
├── populate_eval_data.py      # Populate evaluation documents
├── reembed_documents.py       # Re-embed documents
├── benchmark_hnsw.py           # HNSW benchmarking
│
├── main.py                     # Main/utility entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
└── data/
    └── .gitkeep                # Database directory placeholder
```

The runtime SQLite database is intentionally not committed to GitHub.

------------------------------------------------------------------------

## 5. The Journey of a Query

Let's trace a query through NeuroIndex.

Suppose the user enters:

``` text
binary tree
```

### Step 1: User Enters a Query

The browser sends the query to the server.

``` text
"binary tree"
```

The UI can also visualize the query as a star inside the 2D semantic
space.

### Step 2: Create the Query Embedding

The server sends the text to Ollama:

``` text
binary tree
     │
     ▼
nomic-embed-text
     │
     ▼
768-dimensional vector
```

### Step 3: Select Search Algorithm

The user can choose:

``` text
HNSW
KD-Tree
Brute Force
```

The selected index searches the stored vectors.

### Step 4: Calculate Similarity

The query vector is compared against stored vectors.

Conceptually:

``` text
Query Vector
     │
     ├── Binary Search Tree      → high similarity
     ├── KD-Tree                 → high similarity
     ├── Linked List             → medium similarity
     ├── Pizza                   → low similarity
     └── Football                → low similarity
```

### Step 5: Return Top-K Results

If:

``` text
Top-K = 5
```

the system returns the five highest-ranked vectors.

### Step 6: Visualize the Results

The browser projects the vectors into two dimensions using PCA.

``` text
                 Semantic Space

       ●
                    ●

             ⭐ Query

       ●                    ●
```

The query is displayed as a star and retrieved results are highlighted.

------------------------------------------------------------------------

## 6. Deep Dive: Each Component

### vectordb.py

This is the central database layer.

It coordinates:

-   Vector storage
-   Vector retrieval
-   Search indexes
-   Document storage
-   Metadata
-   Similarity search

The vector database exposes operations for inserting vectors, counting
vectors, listing items, and retrieving nearest neighbors.

The document database manages:

``` text
document_id
title
text
embedding
metadata
```

------------------------------------------------------------------------

### distances.py

Contains distance/similarity functions.

The main metric used by the current system is:

``` text
Cosine
```

The distance layer allows the search algorithms to work with a
consistent metric interface.

------------------------------------------------------------------------

### bruteforce.py

Brute-force search compares the query against every stored vector.

``` text
Query
  │
  ├── Compare with Vector 1
  ├── Compare with Vector 2
  ├── Compare with Vector 3
  ├── ...
  └── Compare with Vector N
          │
          ▼
       Sort scores
          │
          ▼
       Top-K
```

### Advantages

-   Exact
-   Simple
-   Easy to understand
-   Good baseline for evaluation

### Disadvantage

Search cost grows with the number of vectors.

------------------------------------------------------------------------

### kdtree.py

A KD-Tree recursively partitions points across dimensions.

Conceptually:

``` text
                 Root
                  │
          ┌───────┴───────┐
          ▼               ▼
        Left             Right
         │                 │
     ┌───┴───┐         ┌───┴───┐
     ▼       ▼         ▼       ▼
   Point   Point     Point   Point
```

The purpose is to avoid comparing the query against every point when
possible.

KD-Trees are particularly useful for lower-dimensional numeric spaces.
High-dimensional embeddings are generally more challenging because of
the curse of dimensionality, which is one reason NeuroIndex also
implements HNSW.

------------------------------------------------------------------------

## 7. How HNSW Search Works

**HNSW** stands for:

**Hierarchical Navigable Small World**

Instead of comparing a query against every vector, HNSW creates a graph
connecting nearby vectors.

Conceptually:

``` text
Layer 2:

        A -------- D
         \        /
          \      /
            C

Layer 1:

 A ---- B ---- C ---- D ---- E
       \      /      \
        F ----        G
```

A query starts from an entry point and navigates through increasingly
better neighbors.

### HNSW Parameters

NeuroIndex exposes configuration values such as:

``` text
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 50
```

### M

Controls the approximate number of graph connections maintained by each
node.

Higher values can improve recall but increase memory and construction
cost.

### efConstruction

Controls how much work is performed while building the graph.

Higher values generally improve graph quality at the cost of
construction time.

### efSearch

Controls the search effort.

Higher values generally improve recall while increasing query latency.

### Why HNSW?

The goal is to make nearest-neighbor search efficient as the number of
vectors grows.

------------------------------------------------------------------------

## 8. How KD-Tree Search Works

A KD-Tree recursively splits points according to dimensions.

For example:

``` text
Dimension 0
    │
    ├── values < split
    │
    └── values >= split
```

The next level can split using another dimension.

``` text
              Dimension 0
                  │
             ┌────┴────┐
             ▼         ▼
          Dimension 1  Dimension 1
             │            │
          ┌──┴──┐      ┌──┴──┐
          ▼     ▼      ▼     ▼
```

The implementation provides another retrieval strategy that can be
compared against brute force and HNSW.

------------------------------------------------------------------------

## 9. How BM25 and Hybrid Search Work

### BM25

BM25 is a lexical ranking algorithm.

It considers factors such as:

-   Term frequency
-   Inverse document frequency
-   Document length

For example:

``` text
Query:
"dynamic programming"

Document A:
"Dynamic programming uses overlapping subproblems."

Document B:
"Football is played by two teams."

BM25
   ↓
Document A receives a much higher lexical score.
```

BM25 is useful when exact terminology matters.

------------------------------------------------------------------------

### Hybrid Search

Hybrid search combines:

``` text
Semantic Vector Search
          +
Lexical BM25 Search
```

This gives the system both:

-   semantic understanding
-   exact keyword matching

Conceptually:

``` text
                  Query
                    │
             ┌──────┴──────┐
             ▼             ▼
        Vector Search    BM25
             │             │
             ▼             ▼
          Ranking        Ranking
             │             │
             └──────┬──────┘
                    ▼
          Reciprocal Rank Fusion
                    │
                    ▼
               Final Ranking
```

### Reciprocal Rank Fusion

RRF combines rankings from different retrieval systems.

A document that appears near the top of multiple rankings receives a
strong combined score.

This makes hybrid retrieval more robust than relying on only one
retrieval method.

------------------------------------------------------------------------

## 10. How RAG and Ask AI Work

The **Ask AI** section turns NeuroIndex into a RAG system.

RAG means:

**Retrieval-Augmented Generation**

Instead of asking the LLM to answer using only its internal knowledge,
NeuroIndex first retrieves relevant documents.

### RAG Pipeline

``` text
User Question
      │
      ▼
Query Embedding
      │
      ▼
Document Retrieval
      │
      ├── Vector retrieval
      ├── BM25 retrieval
      └── Hybrid ranking
      │
      ▼
Top-K Context
      │
      ▼
Prompt Construction
      │
      ▼
Llama 3.2
      │
      ▼
Generated Answer
```

### Example

Question:

``` text
What is dynamic programming?
```

NeuroIndex retrieves relevant documents such as:

``` text
Dynamic Programming
Binary Search Tree
Algorithms
```

The retrieved text is passed to the LLM as context.

The final response is therefore grounded in the retrieved documents.

### Retrieved Context

The UI displays the retrieved context as expandable chips so the user
can see which documents contributed to the answer.

------------------------------------------------------------------------

## 11. Interactive Vector Visualization

The web interface provides a 2D visualization of the vector space.

The actual embeddings have:

``` text
768 dimensions
```

A browser-side projection reduces them to:

``` text
2 dimensions
```

using PCA.

``` text
768D Vector
     │
     ▼
    PCA
     │
     ▼
 2D Point
```

### Visualization

The interface shows:

-   Document/vector points
-   Categories
-   Search results
-   Query star
-   Connections from the query to retrieved results
-   Tooltips with vector/document information

Conceptually:

``` text
                 2D PCA Semantic Space

          ●
                     ●

                ⭐ Query
             ╱    │    ╲
            ╱     │     ╲

       ●          ●          ●
```

The query star represents the query embedding projected into the same 2D
space as the visible vectors.

------------------------------------------------------------------------

## 12. Evaluation and Benchmarking

NeuroIndex includes a retrieval evaluation pipeline.

The evaluation dataset contains multiple queries covering areas such as:

``` text
Computer Science
Mathematics
Machine Learning
Search
Food
Sports
```

The system compares:

``` text
Vector Search
BM25
Hybrid Search
```

using metrics including:

### Precision@K

Measures how many of the returned top-K documents are relevant.

``` text
Precision@K =
Relevant retrieved documents
─────────────────────────────
          K
```

### Recall@K

Measures how many relevant documents were successfully retrieved.

``` text
Recall@K =
Relevant retrieved documents
─────────────────────────────
      Total relevant documents
```

### MRR

**Mean Reciprocal Rank** measures how highly the first relevant result
appears.

``` text
MRR = average(1 / rank_of_first_relevant_result)
```

### Latency

The evaluation also records average retrieval latency.

### Current Evaluation Result

The current evaluation run achieved:

``` text
Vector Search
Recall@5: 100%
MRR:      0.935

BM25
Recall@5: 95.33%
MRR:      0.9039

Hybrid
Recall@5: 100%
MRR:      0.9366
```

The hybrid approach slightly improved MRR over vector search while
maintaining full Recall@5 in this evaluation run.

------------------------------------------------------------------------

## 13. Persistence and Storage

NeuroIndex uses SQLite for local persistence.

The database is stored under:

``` text
data/neuroindex.db
```

The repository intentionally ignores runtime database files.

``` text
data/
├── .gitkeep
└── neuroindex.db       # created locally at runtime
```

This keeps the GitHub repository lightweight and avoids committing
machine-specific runtime state.

### Why SQLite?

SQLite provides:

-   Local persistence
-   No separate database server
-   Simple deployment
-   Easy development
-   Reliable structured storage

------------------------------------------------------------------------

## 14. Building and Running

### Prerequisites

-   macOS or Linux
-   Python 3
-   Ollama
-   `nomic-embed-text`
-   `llama3.2`

### Create Virtual Environment

``` bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

``` bash
pip install -r requirements.txt
```

### Install Ollama Models

``` bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Make sure Ollama is running.

### Populate Documents

``` bash
python populate_eval_data.py
```

### Start NeuroIndex

``` bash
python server.py
```

The server runs at:

``` text
http://localhost:8080
```

Open that address in your browser.

### Verify the Server

``` bash
curl http://localhost:8080/stats
```

Expected current demo configuration:

``` json
{
  "vectors": 20,
  "documents": 31,
  "dimension": 768,
  "metric": "cosine",
  "embeddingModel": "nomic-embed-text"
}
```

------------------------------------------------------------------------

## 15. Understanding the Output

When the server starts, NeuroIndex displays information similar to:

``` text
============================================================
Loading NeuroIndex demo vectors
============================================================
Inserted 20/20 demo vectors
Vector count: 20
============================================================

============================================================
NeuroIndex server
============================================================
Server:     http://localhost:8080
Database:   .../data/neuroindex.db
Embedding:  nomic-embed-text
Dimension:  768
Documents:  31
============================================================
```

### Web Interface

The interface is divided into three main areas:

``` text
┌──────────────┬──────────────────────────┬─────────────────┐
│              │                          │ SEARCH          │
│   Controls   │                          │ DOCUMENTS       │
│              │     Vector Space         │ ASK AI          │
│              │       PCA View           │                 │
│              │                          │                 │
│              │         ⭐ Query          │                 |
└──────────────┴──────────────────────────┴─────────────────┘
```

### Left Panel

Contains controls for:

-   Query input
-   Search
-   Algorithm selection
-   Distance metric
-   Top-K
-   Category filtering
-   Demo vector insertion
-   Benchmarking

### Center Panel

Displays:

-   2D PCA semantic space
-   Vector points
-   Query star
-   Search connections
-   Tooltips

### Right Panel

Contains:

-   Search
-   Documents
-   Ask AI

The Ask AI section displays:

-   User questions
-   Generated answers
-   Retrieved context
-   Source documents

------------------------------------------------------------------------

## 16. Extending the Project

### Add Another Distance Metric

The distance layer can be extended with additional metrics such as:

``` text
Euclidean
Manhattan
Dot Product
```

The new metric can then be exposed through the existing
configuration/search pipeline.

### Improve HNSW

Possible improvements:

-   More advanced level generation
-   Better neighbor selection
-   Improved deletion support
-   Persistent HNSW graph storage
-   Parallel index construction

### Improve Hybrid Retrieval

Possible improvements:

-   Tunable vector/BM25 weights
-   Cross-encoder reranking
-   More advanced RRF weighting
-   Query expansion
-   Metadata-aware filtering

### Scale the Vector Store

Possible future improvements:

-   Memory-mapped vectors
-   Quantization
-   Batch insertion
-   Parallel indexing
-   Sharded indexes
-   Persistent ANN graphs

### Improve RAG

Possible additions:

-   Conversation memory
-   Source citations
-   Reranking
-   Chunk-level retrieval
-   Streaming generation
-   More sophisticated prompt construction

------------------------------------------------------------------------

## Summary

NeuroIndex demonstrates how a modern vector retrieval and RAG system can
be constructed from the ground up.

The project combines:

1.  **Vector Embeddings** - converting text into 768-dimensional
    representations
2.  **Similarity Search** - finding semantically related vectors
3.  **HNSW** - approximate nearest-neighbor graph search
4.  **KD-Tree** - tree-based spatial search
5.  **Brute Force** - exact nearest-neighbor baseline
6.  **BM25** - lexical retrieval
7.  **Hybrid Search** - combining semantic and lexical retrieval
8.  **RRF** - merging ranked result lists
9.  **SQLite** - persistent local storage
10. **RAG** - retrieving context before LLM generation
11. **Ollama** - local embeddings and LLM inference
12. **PCA Visualization** - visualizing the semantic vector space
13. **Evaluation** - measuring precision, recall, MRR, and latency

The key idea behind NeuroIndex is simple:

``` text
Raw Text
   │
   ▼
Embeddings
   │
   ▼
Vector Database
   │
   ├──────────────┐
   ▼              ▼
Vector Search    BM25
   │              │
   └──────┬───────┘
          ▼
       Hybrid
       Retrieval
          │
          ▼
      Relevant Docs
          │
          ▼
        RAG
          │
          ▼
       Llama 3.2
          │
          ▼
      Final Answer
```

NeuroIndex is designed to make the internals of vector databases and RAG
systems understandable rather than hiding them behind a managed service.

------------------------------------------------------------------------

## Questions?

The code is organized so that each major retrieval concept can be
studied independently.

A good learning path is:

``` text
distances.py
      ↓
bruteforce.py
      ↓
kdtree.py
      ↓
hnsw.py
      ↓
bm25.py
      ↓
vectordb.py
      ↓
server.py
      ↓
index.html
      ↓
RAG / Evaluation
```

Happy building! 🚀
