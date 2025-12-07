# RAG Usage in Soccer LLM Analyst

## Current RAG Implementation Locations

### 1. **Web Search RAG Pipeline** (`web_search_agent.py`)
**Function:** `search_with_rag()`

**Location:** Lines 1262-1371

**What it does:**
1. **Search** - Performs web search using DuckDuckGo
2. **Chunk** - `_chunk_search_results()` - Converts search results into chunks
3. **Index** - `_index_chunks_for_rag()` - Embeds chunks using SentenceTransformers and stores in ChromaDB
4. **Retrieve** - `_retrieve_relevant_context()` - Uses semantic search (embeddings) to find most relevant chunks
5. **Generate** - LLM generates answer from retrieved context

**RAG Components Used:**
- `_get_embedding_model()` - SentenceTransformer model
- `_get_collection()` - ChromaDB collection
- Embeddings for semantic similarity

---

### 2. **YouTube Video Validation RAG** (`youtube_search_agent.py`)
**Function:** `_validate_videos_with_rag()`

**Location:** Lines 2497-2588

**What it does:**
1. **Embeddings Validation** - `_validate_video_with_embeddings()` - Computes semantic similarity between video metadata and web search summary
2. **LLM Validation** - `_validate_video_against_web_context()` - Uses LLM to verify video matches web search results (teams, date, score)

**RAG Components Used:**
- `_compute_semantic_similarity()` - Uses embeddings to compare video text vs web context
- LLM validation against web search summary

---

## Current Query Flow

```
User Query
    ↓
Query Parser Agent (query_parser_agent.py) - Uses LLM to understand intent
    ↓
Web Search with RAG (web_search_agent.py)
    ├─ Search web
    ├─ Chunk results
    ├─ Index in ChromaDB (embeddings)
    ├─ Retrieve relevant chunks (semantic search)
    └─ Generate summary (LLM) ← RAG USED HERE
    ↓
YouTube Search with RAG Validation (youtube_search_agent.py)
    ├─ Search YouTube (competition-aware sources)
    ├─ Filter by source
    └─ RAG Validation ← RAG USED HERE
        ├─ Embeddings similarity check (semantic search)
        └─ LLM validation against web summary (validates home/away order)
```

---

## RAG Usage Summary

### ✅ **RAG is used in 2 places:**

1. **Web Search Summarization** (`web_search_agent.py::search_with_rag()`)
   - **Chunks** web search results
   - **Indexes** chunks in ChromaDB with embeddings
   - **Retrieves** most relevant chunks using semantic search
   - **Generates** summary using LLM with retrieved context

2. **YouTube Video Validation** (`youtube_search_agent.py::_validate_videos_with_rag()`)
   - **Embeddings**: Computes semantic similarity between video metadata and web summary
   - **LLM Validation**: Verifies video matches web search results (teams, date, score, **home/away order**)

---

## Recent Fixes

✅ **Competition-Aware Source Selection**
- Champions League: Prioritizes CBS Sports Golazo, UEFA, official clubs
- Premier League: Prioritizes NBC Sports, Premier League official, official clubs
- Other competitions: Uses all trusted sources

✅ **Home/Away Order Validation**
- RAG validation now checks that video title matches home/away team order
- Validates that "{home_team} vs {away_team}" means home_team is HOME

✅ **Competition Extraction**
- Web search now extracts competition (Premier League, Champions League, etc.)
- Competition info passed to YouTube search for source prioritization

