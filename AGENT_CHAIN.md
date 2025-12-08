# Agent Chain Architecture

## Overview

The Soccer LLM Analyst uses a **chained agent architecture** where each agent receives results from the previous agent and passes its output to the next:

```
User Query
    ↓
Query Parser Agent (LLM) - Parses query ONCE, extracts intent, teams, date
    ↓
Web Search Agent (RAG) - Uses parsed data, searches web, extracts match metadata
    ↓
Game Analyst Agent (LLM) - Uses web search results, analyzes match
    ↓
Highlights Agent (RAG Validation) - Uses parsed data + web results, validates videos
    ↓
Response
```

**IMPORTANT**: Each agent receives parsed data from the previous agent. The YouTube search agent does NOT parse queries itself - it receives `home_team`, `away_team`, `match_date` from the query parser. This ensures:
- No duplicate parsing (saves API credits)
- Consistent data across agents
- Proper agent chain flow

## Agent Chain Flow

### 1. Query Parser Agent
**File**: `src/query_parser_agent.py`

**Purpose**: Understands user intent and extracts entities

**Input**: Natural language query

**Output**:
- `intent`: Query type (match_result, match_highlights, etc.)
- `search_query`: Optimized search query
- `teams`: Extracted team names
- `summary_focus`: What to focus on in summary

**Example**:
```python
parse_query("What was the score of Liverpool vs Sunderland?")
# Returns: {
#   "intent": "match_result",
#   "search_query": "Liverpool vs Sunderland score result",
#   "teams": ["Liverpool", "Sunderland"],
#   "summary_focus": "match score and key moments"
# }
```

### 2. Web Search Agent (RAG)
**File**: `src/web_search_agent.py`

**Purpose**: Retrieves and summarizes information using RAG

**Input**: 
- Query from Query Parser
- Intent
- Original user query

**Output**:
- `web_summary`: LLM-generated summary from web search results (RAG-validated)
- `match_metadata`: Extracted match information:
  - Teams, score, date, competition
  - Key moments (chronological)
  - Man of the match
  - Match summary

**RAG Components**:
- Uses DuckDuckGo for web search
- Stores chunks in ChromaDB vector store
- Uses SentenceTransformers for embeddings
- LLM summarizes with strict anti-hallucination rules

**Example**:
```python
web_summary, match_metadata = search_with_rag(
    query="Liverpool vs Sunderland score result",
    intent="match_result",
    original_query="What was the score of Liverpool vs Sunderland?"
)
```

### 3. Game Analyst Agent
**File**: `src/game_analyst_agent.py`

**Purpose**: Provides sophisticated match analysis

**Input** (from Web Search Agent):
- `web_summary`: RAG-validated summary
- `match_metadata`: Match information with key moments

**Output**:
- `deep_analysis`: Comprehensive LLM-generated analysis
- `momentum_analysis`: Momentum shift analysis
- `tactical_analysis`: Tactical pattern analysis
- `match_info`: Structured match information

**Analysis Includes**:
1. **Match Narrative**: How the match unfolded
2. **Momentum Analysis**: When and why momentum shifted
3. **Tactical Breakdown**: Formations, key decisions
4. **Key Performances**: Standout players
5. **Implications**: What the result means
6. **Statistical Insights**: Patterns and trends

**Example**:
```python
analysis = analyze_match_from_web_results(
    web_summary=web_summary,
    match_metadata=match_metadata,
    original_query="What was the score of Liverpool vs Sunderland?"
)
```

### 4. Highlights Agent
**File**: `src/youtube_search_agent.py`

**Purpose**: Finds and validates match highlight videos using RAG

**Input** (from Web Search Agent and Game Analyst):
- `home_team`, `away_team`, `match_date`: Match identifiers
- `web_summary`: RAG-validated summary (for video validation)
- `match_metadata`: Match information (for validation)

**Output**:
- `highlights`: List of validated highlight videos with:
  - Title, URL, duration
  - Source type (NBC, CBS Golazo, Official Club, etc.)
  - Confidence score (RAG validation)

**RAG Validation**:
- Uses embeddings to validate video titles against web summary
- Uses LLM to validate video content against match context
- Ensures highlights match the actual match (teams, score, date)

**Example**:
```python
highlights = search_and_display_highlights_with_metadata(
    home_team="Liverpool",
    away_team="Sunderland",
    match_date="2025-12-03",
    web_summary=web_summary,  # From web search agent
    match_metadata=match_metadata  # From web search agent
)
```

## Chain Implementation in API

### `/query` Endpoint

```python
# Step 1: Query Parser
parsed = parse_query(request.query)
intent = parsed.get("intent")
search_query = parsed.get("search_query")

# Step 2: Web Search (RAG)
web_summary, match_metadata = search_with_rag(
    query=search_query,
    intent=intent,
    original_query=request.query
)

# Step 3: Game Analyst (for match results)
if intent in ["match_result", "match_highlights"]:
    game_analysis = analyze_match_from_web_results(
        web_summary=web_summary,
        match_metadata=match_metadata,
        original_query=request.query
    )

# Step 4: Highlights (LAST in chain - uses all previous results)
# Chain: Query Parser → Web Search → Game Analyst → Highlights
if show_highlights:
    highlights = search_and_display_highlights_with_metadata(
        home_team=match_metadata.get("home_team"),
        away_team=match_metadata.get("away_team"),
        match_date=match_metadata.get("match_date"),
        web_summary=web_summary,  # From web search agent (for RAG validation)
        match_metadata=match_metadata  # From web search agent (for validation)
    )
```

### `/analyze` Endpoint

Same chain, but focuses on comprehensive analysis:

```python
# Step 1: Query Parser
parsed = parse_query(request.query)

# Step 2: Web Search (RAG)
web_summary, match_metadata = search_with_rag(...)

# Step 3: Game Analyst (always runs)
analysis = analyze_match_from_web_results(
    web_summary=web_summary,
    match_metadata=match_metadata,
    original_query=request.query
)

# Step 3: Highlights (LAST in chain - uses all previous results)
# Chain: Query Parser → Web Search → Game Analyst → Highlights
if include_highlights:
    highlights = search_and_display_highlights_with_metadata(
        home_team=match_metadata.get("home_team"),
        away_team=match_metadata.get("away_team"),
        match_date=match_metadata.get("match_date"),
        web_summary=web_summary,  # From web search agent
        match_metadata=match_metadata  # From web search agent
    )
```

## Key Benefits of Chaining

1. **No Redundant API Calls**: Game Analyst receives results directly, doesn't call API again
2. **RAG-Validated Data**: All agents use the same RAG-validated information
3. **Efficient**: Each agent does its job once, results flow downstream
4. **Modular**: Each agent can be tested/improved independently
5. **Context Preservation**: Original query and intermediate results preserved

## Data Flow Example

```
User: "What was the score of Liverpool vs Sunderland?"

1. Query Parser:
   → intent: "match_result"
   → search_query: "Liverpool vs Sunderland score result"
   → teams: ["Liverpool", "Sunderland"]

2. Web Search (RAG):
   → Searches web with DuckDuckGo
   → Stores chunks in ChromaDB
   → LLM summarizes with RAG retrieval
   → Extracts match metadata
   → Returns: web_summary, match_metadata

3. Game Analyst:
   → Receives web_summary and match_metadata
   → Analyzes momentum shifts from key_moments
   → Analyzes tactical patterns
   → Generates deep LLM analysis
   → Returns: deep_analysis, momentum_analysis, tactical_analysis

4. Highlights Agent (LAST):
   → Receives web_summary and match_metadata (for RAG validation)
   → Searches YouTube for highlight videos
   → Validates videos using RAG (embeddings + LLM)
   → Returns: validated highlight videos with confidence scores

5. Response:
   → Combines all results
   → Returns structured JSON
```

## RAG Usage in Chain

RAG (Retrieval Augmented Generation) is used in:

1. **Web Search Agent**: 
   - Stores web search results in ChromaDB
   - Retrieves relevant chunks for LLM summarization
   - Ensures LLM only uses retrieved context

2. **YouTube Search Agent**:
   - Validates videos against web_summary using embeddings
   - Uses LLM to validate video content against match context
   - Ensures highlights match the actual match

3. **Game Analyst Agent**:
   - Uses RAG-validated web_summary as context
   - Ensures analysis is based on factual information
   - No hallucination - only uses provided context

## Error Handling

Each agent handles errors gracefully:

- **Query Parser fails**: Falls back to general intent
- **Web Search fails**: Returns error, chain stops
- **Game Analyst fails**: Returns basic response without analysis
- **Highlights fail**: Returns response without highlights

The chain is designed to be resilient - if one agent fails, previous results are still returned.

