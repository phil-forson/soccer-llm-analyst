# Soccer LLM Analyst API Documentation

REST API backend for the Soccer LLM Analyst system. This API provides intelligent football/soccer information using RAG (Retrieval-Augmented Generation).

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Server

```bash
# Development mode (with auto-reload)
python -m src.api

# Or using uvicorn directly
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### 4. View API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### `GET /health`
Health check endpoint (same as `/`).

### `POST /query/stream`
**Streaming query endpoint** - shows thinking process at each stage (like ChatGPT).

This endpoint uses Server-Sent Events (SSE) to stream real-time updates showing what's happening at each stage of the RAG pipeline.

**Request Body:**
```json
{
  "query": "What was the score of Liverpool vs Sunderland?",
  "include_highlights": true
}
```

**Response:** Server-Sent Events stream with thinking messages:

```
data: {"stage": "query_parser", "message": "Analyzing query intent...", "status": "starting", "data": {}}

data: {"stage": "query_parser", "message": "Query parsed: Intent = match_result", "status": "complete", "data": {"intent": "match_result"}}

data: {"stage": "web_search", "message": "Searching the web for match information...", "status": "starting", "data": {}}

data: {"stage": "web_search", "message": "Executing web search queries...", "status": "processing", "data": {}}

data: {"stage": "web_search", "message": "Found match information: Liverpool vs Sunderland", "status": "complete", "data": {"has_match": true}}

data: {"stage": "game_analyst", "message": "Analyzing match momentum and tactics...", "status": "starting", "data": {}}

data: {"stage": "highlights", "message": "Searching for match highlight videos...", "status": "starting", "data": {}}

data: {"type": "result", "data": {"success": true, "intent": "match_result", ...}}

data: [DONE]
```

**JavaScript Example:**
```javascript
const eventSource = new EventSource('http://localhost:8000/query/stream', {
  method: 'POST',
  body: JSON.stringify({ query: "Liverpool vs Sunderland" }),
  headers: { 'Content-Type': 'application/json' }
});

// Note: EventSource only supports GET, so use fetch with ReadableStream instead:
async function streamQuery(query) {
  const response = await fetch('http://localhost:8000/query/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        if (data === '[DONE]') {
          console.log('Stream complete');
          break;
        }
        
        const parsed = JSON.parse(data);
        if (parsed.type === 'result') {
          console.log('Final result:', parsed.data);
        } else {
          console.log('Thinking:', parsed.message, 'Stage:', parsed.stage);
        }
      }
    }
  }
}
```

### `POST /query`
Main query endpoint - processes natural language football queries.

**Request Body:**
```json
{
  "query": "What was the score of Liverpool vs Sunderland?",
  "include_highlights": true  // Optional: auto-detected if not provided
}
```

**Response:**
```json
{
  "success": true,
  "intent": "match_result",
  "summary": "The final score of the Liverpool vs Sunderland match...",
  "match_metadata": {
    "home_team": "Liverpool",
    "away_team": "Sunderland",
    "match_date": "2025-12-03",
    "score": "1-1",
    "competition": "Premier League",
    "key_moments": [
      {
        "minute": "45",
        "event": "GOAL",
        "description": "Wirtz scores for Liverpool",
        "team": "home"
      }
    ],
    "man_of_the_match": "Florian Wirtz",
    "match_summary": "Liverpool dropped points at home..."
  },
  "highlights": [
    {
      "title": "Liverpool vs Sunderland 1-1 Highlights",
      "url": "https://www.youtube.com/watch?v=...",
      "duration": "8:16",
      "source_type": "NBC Sports",
      "is_nbc_sports": true,
      "is_official_club": false,
      "confidence": 0.9
    }
  ],
  "sources": [
    "https://www.espn.com/...",
    "https://www.bbc.com/sport/..."
  ],
  "error": null
}
```

### `GET /intents`
List all available query intents.

**Response:**
```json
{
  "intents": [
    {
      "value": "match_result",
      "description": "Get match score/result"
    },
    {
      "value": "match_highlights",
      "description": "Watch match highlights"
    },
    // ... more intents
  ]
}
```

### `POST /analyze`
Comprehensive game analysis endpoint - provides sophisticated match analysis.

**Request Body:**
```json
{
  "query": "Liverpool vs Sunderland",
  "include_highlights": true  // Optional
}
```

**Response:**
```json
{
  "success": true,
  "match_info": {
    "home_team": "Liverpool",
    "away_team": "Sunderland",
    "score": "1-1",
    "match_date": "2025-12-03",
    "competition": "Premier League",
    "man_of_the_match": "Florian Wirtz",
    "match_summary": "Brief summary..."
  },
  "deep_analysis": "Comprehensive match analysis covering narrative, momentum, tactics, implications...",
  "momentum_analysis": [
    {
      "minute": "45",
      "event": "GOAL",
      "description": "Wirtz scores for Liverpool",
      "team": "home",
      "momentum_impact": "high",
      "reasoning": "Early goal sets the tone and puts pressure on the opponent"
    }
  ],
  "tactical_analysis": {
    "match_phases": {
      "first_half_events": 3,
      "second_half_events": 2,
      "more_active_half": "first"
    },
    "goal_distribution": {
      "home_goals": 1,
      "away_goals": 1,
      "total_goals": 2
    },
    "team_activity": {
      "home_events": 4,
      "away_events": 3,
      "more_active_team": "Liverpool"
    }
  },
  "key_moments": [...],
  "highlights": [...],
  "error": null
}
```

**Analysis Includes:**
- **Match Narrative**: How the match unfolded
- **Momentum Analysis**: When and why momentum shifted
- **Tactical Breakdown**: Formation, style, key decisions
- **Key Performances**: Standout players and impact
- **Implications**: What the result means for both teams
- **Statistical Insights**: Patterns and trends

## Frontend Integration Examples

### JavaScript/React

```javascript
// Query endpoint
async function querySoccer(query, includeHighlights = null) {
  const response = await fetch('http://localhost:8000/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: query,
      include_highlights: includeHighlights
    })
  });
  
  return await response.json();
}

// Usage
const result = await querySoccer("What was the score of Liverpool vs Sunderland?");
console.log(result.summary);
console.log(result.match_metadata);
console.log(result.highlights);
```

### Python

```python
import requests

def query_soccer(query, include_highlights=None):
    response = requests.post(
        'http://localhost:8000/query',
        json={
            'query': query,
            'include_highlights': include_highlights
        }
    )
    return response.json()

# Usage
result = query_soccer("What was the score of Liverpool vs Sunderland?")
print(result['summary'])
print(result['match_metadata'])
print(result['highlights'])
```

### cURL

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was the score of Liverpool vs Sunderland?",
    "include_highlights": true
  }'
```

## Query Examples

### Match Results
```json
{
  "query": "What was the score of Arsenal vs Chelsea?"
}
```

### Match Highlights
```json
{
  "query": "Show me highlights of Manchester City vs Liverpool",
  "include_highlights": true
}
```

### Team News
```json
{
  "query": "Latest news about Real Madrid"
}
```

### Transfer News
```json
{
  "query": "Transfer news for Haaland"
}
```

### Standings
```json
{
  "query": "Premier League table"
}
```

### Player Info
```json
{
  "query": "How is Messi performing this season?"
}
```

## CORS Configuration

The API is configured to allow CORS from all origins by default. For production, update the CORS settings in `src/api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],  # Specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Error Handling

The API returns structured error responses:

```json
{
  "success": false,
  "intent": "general",
  "summary": "",
  "error": "Error message here"
}
```

## Production Deployment

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Gunicorn

```bash
gunicorn src.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Rate Limiting

Consider adding rate limiting for production. Example with `slowapi`:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/query")
@limiter.limit("10/minute")
async def query_endpoint(request: Request, query: QueryRequest):
    # ... existing code
```

## Environment Variables

- `OPENAI_API_KEY`: Required - Your OpenAI API key for LLM functionality

## Notes

- The API uses RAG (Retrieval-Augmented Generation) for intelligent search
- Highlights are automatically validated against web search results
- Competition-aware source selection (CBS Golazo for UCL, NBC Sports for PL, etc.)
- Home/away team order is validated in video results

