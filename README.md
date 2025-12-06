# ⚽ Soccer Match LLM Analyst (CLI)

A terminal-based application that helps you find football match highlights and information using AI-powered search.

## Features

- 🔍 **Smart Match Search** - Find matches by natural language query (e.g., "Chelsea vs Barcelona")
- 🎬 **YouTube Highlights** - Automatically finds real match highlights (filters out simulations!)
- 🌐 **Web Search** - Searches the web for match information with LLM summarization
- ⏰ **Recency Aware** - Prioritizes recent matches when no date specified
- 🏠 **Home/Away Smart** - Understands team order (first team = home)

## Tech Stack

- **Web Search**: `ddgs` (DuckDuckGo - no API key needed)
- **YouTube Search**: DuckDuckGo video search + validation
- **LLM**: OpenAI API (gpt-4.1-mini by default)

## Setup

### 1. Clone and Install Dependencies

```bash
cd soccer-llm-analyst
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI API Key
OPENAI_API_KEY=your_openai_key_here
```

## Usage

### Run the CLI

```bash
python -m src.cli
```

### Example Session

```
╔══════════════════════════════════════════════════════════════╗
║           ⚽ Soccer Match LLM Analyst (CLI) ⚽                ║
║                                                              ║
║  Find match info and highlights using AI-powered search      ║
╚══════════════════════════════════════════════════════════════╝

Enter a match description to find highlights and info.
Examples:
  • 'Chelsea vs Barcelona'  (finds most recent match)
  • 'Arsenal vs Chelsea 2024-12-01'  (specific date)
  • 'Real Madrid vs Man City Champions League'

Tip: First team = home team, second team = away team

🎮 Describe the match (or 'q' to quit): Chelsea vs Barcelona

🔍 Searching for: Chelsea vs Barcelona

🌐 Searching for match information...

🔍 Match Information:
Chelsea defeated Barcelona 3-0 in the UEFA Champions League...

⏳ Finding match highlights...

[YouTubeSearch] Looking for: Chelsea (home) vs Barcelona (away), season: 2024-25
[YouTubeSearch] Filtered out 8 videos >1 hour
[YouTubeSearch] Filtered out 12 simulation/game videos

🎬 Match Highlights:

  1. Chelsea vs Barcelona 3-0 | HIGHLIGHTS | Champions League 2024-25
     ⏱️  12:34
     📺 TNT Sports ⭐ Official
     🔗 https://youtube.com/watch?v=...

  2. Chelsea vs Barcelona | Extended Highlights | UCL
     ⏱️  8:45
     📺 Chelsea FC ⭐ Official
     🔗 https://youtube.com/watch?v=...
```

## Highlight Filtering

The system automatically filters out:

- ❌ **Simulations** - FIFA, EA FC, eFootball, PES gameplay
- ❌ **Long videos** - Over 1 hour (likely full matches or simulations)
- ❌ **Old content** - "REWIND", "throwback", "classic" videos
- ❌ **Non-highlights** - Previews, reactions, press conferences

And prioritizes:

- ⭐ **Official channels** - Club channels, Champions League, TNT Sports
- ✅ **Trusted broadcasters** - NBC Sports, Sky Sports, ESPN
- 🕐 **Recent content** - Current year/season
- ⏱️ **Ideal duration** - 5-20 minutes

## Project Structure

```
soccer-llm-analyst/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration and env var helpers
│   ├── web_search_agent.py    # Web search agent (DuckDuckGo + LLM)
│   ├── youtube_search_agent.py # YouTube highlight finder
│   ├── embeddings_store.py    # Vector store (ChromaDB)
│   ├── qa.py                  # Q&A logic (RAG + LLM)
│   └── cli.py                 # Command-line interface
├── tests/
│   ├── __init__.py
│   └── test_chunking.py       # Unit tests
├── .env                       # Your API keys (not in git)
├── requirements.txt
└── README.md
```

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## License

MIT
