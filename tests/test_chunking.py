"""
Tests for the embeddings store chunking logic.

Run with: python -m pytest tests/test_chunking.py -v
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.embeddings_store import _create_chunks, CHUNK_SIZE, CHUNK_OVERLAP


class TestChunking:
    """Tests for the _create_chunks function."""
    
    def test_empty_events(self):
        """Test that empty events list returns empty chunks."""
        events = []
        chunks = _create_chunks(events)
        assert chunks == []
    
    def test_single_event(self):
        """Test chunking with a single event."""
        events = [{"minute": 10, "text": "Goal scored!"}]
        chunks = _create_chunks(events)
        
        assert len(chunks) == 1
        assert chunks[0]["start_minute"] == 10
        assert chunks[0]["end_minute"] == 10
        assert "[10']" in chunks[0]["text"]
        assert "Goal scored!" in chunks[0]["text"]
    
    def test_few_events(self):
        """Test chunking with fewer events than CHUNK_SIZE."""
        events = [
            {"minute": 5, "text": "Kickoff"},
            {"minute": 10, "text": "Shot wide"},
            {"minute": 15, "text": "Corner kick"},
        ]
        chunks = _create_chunks(events)
        
        assert len(chunks) == 1
        assert chunks[0]["start_minute"] == 5
        assert chunks[0]["end_minute"] == 15
    
    def test_many_events_creates_overlapping_chunks(self):
        """Test that many events create overlapping chunks."""
        # Create 25 events
        events = [{"minute": i * 4, "text": f"Event at minute {i * 4}"} for i in range(25)]
        chunks = _create_chunks(events)
        
        # With CHUNK_SIZE=10 and CHUNK_OVERLAP=2, we should have multiple chunks
        assert len(chunks) > 1
        
        # Check that chunks are sorted by start_minute
        for i in range(len(chunks) - 1):
            assert chunks[i]["start_minute"] <= chunks[i + 1]["start_minute"]
    
    def test_chunk_text_format(self):
        """Test that chunk text is formatted correctly with minute markers."""
        events = [
            {"minute": 45, "text": "GOAL! Player scores!"},
            {"minute": 46, "text": "Celebration"},
        ]
        chunks = _create_chunks(events)
        
        assert len(chunks) == 1
        text = chunks[0]["text"]
        
        # Check minute markers are present
        assert "[45']" in text
        assert "[46']" in text
        assert "GOAL! Player scores!" in text
        assert "Celebration" in text
    
    def test_events_sorted_by_minute(self):
        """Test that events are sorted by minute before chunking."""
        # Events in random order
        events = [
            {"minute": 30, "text": "Event C"},
            {"minute": 10, "text": "Event A"},
            {"minute": 20, "text": "Event B"},
        ]
        chunks = _create_chunks(events)
        
        assert len(chunks) == 1
        text = chunks[0]["text"]
        
        # Events should be in chronological order in the chunk
        pos_a = text.find("Event A")
        pos_b = text.find("Event B")
        pos_c = text.find("Event C")
        
        assert pos_a < pos_b < pos_c
    
    def test_chunk_minute_range(self):
        """Test that chunk minute ranges are correct."""
        events = [{"minute": i * 5, "text": f"Event {i}"} for i in range(15)]
        chunks = _create_chunks(events)
        
        # Each chunk should have correct start and end minutes
        for chunk in chunks:
            assert chunk["start_minute"] <= chunk["end_minute"]
            assert chunk["start_minute"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
