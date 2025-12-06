"""
Embeddings store module using SentenceTransformers and ChromaDB.

Provides functions to index match transcripts and retrieve context for Q&A.
"""

import hashlib
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# =============================================================================
# Configuration
# =============================================================================

# Embedding model - lightweight and fast
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ChromaDB collection name
COLLECTION_NAME = "match_chunks"

# Chunking parameters
CHUNK_SIZE = 10  # Number of events per chunk
CHUNK_OVERLAP = 2  # Overlap between chunks for context continuity

# =============================================================================
# Global instances (lazy initialization)
# =============================================================================

_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.Client] = None
_collection: Optional[chromadb.Collection] = None


def _get_embedding_model() -> SentenceTransformer:
    """
    Get or initialize the SentenceTransformer embedding model.
    
    Returns:
        SentenceTransformer: The embedding model instance.
    """
    global _embedding_model
    if _embedding_model is None:
        print(f"[Embeddings] Loading model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_collection() -> chromadb.Collection:
    """
    Get or initialize the ChromaDB collection.
    
    Returns:
        chromadb.Collection: The collection for storing match chunks.
    """
    global _chroma_client, _collection
    
    if _collection is None:
        # Use persistent storage in a local directory
        _chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
        ))
        
        # Get or create the collection
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Match transcript chunks for RAG"}
        )
        print(f"[Embeddings] ChromaDB collection '{COLLECTION_NAME}' ready.")
    
    return _collection


# =============================================================================
# Indexing Functions
# =============================================================================

def index_match_transcript(match_id: str, events: list[dict]) -> int:
    """
    Index a match transcript into the vector store.
    
    Chunks the events into blocks, embeds them, and stores in ChromaDB.
    If the match is already indexed, it will be re-indexed (old chunks deleted).
    
    Args:
        match_id: Unique identifier for the match (will be converted to string).
        events: List of events as [{"minute": int, "text": str}, ...].
        
    Returns:
        int: Number of chunks indexed.
    """
    match_id = str(match_id)
    
    if not events:
        print(f"[Embeddings] No events to index for match {match_id}.")
        return 0
    
    collection = _get_collection()
    model = _get_embedding_model()
    
    # Delete existing chunks for this match (to allow re-indexing)
    _delete_match_chunks(match_id)
    
    # Create chunks from events
    chunks = _create_chunks(events)
    
    if not chunks:
        print(f"[Embeddings] No chunks created for match {match_id}.")
        return 0
    
    # Prepare data for ChromaDB
    ids = []
    documents = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        # Create a unique ID for this chunk
        chunk_id = _generate_chunk_id(match_id, i)
        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append({
            "match_id": match_id,
            "start_minute": chunk["start_minute"],
            "end_minute": chunk["end_minute"],
            "chunk_index": i,
        })
    
    # Generate embeddings
    print(f"[Embeddings] Generating embeddings for {len(documents)} chunks...")
    embeddings = model.encode(documents, show_progress_bar=False).tolist()
    
    # Upsert into ChromaDB
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    
    print(f"[Embeddings] Indexed {len(chunks)} chunks for match {match_id}.")
    return len(chunks)


def _create_chunks(events: list[dict]) -> list[dict]:
    """
    Create overlapping chunks from a list of events.
    
    Args:
        events: List of events as [{"minute": int, "text": str}, ...].
        
    Returns:
        List of chunks as:
            [{
                "text": str,  # Joined event lines with minute markers
                "start_minute": int,
                "end_minute": int,
            }, ...]
    """
    chunks = []
    
    # Sort events by minute
    sorted_events = sorted(events, key=lambda x: x.get("minute", 0))
    
    i = 0
    while i < len(sorted_events):
        # Get chunk_size events starting from i
        chunk_events = sorted_events[i:i + CHUNK_SIZE]
        
        if not chunk_events:
            break
        
        # Build chunk text with minute markers
        lines = []
        for event in chunk_events:
            minute = event.get("minute", 0)
            text = event.get("text", "")
            lines.append(f"[{minute}'] {text}")
        
        chunk_text = "\n".join(lines)
        
        # Get minute range
        start_minute = chunk_events[0].get("minute", 0)
        end_minute = chunk_events[-1].get("minute", 0)
        
        chunks.append({
            "text": chunk_text,
            "start_minute": start_minute,
            "end_minute": end_minute,
        })
        
        # Move forward by (chunk_size - overlap) to create overlapping chunks
        i += max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    
    return chunks


def _generate_chunk_id(match_id: str, chunk_index: int) -> str:
    """
    Generate a unique ID for a chunk.
    
    Args:
        match_id: The match identifier.
        chunk_index: Index of the chunk within the match.
        
    Returns:
        str: A unique chunk ID.
    """
    raw = f"{match_id}_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _delete_match_chunks(match_id: str) -> None:
    """
    Delete all existing chunks for a match.
    
    Args:
        match_id: The match identifier.
    """
    collection = _get_collection()
    
    try:
        # Query to find existing chunks for this match
        results = collection.get(
            where={"match_id": match_id},
        )
        
        if results["ids"]:
            collection.delete(ids=results["ids"])
            print(f"[Embeddings] Deleted {len(results['ids'])} existing chunks for match {match_id}.")
    except Exception as e:
        # Collection might be empty or match not found - that's OK
        print(f"[Embeddings] Note: {e}")


# =============================================================================
# Retrieval Functions
# =============================================================================

def retrieve_match_context(match_id: str, question: str, k: int = 6) -> list[dict]:
    """
    Retrieve relevant context chunks for a question about a specific match.
    
    Args:
        match_id: The match identifier.
        question: The user's question.
        k: Number of chunks to retrieve (default: 6).
        
    Returns:
        List of relevant chunks as:
            [{
                "text": str,
                "start_minute": int,
                "end_minute": int,
            }, ...]
        Sorted by minute (earliest first).
    """
    match_id = str(match_id)
    collection = _get_collection()
    model = _get_embedding_model()
    
    # Embed the question
    question_embedding = model.encode(question, show_progress_bar=False).tolist()
    
    # Query ChromaDB for this match only
    try:
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=k,
            where={"match_id": match_id},
            include=["documents", "metadatas"],
        )
    except Exception as e:
        print(f"[Embeddings] Error querying: {e}")
        return []
    
    # Extract and format results
    chunks = []
    
    if results["documents"] and results["documents"][0]:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
        
        for doc, meta in zip(documents, metadatas):
            chunks.append({
                "text": doc,
                "start_minute": meta.get("start_minute", 0),
                "end_minute": meta.get("end_minute", 0),
            })
    
    # Sort by start_minute for chronological order
    chunks.sort(key=lambda x: x["start_minute"])
    
    return chunks


def get_match_chunk_count(match_id: str) -> int:
    """
    Get the number of indexed chunks for a match.
    
    Args:
        match_id: The match identifier.
        
    Returns:
        int: Number of chunks indexed for this match.
    """
    match_id = str(match_id)
    collection = _get_collection()
    
    try:
        results = collection.get(
            where={"match_id": match_id},
        )
        return len(results["ids"]) if results["ids"] else 0
    except Exception:
        return 0
