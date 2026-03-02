import os
import sys
import logging
from typing import List, Dict, Any
from supabase import create_client, Client
import asyncio

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {str(e)}")
else:
    logger.warning("Supabase URL or Key is missing. Database operations will fail.")

async def store_chunks(filename: str, chunks: List[str], embeddings: List[List[float]]) -> bool:
    """Store document chunks and their embeddings into Supabase."""
    if not supabase:
        logger.error("Supabase client not initialized")
        return False
        
    if len(chunks) != len(embeddings):
        logger.error("Number of chunks does not match number of embeddings")
        return False
        
    if not chunks:
        logger.warning(f"No chunks provided for {filename}")
        return True
        
    try:
        # Prepare data for bulk insert
        data = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            data.append({
                "filename": filename,
                "content": chunk,
                "chunk_index": i,
                "embedding": emb
            })
            
        def insert_sync():
            return supabase.table("documents").insert(data).execute()
            
        await asyncio.to_thread(insert_sync)
        logger.info(f"Successfully stored {len(chunks)} chunks for {filename}.")
        return True
    except Exception as e:
        logger.error(f"Error storing chunks for {filename}: {str(e)}")
        return False

async def similarity_search(query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """Find relevant chunks using pgvector distance function via RPC."""
    if not supabase:
        return []
        
    try:
        def search_sync():
            return supabase.rpc(
                'match_documents',
                {'query_embedding': query_embedding, 'match_threshold': 0.0, 'match_count': limit}
            ).execute()
        
        response = await asyncio.to_thread(search_sync)
        return response.data
    except Exception as e:
        logger.error(f"Error in similarity search: {str(e)}")
        return []

async def get_all_documents() -> List[str]:
    """List unique uploaded documents."""
    if not supabase:
        return []
        
    try:
        def get_docs_sync():
            return supabase.table("documents").select("filename").execute()
            
        response = await asyncio.to_thread(get_docs_sync)
        # Extract unique filenames
        unique_files = list(set(repo["filename"] for repo in response.data if "filename" in repo))
        return unique_files
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
        return []

async def delete_document(filename: str) -> bool:
    """Remove a document and all its chunks."""
    if not supabase:
        return False
        
    try:
        def delete_sync():
            return supabase.table("documents").delete().eq("filename", filename).execute()
            
        await asyncio.to_thread(delete_sync)
        logger.info(f"Successfully deleted document {filename}.")
        return True
    except Exception as e:
        logger.error(f"Error deleting document {filename}: {str(e)}")
        return False
        
SCHEMA_INSTRUCTIONS = """
-- Please run the following SQL commands in your Supabase SQL Editor:
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT,
    content TEXT,
    chunk_index INTEGER,
    embedding VECTOR(1536),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION match_documents (
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
RETURNS TABLE (
  id uuid,
  filename text,
  content text,
  chunk_index int,
  similarity float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    documents.id,
    documents.filename,
    documents.content,
    documents.chunk_index,
    1 - (documents.embedding <=> query_embedding) AS similarity
  FROM documents
  WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
  ORDER BY documents.embedding <=> query_embedding
  LIMIT match_count;
$$;
"""
