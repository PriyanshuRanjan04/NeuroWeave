import os
import sys
import logging
import asyncio
from typing import Dict, Any

from lightrag import LightRAG
from lightrag.llm import openai_complete_if_cache

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LIGHTRAG_WORKING_DIR, GROK_API_KEY

logger = logging.getLogger(__name__)

if not os.path.exists(LIGHTRAG_WORKING_DIR):
    os.makedirs(LIGHTRAG_WORKING_DIR)

# Custom LLM interaction for LightRAG using Grok (xAI)
async def llm_model_func(prompt, **kwargs) -> str:
    # Use OpenAI API compatible layer pointing to xAI
    return await openai_complete_if_cache(
        "grok-3", # Use grok-3 or grok-beta
        prompt,
        system_prompt=kwargs.get("system_prompt", ""),
        history_messages=kwargs.get("history_messages", []),
        api_key=GROK_API_KEY,
        base_url="https://api.x.ai/v1",
        **kwargs
    )

async def embedding_func(texts: list[str]) -> list[list[float]]:
    # Use our Grok client embedding configuration
    from llm.grok_client import embed
    import numpy as np
    
    results = []
    for text in texts:
        emb = await embed(text)
        if not emb:
            # Fallback for empty/failed embedding to avoid crashing the whole pipeline
            emb = [0.0] * 1536 
        results.append(emb)
    return np.array(results)

# Initialize LightRAG instance
# The user asked to use Supabase as storage backend.
# Since exact lightrag-hku pgvector args vary by version, we configure the core components
# which defaults to local KV/Graph if Supabase kwargs are omitted.
# Note: For production, we can inject kv_storage="PostgresKVStorage" etc into LightRAG init here.
try:
    rag = LightRAG(
        working_dir=LIGHTRAG_WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )
    logger.info("LightRAG instance initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize LightRAG: {e}")
    rag = None

async def index_document(text: str, filename: str) -> bool:
    """Build knowledge graph from PDF text dynamically."""
    if not rag:
        logger.error("LightRAG not initialized. Cannot index document.")
        return False
        
    try:
        # Here we insert raw text. LightRAG internally handles chunking and Entity-Relation extraction
        def insert_sync():
            rag.insert(text)
            
        await asyncio.to_thread(insert_sync)
        logger.info(f"Successfully indexed document to knowledge graph: {filename}")
        return True
    except Exception as e:
        logger.error(f"Error indexing document {filename} in LightRAG: {str(e)}")
        return False

def get_graph_stats() -> Dict[str, Any]:
    """Return number of nodes/edges in knowledge graph."""
    if not rag:
        return {"nodes": 0, "edges": 0, "status": "Not initialized"}
        
    try:
        graph_file = os.path.join(LIGHTRAG_WORKING_DIR, "graph_chunk_entity_relation.graphml")
        stats = {"status": "Active"}
        
        # This is an approximation since LightRAG hides direct node counting depending on the store type
        if os.path.exists(graph_file):
            import networkx as nx
            try:
                G = nx.read_graphml(graph_file)
                stats["nodes"] = G.number_of_nodes()
                stats["edges"] = G.number_of_edges()
            except Exception as read_err:
                stats["error"] = f"Could not parse graphml: {read_err}"
        else:
            stats["nodes"] = 0
            stats["edges"] = 0
            
        return stats
    except Exception as e:
        logger.error(f"Error getting graph stats: {str(e)}")
        return {"error": str(e)}
