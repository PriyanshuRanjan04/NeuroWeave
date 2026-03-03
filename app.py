import gradio as gr
import asyncio
import os
import sys
import logging

# Ensure project modules are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import validate_config
from pdf_processing.extractor import process_multiple_pdfs
from rag.lightrag_client import index_document, get_graph_stats
from rag.retriever import query
from database.supabase_client import store_chunks, similarity_search
from llm.grok_client import chat, embed
from llm.handbook_generator import generate_handbook

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def handle_pdfs(files, progress=gr.Progress()):
    if not files:
        return "Please upload at least one PDF.", get_graph_stats_str()
        
    progress(0, desc="Extracting text from PDFs...")
    file_paths = [f.name for f in files]
    
    # Extract
    results = process_multiple_pdfs(file_paths)
    
    total_chunks = 0
    failed_files = []
    
    progress(0.2, desc="Processing into Database & Knowledge Graph...")
    for i, res in enumerate(results):
        if "error" in res:
            failed_files.append(f"{res['filename']} ({res['error']})")
            continue
            
        progress(0.2 + 0.6 * (i / len(results)), desc=f"Indexing {res['filename']}...")
        
        # 1. RAG Knowledge Graph Indexing
        combined_text = " ".join(res["chunks"])
        if combined_text.strip():
            await index_document(combined_text, res["filename"])
            
            # 2. Vector Store Indexing (Supabase)
            chunks = res["chunks"]
            if chunks:
                embeddings = []
                for chunk in chunks:
                    emb = await embed(chunk)
                    if not emb:
                        emb = [0.0] * 1536 # Fallback empty embedding
                    embeddings.append(emb)
                    
                await store_chunks(res["filename"], chunks, embeddings)
                total_chunks += len(chunks)
            
    progress(1.0, desc="Done!")
    
    msg = f"Processed {len(results) - len(failed_files)} PDFs successfully. Indexed {total_chunks} chunks."
    if failed_files:
        msg += f"\nFailed: {', '.join(failed_files)}"
        
    return msg, get_graph_stats_str()

def get_graph_stats_str():
    stats = get_graph_stats()
    if 'error' in stats:
        return f"Graph Error: {stats['error']}"
    return f"Graph Nodes: {stats.get('nodes', 0)} | Graph Edges: {stats.get('edges', 0)}"

async def chat_handler(message, history):
    if not message:
        return ""
        
    # 1. Retrieve context
    logger.info(f"Retrieving context for: {message}")
    context = await query(message, mode="hybrid")
    
    # 2. Build prompt
    system_prompt = (
        "You are an AI assistant for NeuroWeave, answering questions based on uploaded documents. "
        "Use the provided context to answer the user's question accurately. "
        "Cite the context where applicable."
    )
    
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{message}"
    
    # 3. Generate response
    messages = [{"role": "user", "content": user_prompt}]
    response = await chat(messages, system_prompt=system_prompt)
    
    # Append sources reference simply
    if "Error" not in response:
        response += "\n\n*(Sourced from Knowledge Graph & Vector DB)*"
    return response

async def handle_handbook(topic, progress=gr.Progress()):
    if not topic:
        return "Please enter a topic for the handbook."
        
    def update_fn(msg):
        # Progress accepts a float (0-1) or None for indeterminate animations
        progress(None, desc=msg)
        
    # Gather broad context about topic
    global_context = await query(f"Provide a comprehensive overview of {topic}", mode="global")
    
    handbook_content = await generate_handbook(topic, global_context, update_fn=update_fn)
    return handbook_content

# --- Gradio UI Layout ---
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_950",
    body_text_color="*neutral_50",
    block_background_fill="*neutral_900",
)

with gr.Blocks(title="NeuroWeave") as demo:
    gr.Markdown("# 🧠 NeuroWeave")
    gr.Markdown("AI-Powered Document Intelligence & Handbook Generator")
    
    valid_conf = validate_config()
    if not valid_conf:
        gr.Markdown("⚠️ **WARNING:** Essential API keys (.env) are missing. Certain components will fail. See terminal for warnings.")
        
    with gr.Tabs():
        # Tab 1: Upload & Process
        with gr.Tab("1. Upload Documents"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"])
                    upload_btn = gr.Button("Process Documents", variant="primary")
                with gr.Column():
                    upload_status = gr.Textbox(label="Status", interactive=False)
                    graph_stats = gr.Textbox(label="Graph Statistics", interactive=False, value=get_graph_stats_str())
                    
            upload_btn.click(
                fn=handle_pdfs,
                inputs=[file_input],
                outputs=[upload_status, graph_stats]
            )
            
        # Tab 2: Chat
        with gr.Tab("2. Chat with Documents"):
            chat_interface = gr.ChatInterface(
                fn=chat_handler,
                title="NeuroWeave Chat",
                description="Ask questions about your uploaded documents."
            )
            
        # Tab 3: Handbook Generation
        with gr.Tab("3. Generate Handbook"):
            gr.Markdown("Generate a highly structured, 20,000-word handbook based on the uploaded documents.")
            topic_input = gr.Textbox(label="Handbook Topic / Main Subject")
            generate_btn = gr.Button("Generate Handbook (This will take time)", variant="primary")
            handbook_output = gr.Markdown(label="Generated Handbook will appear here.")
            
            generate_btn.click(
                fn=handle_handbook,
                inputs=[topic_input],
                outputs=[handbook_output]
            )

if __name__ == "__main__":
    demo.launch(theme=theme)
