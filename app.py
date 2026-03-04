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
from llm.grok_client import chat, embed, stream_chat
from llm.handbook_generator import generate_handbook

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def handle_pdfs(files, progress=gr.Progress()):
    if not files:
        return "Please upload at least one PDF.", get_graph_stats_str()

    progress(0, desc="Extracting text from PDFs...")
    file_paths = [f.name for f in files]

    # Extract text + chunks from all PDFs (fast — pure Python)
    results = process_multiple_pdfs(file_paths)

    total_chunks = 0
    failed_files = []
    docs_for_rag: list = []   # collected for background graph indexing

    progress(0.2, desc="Embedding & storing chunks in vector database...")
    for i, res in enumerate(results):
        if "error" in res:
            failed_files.append(f"{res['filename']} ({res['error']})")
            continue

        progress(0.2 + 0.6 * (i / len(results)), desc=f"Storing {res['filename']}...")

        chunks = res.get("chunks", [])
        combined_text = " ".join(chunks)

        # ── Step 1: Supabase vector store (fast, ~5 s) ──────────────────
        if chunks:
            embeddings = []
            for chunk in chunks:
                emb = await embed(chunk)
                if not emb:
                    emb = [0.0] * 1536
                embeddings.append(emb)
            await store_chunks(res["filename"], chunks, embeddings)
            total_chunks += len(chunks)

        # ── Step 2: Queue for background LightRAG graph indexing ─────────
        if combined_text.strip():
            docs_for_rag.append((combined_text, res["filename"]))

    # Fire-and-forget: knowledge graph building runs in background.
    # Chat can already answer via Supabase fallback while this processes.
    async def _background_rag():
        for text, fname in docs_for_rag:
            try:
                await index_document(text, fname)
                logger.info(f"Background RAG indexing complete: {fname}")
            except Exception as e:
                logger.error(f"Background RAG indexing failed for {fname}: {e}")

    if docs_for_rag:
        asyncio.create_task(_background_rag())
        logger.info(f"Started background knowledge-graph indexing for {len(docs_for_rag)} doc(s).")

    progress(1.0, desc="Done! Knowledge graph building in background...")

    msg = (
        f"✅ Processed {len(results) - len(failed_files)} PDF(s) — "
        f"{total_chunks} chunks stored. You can chat now!\n"
        f"⏳ Knowledge graph is building in the background (may take a few minutes)."
    )
    if failed_files:
        msg += f"\n❌ Failed: {', '.join(failed_files)}"

    return msg, get_graph_stats_str()

def get_graph_stats_str():
    stats = get_graph_stats()
    if 'error' in stats:
        return f"Graph Error: {stats['error']}"
    return f"Graph Nodes: {stats.get('nodes', 0)} | Graph Edges: {stats.get('edges', 0)}"

async def chat_handler(message, history):
    if not message:
        yield ""
        return

    # 1. Retrieve context — try LightRAG knowledge graph first
    logger.info(f"Retrieving context for: {message}")
    context = await query(message, mode="hybrid")

    # Guard: LightRAG aquery() returns None or unhelpful strings when the
    # knowledge graph is empty / still building. Fall back to Supabase
    # vector search to serve raw document chunks in that case.
    _EMPTY_SIGNALS = {"", "none", "no information found", "no context"}
    if not context or str(context).strip().lower() in _EMPTY_SIGNALS:
        logger.info("LightRAG returned no context — falling back to Supabase vector search.")
        try:
            query_emb = await embed(message)
            if query_emb:
                hits = await similarity_search(query_emb, limit=6)
                if hits:
                    chunks = [h["content"] for h in hits if h.get("content")]
                    context = "\n\n---\n\n".join(chunks)
                    logger.info(f"Supabase fallback returned {len(chunks)} chunks.")
        except Exception as emb_err:
            logger.warning(f"Supabase fallback failed: {emb_err}")

    # Final safety net
    if not context or not str(context).strip():
        context = (
            "No document context is available yet. "
            "Please upload and process a PDF in the 'Upload Documents' tab first."
        )

    # 2. Build prompt
    system_prompt = (
        "You are an AI assistant for NeuroWeave, answering questions "
        "based on uploaded documents. Use the provided context to answer "
        "accurately. If the context is insufficient, say so clearly."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion:\n{message}"
    messages = [{"role": "user", "content": user_prompt}]

    # 3. Stream response token by token
    partial = ""
    async for chunk in stream_chat(messages, system_prompt=system_prompt):
        partial += chunk
        yield partial

async def handle_handbook(topic, temperature, max_length, progress=gr.Progress()):
    if not topic:
        return "Please enter a topic.", "Word count: 0", None

    def update_fn(msg):
        progress(None, desc=msg)

    # Gather broad context about topic
    global_context = await query(
        f"Provide a comprehensive overview of {topic}",
        mode="global"
    )

    result = await generate_handbook(topic, global_context, update_fn=update_fn)

    # Handle both string and dict return types
    if isinstance(result, dict):
        content = result.get("content", str(result))
        word_count = result.get("word_count", len(content.split()))
    else:
        content = result
        word_count = len(content.split())

    # Save to temp file for download
    download_path = "handbook_output.md"
    with open(download_path, "w", encoding="utf-8") as f:
        f.write(content)

    return content, f"📊 Word Count: {word_count:,} words", download_path

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
    gr.Markdown("""
# 🧠 NeuroWeave
### AI-Powered Document Intelligence & Handbook Generator
> Upload PDFs → Chat with your documents → Generate 20,000-word handbooks
""")
    
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
                description="Ask questions about your uploaded documents.",
                examples=[
                    "What are the main topics in the uploaded documents?",
                    "Summarize the key findings",
                    "What methodology was used?"
                ]
            )
            
        # Tab 3: Handbook Generation
        with gr.Tab("3. Generate Handbook"):
            gr.Markdown("## 📚 Generate 20,000-Word Handbook")
            gr.Markdown("Generate a highly structured handbook based on uploaded documents.")

            with gr.Row():
                with gr.Column(scale=3):
                    topic_input = gr.Textbox(
                        label="Handbook Topic",
                        placeholder="e.g. Retrieval Augmented Generation",
                        lines=2
                    )
                with gr.Column(scale=1):
                    temp_slider = gr.Slider(
                        0.1, 1.0, value=0.7, step=0.1,
                        label="Temperature",
                        interactive=True
                    )
                    max_len_slider = gr.Slider(
                        5000, 32000, value=20000, step=1000,
                        label="Target Length (words)",
                        interactive=True
                    )

            generate_btn = gr.Button(
                "🚀 Generate Handbook",
                variant="primary",
                size="lg"
            )

            word_count_display = gr.Textbox(
                label="Generation Stats",
                interactive=False,
                value="Word count will appear here after generation"
            )

            handbook_output = gr.Markdown(label="Generated Handbook")

            download_btn = gr.File(label="📥 Download Handbook (.md)")

            generate_btn.click(
                fn=handle_handbook,
                inputs=[topic_input, temp_slider, max_len_slider],
                outputs=[handbook_output, word_count_display, download_btn]
            )

    with gr.Row():
        gr.Markdown(
            "Built with Gradio · LightRAG · Supabase · Groq API | "
            "LongWriter AgentWrite Technique"
        )

if __name__ == "__main__":
    demo.launch(theme=theme)
