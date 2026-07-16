"""
app.py
Gradio interface for the Legal RAG pipeline over CUAD contracts.
Runs entirely locally — no Hugging Face Spaces required.

Usage:
    python app.py
Then open the URL it prints (usually http://127.0.0.1:7860) in your browser.
"""

import os
import gradio as gr

from ingest import build_and_save_indices
from rag_pipeline import LegalRAGPipeline

INDEX_DIR = "indices"


def build_indices_if_missing():
    if not os.path.exists(os.path.join(INDEX_DIR, "dense.index")):
        print("[app] No indices found — building from CUAD (first run only, a few minutes)...")
        build_and_save_indices(max_records=300)
    else:
        print("[app] Existing indices found, skipping ingest.")


build_indices_if_missing()
pipeline = LegalRAGPipeline()


def query_rag(question: str, retrieval_mode: str, top_k: int):
    if not question.strip():
        return "Please enter a question.", ""

    result = pipeline.answer(question, retrieval_mode=retrieval_mode, top_k=int(top_k))

    sources_md = "\n\n".join(
        f"**[{i+1}] {c['title']}**\n\n> {c['text'][:400]}{'...' if len(c['text']) > 400 else ''}"
        for i, c in enumerate(result["retrieved_chunks"])
    )
    return result["answer"], sources_md


with gr.Blocks(title="Legal RAG — CUAD Contracts") as demo:
    gr.Markdown(
        """
        # ⚖️ Legal Contract RAG Assistant
        Ask questions about commercial contracts from the **CUAD** dataset
        (Contract Understanding Atticus Dataset). Answers are generated
        **only** from retrieved contract excerpts, with citations.

        Retrieval modes:
        - **dense_only** — pure semantic (embedding) search
        - **hybrid** — BM25 + dense fused with Reciprocal Rank Fusion
        - **hybrid_rerank** — hybrid retrieval + cross-encoder reranking (most accurate)
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(
                label="Your question",
                placeholder="e.g. What is the termination notice period in this agreement?",
                lines=2,
            )
            with gr.Row():
                mode = gr.Radio(
                    ["dense_only", "hybrid", "hybrid_rerank"],
                    value="hybrid_rerank",
                    label="Retrieval mode",
                )
                top_k = gr.Slider(1, 10, value=5, step=1, label="Chunks to retrieve (top_k)")
            submit_btn = gr.Button("Ask", variant="primary")

        with gr.Column(scale=3):
            answer_box = gr.Markdown(label="Answer")
            sources_box = gr.Markdown(label="Retrieved sources")

    submit_btn.click(
        fn=query_rag,
        inputs=[question, mode, top_k],
        outputs=[answer_box, sources_box],
    )

    gr.Markdown(
        """
        ---
        Built with hybrid retrieval (BM25 + dense embeddings), cross-encoder
        reranking, and RAGAS-based evaluation. Dataset:
        [theatticusproject/cuad-qa](https://huggingface.co/datasets/theatticusproject/cuad-qa)
        """
    )

if __name__ == "__main__":
    # Plain local launch — opens on http://127.0.0.1:7860 by default.
    # inbrowser=True auto-opens it in your default browser on Mac.
    demo.queue().launch(inbrowser=True)
