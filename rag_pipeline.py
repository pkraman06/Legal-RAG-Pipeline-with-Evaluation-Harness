"""
rag_pipeline.py
Ties retrieval (retrieval.py) together with an LLM call to produce a final,
citation-backed answer. Uses the Hugging Face Inference API for generation
(a lightweight remote call), so your Mac only needs to run retrieval and
embeddings locally — no local LLM weights required.

Set the HF_TOKEN environment variable with a Hugging Face access token that
has "Make calls to Inference Providers" permission:
    export HF_TOKEN=your_token_here          # macOS/Linux (zsh/bash)
Get a token at: https://huggingface.co/settings/tokens
"""

import os
from huggingface_hub import InferenceClient
from retrieval import LegalRetriever

# Any instruction-tuned model available on the HF Inference API works here.
GENERATION_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = """You are a legal contract assistant. Answer the user's question
ONLY using the provided contract excerpts. If the excerpts don't contain the
answer, say "I could not find this in the provided contract excerpts."
Always cite which excerpt(s) [1], [2], etc. you used to support each claim."""


class LegalRAGPipeline:
    def __init__(self, retrieval_mode: str = "hybrid_rerank", top_k: int = 5):
        self.retriever = LegalRetriever()
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print(
                "[rag_pipeline] WARNING: HF_TOKEN environment variable not set. "
                "Generation calls will fail until you run:\n"
                "  export HF_TOKEN=your_token_here"
            )
        self.client = InferenceClient(model=GENERATION_MODEL, token=hf_token)

    def _build_context_block(self, chunks):
        lines = []
        for i, c in enumerate(chunks, start=1):
            lines.append(f"[{i}] (Source: {c['title']})\n{c['text']}")
        return "\n\n".join(lines)

    def answer(self, question: str, retrieval_mode: str = None, top_k: int = None):
        mode = retrieval_mode or self.retrieval_mode
        k = top_k or self.top_k

        retrieved_chunks = self.retriever.retrieve(question, mode=mode, top_k=k)
        context_block = self._build_context_block(retrieved_chunks)

        user_prompt = f"""Contract excerpts:
{context_block}

Question: {question}

Answer using only the excerpts above, with citation numbers like [1], [2]."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat_completion(
            messages=messages, max_tokens=500, temperature=0.1
        )
        answer_text = response.choices[0].message.content

        return {
            "answer": answer_text,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_mode": mode,
        }


if __name__ == "__main__":
    pipeline = LegalRAGPipeline()
    result = pipeline.answer("What is the governing law of this agreement?")
    print(result["answer"])
    print("\nSources used:")
    for c in result["retrieved_chunks"]:
        print(f"- {c['title']} ({c['chunk_id']})")
