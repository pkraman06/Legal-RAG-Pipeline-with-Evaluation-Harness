# Legal RAG Pipeline — CUAD Contracts (Local, Mac M4)

A hybrid retrieval-augmented generation (RAG) system over the [CUAD](https://huggingface.co/datasets/theatticusproject/cuad-qa)
legal contracts dataset, with a full evaluation harness comparing retrieval
strategies. Runs entirely on your Mac — no Hugging Face Spaces or GPU
server needed.

## Architecture

```
CUAD contracts → chunking → BM25 index + dense (FAISS) index
                                   │
                    query ─────────┼──────────────
                                   ▼
                    BM25 search        dense search
                          │                 │
                          └── RRF fusion ───┘
                                   │
                          cross-encoder rerank
                                   │
                                   ▼
                          top-k chunks → LLM (remote call) → cited answer
```

Retrieval + embeddings run locally on your Mac (accelerated via Apple's MPS
backend where available). Generation calls the Hugging Face Inference API
remotely, so you don't need to download or run a large LLM locally.

## Files

| File | Purpose |
|---|---|
| `data_loader.py` | Loads CUAD-QA from Hugging Face, reshapes into flat records |
| `ingest.py` | Chunks contracts, builds BM25 + FAISS dense indices |
| `retrieval.py` | Hybrid retrieval: dense-only / hybrid (RRF) / hybrid+rerank |
| `rag_pipeline.py` | Combines retrieval with LLM generation, produces cited answers |
| `evaluate.py` | RAGAS evaluation harness — compares all 3 retrieval modes |
| `app.py` | Gradio UI — launches locally in your browser |
| `requirements.txt` | Dependencies |

## Setup on Mac (M4 / Apple Silicon)

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Hugging Face token (needed for generation only, not dataset download)
#    Get one at https://huggingface.co/settings/tokens
#    — needs "Make calls to Inference Providers" permission
export HF_TOKEN=your_huggingface_token_here

# 4. Build the retrieval indices (one-time step, a few minutes)
#    First run also downloads CUAD automatically — no manual download needed.
python ingest.py

# 5. Try a query from the command line (optional sanity check)
python rag_pipeline.py

# 6. Run the evaluation harness — produces results.csv / results.json
python evaluate.py --n_samples 30

# 7. Launch the Gradio app
python app.py
```

Step 7 opens automatically in your browser at `http://127.0.0.1:7860`. If it
doesn't, copy that URL from the terminal output manually.

## Apple Silicon (MPS) acceleration

`ingest.py` and `retrieval.py` both auto-detect and use the `mps` device on
Apple Silicon Macs (M1–M4) for embedding generation, which is noticeably
faster than CPU. No configuration needed — this happens automatically via
`torch.backends.mps.is_available()`. If you ever see MPS-related errors
(rare, sometimes happens with specific PyTorch/macOS version combos), you
can force CPU by setting:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Where the dataset lives

CUAD is downloaded automatically on first run of `ingest.py` (via
`datasets.load_dataset`) and cached at `~/.cache/huggingface/datasets/`.
You only need an internet connection for that first run — everything after
that loads from the local cache.

## Evaluation Methodology

`evaluate.py` runs the same set of CUAD QA pairs through all three
retrieval modes and scores each with [RAGAS](https://github.com/explodinggym/ragas):

- **context_precision** — are the retrieved chunks actually relevant?
- **context_recall** — did retrieval surface everything needed to answer?
- **faithfulness** — does the generated answer stick to retrieved context?
- **answer_relevancy** — does the answer address the question asked?

Results are saved to `results.csv`. Fill in your own numbers here after
running the eval — do not reuse numbers from another run/config:

| Mode | Context Precision | Context Recall | Faithfulness | Answer Relevancy |
|---|---|---|---|---|
| dense_only | _run evaluate.py_ | | | |
| hybrid | _run evaluate.py_ | | | |
| hybrid_rerank | _run evaluate.py_ | | | |

## Dataset

[CUAD](https://huggingface.co/datasets/theatticusproject/cuad-qa) — 510
commercial legal contracts with 13,000+ expert clause annotations across 41
categories, from The Atticus Project.
