"""
data_loader.py
Loads the CUAD (Contract Understanding Atticus Dataset) QA dataset from
Hugging Face and reshapes it into a flat list of (context, question, answer)
records that are easy to chunk and index for RAG.

Dataset card: https://huggingface.co/datasets/theatticusproject/cuad-qa

Nothing here needs to be downloaded manually — the first call to
load_dataset() below fetches and caches CUAD locally (under
~/.cache/huggingface/datasets/) automatically. You just need an internet
connection the first time you run this.
"""

from datasets import load_dataset
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CuadRecord:
    doc_id: str          # unique id for the source contract
    title: str           # contract title, e.g. "AdvisorAgreement"
    context: str         # full contract text chunk used to build the QA pair
    question: str        # clause-related question, e.g. "Highlight the parts... Governing Law"
    answer: Optional[str]  # gold answer text (may be empty if unanswerable)


def load_cuad(split: str = "train", max_records: Optional[int] = 300) -> List[CuadRecord]:
    """
    Loads CUAD-QA and returns a flat list of CuadRecord objects.

    max_records: cap the number of records for a manageable local run.
                 300 is a good default for a Mac (fast enough to iterate,
                 still enough contracts for a meaningful eval). Set to None
                 to load everything (~13k+ QA pairs, heavier / slower).
    """
    # CUAD-QA ships a custom dataset loading script rather than plain
    # parquet/CSV files, so recent `datasets` versions require an explicit
    # opt-in before executing it. The Atticus Project is a well-known,
    # reputable source (see https://huggingface.co/datasets/theatticusproject/cuad-qa),
    # so this is safe to allow here.
    ds = load_dataset("theatticusproject/cuad-qa", split=split, trust_remote_code=True)

    records: List[CuadRecord] = []
    seen_docs = set()

    for i, row in enumerate(ds):
        if max_records is not None and len(records) >= max_records:
            break

        title = row.get("title", f"doc_{i}")
        context = row["context"]
        question = row["question"]

        answers = row.get("answers", {})
        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
        answer = answer_texts[0] if answer_texts else None

        doc_id = row.get("id", f"{title}_{i}")

        records.append(
            CuadRecord(
                doc_id=doc_id,
                title=title,
                context=context,
                question=question,
                answer=answer,
            )
        )
        seen_docs.add(title)

    print(f"[data_loader] Loaded {len(records)} QA records across {len(seen_docs)} contracts.")
    return records


def get_unique_contracts(records: List[CuadRecord]):
    """
    Returns a dict of {title: full_context_text} — one entry per unique contract,
    used for building the retrieval corpus (as opposed to the QA eval set).
    """
    contracts = {}
    for r in records:
        if r.title not in contracts:
            contracts[r.title] = r.context
    return contracts


if __name__ == "__main__":
    recs = load_cuad(max_records=50)
    contracts = get_unique_contracts(recs)
    print(f"Example question: {recs[0].question}")
    print(f"Example answer: {recs[0].answer}")
    print(f"Unique contracts loaded: {len(contracts)}")
