import json
import os
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
    Loads CUAD-QA directly from the raw SQuAD-format CUAD_v1.json file
    and returns a flat list of CuadRecord objects.

    This bypasses `datasets.load_dataset`, since CUAD_v1.json is a local
    raw SQuAD-style JSON file (nested data -> paragraphs -> qas), not
    something `load_dataset` can parse correctly from a bare path.

    max_records: cap the number of records for a manageable local run.
                 300 is a good default for a Mac (fast enough to iterate,
                 still enough contracts for a meaningful eval). Set to None
                 to load everything (~13k+ QA pairs, heavier / slower).
    """
    path = "/content/CUAD_v1.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download it first, e.g.:\n"
            "  wget -O /content/CUAD_v1.json "
            "https://raw.githubusercontent.com/TheAtticusProject/cuad/main/data/CUAD_v1.json"
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records: List[CuadRecord] = []
    seen_docs = set()

    for doc in raw["data"]:
        title = doc.get("title", "untitled")
        for para in doc["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if max_records is not None and len(records) >= max_records:
                    print(f"[data_loader] Loaded {len(records)} QA records across {len(seen_docs)} contracts.")
                    return records

                question = qa["question"]
                doc_id = qa.get("id", f"{title}_{len(records)}")

                answers = qa.get("answers", [])
                answer = answers[0]["text"] if answers else None

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
