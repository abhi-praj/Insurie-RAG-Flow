import os
import json
from pathlib import Path
from tqdm import tqdm

from unstructured.partition.auto import partition
from unstructured.documents.elements import Table

import spacy
nlp = spacy.load("en_core_web_sm")

PDF_DIR = "Insurie Vector DB PDFs"
OUTPUT_FILE = "underwriting_semantic_chunks.json"

def extract_elements_from_pdf(file_path):
    """Extract and semantically partition PDF using Unstructured."""
    elements = partition(file_path)
    carrier_name = Path(file_path).stem

    chunks = []
    for el in elements:
        chunk = {
            "type": el.category,
            "text": el.text,
            "metadata": {
                "carrier": carrier_name,
                "page": getattr(el.metadata, "page_number", None),
            },
        }

        if isinstance(el, Table):
            chunk["table_html"] = el.to_html()
            chunk["table_markdown"] = el.to_markdown()

        # Add NER (skip for tables)
        if el.text and not isinstance(el, Table):
            doc = nlp(el.text)
            chunk["entities"] = [
                {"text": ent.text, "label": ent.label_}
                for ent in doc.ents
                if ent.label_ in ("ORG", "DATE", "MONEY", "GPE", "MEDICAL_CONDITION", "PERSON")
            ]

        chunks.append(chunk)

    return chunks

def process_all_pdfs():
    all_chunks = []
    for fname in tqdm(os.listdir(PDF_DIR), desc="Processing PDFs"):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(PDF_DIR, fname)
            try:
                chunks = extract_elements_from_pdf(path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"[ERROR] {fname}: {e}")
    return all_chunks

if __name__ == "__main__":
    chunks = process_all_pdfs()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Output saved to {OUTPUT_FILE}")
