import os
import json
from pathlib import Path
from tqdm import tqdm

from unstructured.partition.auto import partition

PDF_DIR = "Insurie Vector DB PDFs"
OUTPUT_FILE = "underwriting_semantic_chunks.json"

def extract_elements_from_pdf(file_path):
    """Extract and semantically partition PDF using Unstructured."""
    elements = partition(file_path)
    carrier_name = Path(file_path).stem

    full_text = " ".join([el.text for el in elements if el.text])

    chunk = {
        "type": "document",
        "text": full_text,
        "metadata": {
            "carrier": carrier_name,
            "source_file": str(file_path),
        }
    }

    return [chunk]

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
