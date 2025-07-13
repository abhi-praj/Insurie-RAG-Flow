import os
import pdfplumber
import re
from tqdm import tqdm
from pathlib import Path

PDF_DIR = "Insurie Vector DB PDFs"

def extract_text_from_pdf(file_path):
    """Extracts text from PDF file, page by page."""
    carrier_name = Path(file_path).stem
    extracted = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                text = page.extract_text()
                if text:
                    extracted.append({
                        "carrier": carrier_name,
                        "page": i + 1,
                        "text": text
                    })
            except Exception as e:
                print(f"Error on page {i + 1} of {file_path}: {e}")

    return extracted

def split_into_sections(text):
    """Split long text block into rough sections based on ALL CAPS or Title headings."""
    section_split_pattern = r'(?=(?:\n[A-Z][^\n]{3,80}\n))'

    sections = re.split(section_split_pattern, text)
    clean_sections = []

    for raw in sections:
        clean = raw.strip()
        if len(clean) > 100:  # ignore tiny fragments
            clean_sections.append(clean)

    return clean_sections

def process_pdf(file_path):
    pages = extract_text_from_pdf(file_path)

    all_chunks = []

    for page_data in pages:
        carrier = page_data["carrier"]
        page = page_data["page"]
        text = page_data["text"]

        sections = split_into_sections(text)

        for section in sections:
            all_chunks.append({
                "carrier": carrier,
                "page": page,
                "text": section
            })

    return all_chunks

def run_all():
    all_data = []
    for fname in tqdm(os.listdir(PDF_DIR)):
        if fname.endswith(".pdf"):
            path = os.path.join(PDF_DIR, fname)
            chunks = process_pdf(path)
            all_data.extend(chunks)

    return all_data

if __name__ == "__main__":
    all_chunks = run_all()

    import json
    with open("underwriting_raw_chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2)

    print("done")
