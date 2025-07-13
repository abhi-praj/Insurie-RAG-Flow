import json
import re
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load semantic/NER-aware chunks from previous step
with open("underwriting_semantic_chunks.json", "r", encoding="utf-8") as f:
    raw_chunks = json.load(f)

def clean_text(text):
    """Normalize formatting and fix Unicode artifacts."""
    text = text.replace('\u2022', '-')  # bullet points
    text = text.replace('\u2013', '-')  # en-dash
    text = text.replace('\u2014', '-')  # em-dash
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = re.sub(r'\n{2,}', '\n\n', text)  # normalize line breaks
    text = re.sub(r'\s+', ' ', text)       # flatten spacing
    return text.strip()

def chunk_text(text, metadata, chunk_size=1000, overlap=200):
    """Split long text into overlapping semantic chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = splitter.create_documents([text], metadatas=[metadata])
    return docs

all_clean_chunks = []

for entry in tqdm(raw_chunks, desc="Rechunking cleaned text"):
    raw_text = entry.get("text", "")
    if not raw_text or len(raw_text.split()) < 30:
        continue

    cleaned = clean_text(raw_text)
    raw_carrier = entry.get("metadata", {}).get("carrier", "")
    carrier_clean = raw_carrier.split(" - ")[0].strip()

    metadata = {
        "carrier": carrier_clean,
        "page": entry.get("metadata", {}).get("page", None),
        "type": entry.get("type", ""),
    }

    # Include table structure or NER if available
    if "table_html" in entry:
        metadata["table_html"] = entry["table_html"]
    if "table_markdown" in entry:
        metadata["table_markdown"] = entry["table_markdown"]
    if "entities" in entry:
        metadata["entities"] = entry["entities"]

    chunks = chunk_text(cleaned, metadata)
    all_clean_chunks.extend(chunks)

# Format for saving
final_chunks = [
    {
        "text": doc.page_content,
        "metadata": doc.metadata
    }
    for doc in all_clean_chunks
]

with open("underwriting_clean_chunks.json", "w", encoding="utf-8") as f:
    json.dump(final_chunks, f, indent=2, ensure_ascii=False)

print("✅ Rechunking complete. Output: underwriting_clean_chunks.json")
