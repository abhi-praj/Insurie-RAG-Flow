import json
import re
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter

with open("underwriting_raw_chunks.json", "r") as f:
    raw_chunks = json.load(f)

def clean_text(text):
    """Fixes weird formatting and unicode artifacts."""
    text = text.replace('\u2022', '-')  # bullet points
    text = text.replace('\u2013', '-')  # en-dash
    text = text.replace('\u2014', '-')  # em-dash
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = re.sub(r'\n{2,}', '\n\n', text)  # normalize line breaks
    text = re.sub(r'\s+', ' ', text)       # flatten weird spacing
    return text.strip()

def chunk_text(text, metadata, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    docs = splitter.create_documents([text], metadatas=[metadata])
    return docs

all_clean_chunks = []

for entry in tqdm(raw_chunks):
    raw_carrier = entry["carrier"]
    page = entry["page"]
    raw_text = entry["text"]

    carrier = raw_carrier.split(" - ")[0]
    cleaned = clean_text(raw_text)

    if len(cleaned.split()) < 30:
        continue

    metadata = {
        "carrier": carrier,
        "page": page,
    }

    chunks = chunk_text(cleaned, metadata)
    all_clean_chunks.extend(chunks)

final_chunks = [
    {
        "text": doc.page_content,
        "metadata": doc.metadata
    }
    for doc in all_clean_chunks
]

with open("underwriting_clean_chunks.json", "w") as f:
    json.dump(final_chunks, f, indent=2)

print("done")
