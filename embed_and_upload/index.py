import json
import uuid
from tqdm import tqdm
from dotenv import load_dotenv
import os

import cohere
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "underwriting-chunks"
BATCH_SIZE = 64
VECTOR_DIM = 1024  # embed-english-v3.0

co = cohere.Client(api_key=COHERE_API_KEY)

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

def ensure_collection():
    collections = qdrant.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)

    if not exists:
        qdrant.create_collection(
            COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection {COLLECTION_NAME} already exists")

    # Create or ensure payload index on 'carrier' field to speed up filtering
    try:
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="carrier",
            field_type="keyword"
        )
        print("Payload index on 'carrier' ensured.")
    except Exception as e:
        # If index already exists or another error occurs, just print warning and continue
        print(f"Warning: Could not create payload index on 'carrier': {e}")

def batch(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

def main():
    with open("underwriting_clean_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    ensure_collection()

    for batch_chunks in tqdm(list(batch(chunks, BATCH_SIZE))):
        texts = [c["text"] for c in batch_chunks]
        metadatas = [c["metadata"] for c in batch_chunks]

        try:
            response = co.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            embeddings = response.embeddings
        except Exception as e:
            print(f"Embedding error: {e}")
            continue

        points = []
        for i, (embedding, metadata) in enumerate(zip(embeddings, metadatas)):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": texts[i],
                    **metadata
                }
            ))

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        for p in points:
            print(p.payload)

    print(f"Our collection name on Qdrant is '{COLLECTION_NAME}'")

if __name__ == "__main__":
    main()
