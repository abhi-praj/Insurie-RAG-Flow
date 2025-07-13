import os
from dotenv import load_dotenv
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "underwriting-chunks"

co = cohere.Client(COHERE_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def search_underwriting(query, top_k=8, carrier_filter=None, min_score=0.3):
    query_embedding = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]

    filters = None
    if carrier_filter:
        filters = Filter(
            must=[
                FieldCondition(
                    key="carrier",
                    match=MatchValue(value=carrier_filter)
                )
            ]
        )
    
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        query_filter=filters
    )

    filtered_results = [r for r in results.points if r.score >= min_score]
    
    for i, r in enumerate(filtered_results):
        print(f"\n🔹 Match {i+1} (score={r.score:.3f})")
        print(f"Carrier: {r.payload.get('carrier')} (page {r.payload.get('page')})")
        print(r.payload.get("text"))

    return filtered_results


# Test queries
search_underwriting("60 year old male, controlled type 2 diabetes, non-smoker", top_k=5)
search_underwriting("35-year-old male client, non-smoker, 5'10 tall, 180 lbs, looking for $750,000 term life insurance for 20 years. Controlled high blood pressure. What carriers would you recommend and why?", top_k=5)