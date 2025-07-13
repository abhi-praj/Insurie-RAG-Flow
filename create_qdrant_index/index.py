import os
from dotenv import load_dotenv
import cohere
from qdrant_client import QdrantClient

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "underwriting-chunks"

co = cohere.Client(COHERE_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def create_carrier_index():
    """
    Create an index on the 'carrier' field for efficient filtering for future use.
    """
    try:
        from qdrant_client.models import PayloadSchemaType
        
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="carrier",
            field_schema=PayloadSchemaType.KEYWORD
        )
        print("Carrier index created successfully!")
        return True
    except Exception as e:
        print(f"Error creating carrier index: {e}")
        return False

create_carrier_index()