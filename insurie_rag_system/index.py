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
    """
    Search for relevant underwriting guidelines in Qdrant
    """
    # Embed query as a search query
    query_embedding = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]

    # Optional: filter by carrier
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

    # Search Qdrant using query_points
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        query_filter=filters
    )

    # Parse results and filter by score
    filtered_results = [r for r in results.points if r.score >= min_score]
    
    return filtered_results

def generate_answer(query, search_results, max_tokens=1500):
    """
    Generate a natural answer using Command R based on retrieved chunks
    """
    
    # Prepare context from search results
    context_pieces = []
    for i, result in enumerate(search_results):
        carrier = result.payload.get('carrier', 'Unknown')
        page = result.payload.get('page', 'Unknown')
        text = result.payload.get('text', '')
        score = result.score
        
        context_pieces.append(f"Source {i+1} (Carrier: {carrier}, Page: {page}, Relevance: {score:.3f}):\n{text}")
    
    context = "\n\n".join(context_pieces)
    
    # Create prompt for Command R using Insurie's system prompt
    system_prompt = """# Insurie - Insurance Assistant & Carrier Recommendation Specialist

You are **Insurie**, a comprehensive insurance assistant specializing in Canadian life insurance. You help with both general insurance education and specific carrier recommendations for brokerages.

## PRIMARY FUNCTIONS

### 1. General Insurance Questions
- Answer questions about insurance concepts, products, and processes
- Explain differences between insurance types (term, whole life, universal life)
- Clarify underwriting processes and requirements
- Provide educational content about coverage options
- Help users understand insurance terminology and concepts
- Answer general questions related to the client as a part of normal conversational flow

### 2. Carrier Recommendations
- Analyze client profiles against carrier underwriting guidelines
- Recommend specific carriers based on client risk factors
- Explain underwriting fit and competitive advantages
- Suggest optimal strategies for specific client situations

### 3. Conversation Management
- **Insurance-Related Topics**: Engage fully and helpfully
- **Non-Insurance Topics**: Politely redirect to insurance matters

## KNOWLEDGE BASE

Your indexed carrier knowledge includes:
Assumption Life, Beneva, CPP (AZ | EE | Express | CI), Desjardins, Emma, Empire Life, Equitable Life, Foresters, Humania, Industrial Alliance, ivari, Manulife (Accelerated, Financial UW, New Canadians), UV Insurance

## RECOMMENDATION OUTPUT FORMAT

When providing carrier recommendations, always include the two best carriers for the client. Rather than just rates, consider different factors such as:
- **Rates**: What would offer good rates for the client?
- **Underwriting Fit**: How well does each carrier fit with the client's risk profile?
- **Processing Speed**: How quickly can the underwriting process be completed?
- **Competitive Advantage**: How does each carrier compare to others in the market?
- **Ease of Access**: Which plan overall offers the best bang for your buck for the client along with the likelyness to accept the client?

NEVER EVER CONSIDER N0-MEDICAL OR BEST RATES SOLEY, CONSIDER ALL FACTORS

### **4. Next Steps**
Specific actions for the broker to take. Limit to just 1-2 steps

## COMMUNICATION STYLE

**For General Questions:**
- Friendly and educational
- Be concise and to the point

**For Recommendations:**
- Professional and analytical
- Explain reasoning clearly yet concisely
- Reference underwriting factors
- Provide actionable guidance"""

    prompt = f"""{system_prompt}

CONTEXT FROM UNDERWRITING GUIDELINES:
{context}

USER QUESTION: {query}

Based on the underwriting guidelines provided above, please provide a comprehensive yet concise answer as Insurie You don't need to introduce or reference yourself. Reference specific carriers and their guidelines when relevant, and if this appears to be a carrier recommendation request, follow the recommendation output format. Be concise, to the point, and relevant.

ANSWER:"""

    try:
        # Generate response using Command R
        response = co.chat(
            model="command-r-plus",
            message=prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            p=0.9,
            k=0,
            prompt_truncation="AUTO"
        )
        
        return response.text
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

def rag_query(query, top_k=8, carrier_filter=None, min_score=0.3, max_tokens=1500, show_sources=True):
    """
    Complete RAG pipeline: search + generate answer
    """
    
    print(f"Searching for: {query}")
    print("=" * 80)
    
    # Step 1: Search for relevant chunks
    search_results = search_underwriting(query, top_k, carrier_filter, min_score)
    
    if not search_results:
        return "No relevant information found in the underwriting guidelines."
    
    # Step 2: Show search results (optional)
    if show_sources:
        print("\nRELEVANT SOURCES:")
        for i, result in enumerate(search_results):
            print(f"\nSource {i+1} (Score: {result.score:.3f})")
            print(f"Carrier: {result.payload.get('carrier')} (Page: {result.payload.get('page')})")
            print(f"Text: {result.payload.get('text')[:200]}...")
            print("-" * 60)
    
    # Step 3: Generate answer using Command R
    print("\nINSURIE'S ANALYSIS:")
    print("=" * 80)
    
    answer = generate_answer(query, search_results, max_tokens)
    print(answer)
    
    return {
        'query': query,
        'sources': search_results,
        'answer': answer
    }

# Example usage and test cases
if __name__ == "__main__":
    print("Testing Insurie RAG System with Command R")
    print("=" * 80)
    
    # Test 1: General insurance question
    print("\n" + "="*50)
    print("TEST 1: General Insurance Question")
    print("="*50)
    
    general_query = "What is Type 2 diabetes and how does it affect life insurance underwriting?"
    result1 = rag_query(general_query, top_k=5, show_sources=True)
    
    # Test 2: Carrier recommendation request
    print("\n" + "="*50)
    print("TEST 2: Carrier Recommendation Request")
    print("="*50)
    
    recommendation_query = "60 year old male, controlled type 2 diabetes, non-smoker, looking for $500k term life insurance. Which carriers would you recommend? Which ones would you not reccomend and why?"
    result2 = rag_query(recommendation_query, top_k=8, show_sources=True)
    
    # Test 3: Specific carrier filtering
    print("\n" + "="*50)
    print("TEST 3: Carrier-Specific Query")
    print("="*50)
    
    carrier_query = "What are Foresters' underwriting guidelines for diabetes?"
    result3 = rag_query(carrier_query, top_k=5, carrier_filter="Foresters", show_sources=True)
    
    print("\nRAG System Testing Complete!")
    print("=" * 80)

    carrier_query = "I have a 35-year-old male client, non-smoker, 5'10 tall, 180 lbs, looking for $750,000 term life insurance for 20 years. He has controlled high blood pressure. What carriers would you recommend and why? Why would you not reccomend Manulife"
    result4 = rag_query(carrier_query, top_k=8, show_sources=True)
    print("\nRAG System Testing Complete!")
    print("=" * 80)