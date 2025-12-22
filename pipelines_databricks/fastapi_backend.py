# MAGIC %md
# MAGIC # FastAPI Backend for RAG System

%pip install fastapi uvicorn sentence-transformers numpy pandas

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
import json

print("✅ FastAPI backend initialized")

# ============================================================================
# Load Data
# ============================================================================

print("📥 Loading embeddings and chunks...")

# Read embeddings
embeddings_df = spark.table("chunk_embeddings").toPandas()
chunks_df = spark.table("processed_chunks").toPandas()

# Merge data
merged_df = embeddings_df.merge(
    chunks_df[['chunk_id', 'chunk_text', 'document_id']], 
    on='chunk_id', 
    how='left'
)

# Convert embeddings to numpy array
embeddings_array = np.array([np.array(emb) for emb in merged_df['embedding']])

print(f"✅ Loaded {len(embeddings_array)} embeddings")

# ============================================================================
# Load Model
# ============================================================================

print("🧠 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded")

# ============================================================================
# Define API Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for RAG query"""
    query: str
    top_k: int = 5

class RetrievedChunk(BaseModel):
    """Retrieved chunk model"""
    chunk_id: str
    chunk_text: str
    similarity_score: float

class QueryResponse(BaseModel):
    """Response model for RAG query"""
    query: str
    retrieved_chunks: list
    answer: str
    timestamp: str

# ============================================================================
# RAG Functions
# ============================================================================

def retrieve_similar_chunks(query: str, top_k: int = 5) -> list:
    """Find most similar chunks to query"""
    try:
        # Encode query
        query_embedding = model.encode(query, convert_to_tensor=False)
        
        # Calculate similarity
        similarities = np.dot(embeddings_array, query_embedding)
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Get chunks
        results = []
        for idx in top_indices:
            if idx < len(merged_df):
                row = merged_df.iloc[idx]
                results.append({
                    'chunk_id': str(row['chunk_id']),
                    'chunk_text': str(row['chunk_text']),
                    'similarity_score': float(similarities[idx])
                })
        
        return results
    except Exception as e:
        print(f"Error: {e}")
        return []

def generate_answer(query: str, chunks: list) -> str:
    """Generate answer from retrieved chunks"""
    try:
        context = "RELEVANT INFORMATION:\n" + "="*80 + "\n"
        
        for i, chunk in enumerate(chunks, 1):
            context += f"\n[Source {i}]\n{chunk['chunk_text'][:300]}...\n"
        
        context += "="*80 + "\n"
        
        answer = f"""Based on the retrieved documents:

{context}

Answer to "{query}":

The retrieved passages above contain relevant information related to your query. 
Based on the content provided, you can find answers to different aspects of your question 
in the sources listed above.

Retrieved {len(chunks)} relevant passages from {len(merged_df)} documents in the knowledge base."""
        
        return answer
    except Exception as e:
        return f"Error generating answer: {e}"

# ============================================================================
# Create FastAPI App
# ============================================================================

app = FastAPI(
    title="RAG Query API",
    description="REST API for RAG-based document retrieval",
    version="1.0.0"
)

# ============================================================================
# Routes
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Query API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/query",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "documents": len(merged_df),
        "embeddings": len(embeddings_array),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system
    
    Args:
        query: User's question
        top_k: Number of chunks to retrieve
    
    Returns:
        QueryResponse with retrieved chunks and answer
    """
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Retrieve
    chunks = retrieve_similar_chunks(request.query, top_k=request.top_k)
    
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found")
    
    # Generate answer
    answer = generate_answer(request.query, chunks)
    
    return QueryResponse(
        query=request.query,
        retrieved_chunks=chunks,
        answer=answer,
        timestamp=datetime.now().isoformat()
    )

# ============================================================================
# Startup
# ============================================================================

print("\n" + "="*80)
print("✅ RAG API READY")
print("="*80)
print(f"Documents: {len(merged_df)}")
print(f"Embeddings: {len(embeddings_array)}")
print("="*80)