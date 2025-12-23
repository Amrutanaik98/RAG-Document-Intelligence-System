import sys
import os
import json
import numpy as np
from datetime import datetime

print("="*80)
print("🚀 RAG Document Query System - FastAPI Backend")
print("="*80 + "\n")

# ============================================================================
# STEP 1: LOAD DEPENDENCIES
# ============================================================================

print("📦 Loading dependencies...\n")

try:
    import pandas as pd
    print("✅ pandas loaded")
except ImportError:
    pd = None
    print("⚠️  pandas not available")

HAS_TRANSFORMERS = False
model = None

try:
    import importlib.util
    spec = importlib.util.find_spec("sentence_transformers")
    if spec:
        import sentence_transformers
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers loaded")
        HAS_TRANSFORMERS = True
    else:
        print("⚠️  sentence-transformers not available, using fallback")
except Exception as e:
    HAS_TRANSFORMERS = False
    print(f"⚠️  Error loading sentence-transformers: {e}")

# Try to load FastAPI
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
    print("✅ FastAPI loaded")
except Exception as e:
    FASTAPI_AVAILABLE = False
    print(f"⚠️  FastAPI not available: {e}")

print()

# ============================================================================
# STEP 2: DEFINE HELPER FUNCTION FIRST (before using it)
# ============================================================================

def create_keyword_embeddings(chunks):
    """Create embeddings based on keyword presence"""
    embedding_dim = 128
    embeddings = []
    
    keywords = {
        'machine': [1, 0.8, 0.2, 0.1, 0.3, 0.4, 0.1, 0.5, 0.2, 0.1],
        'learning': [1, 0.9, 0.3, 0.2, 0.4, 0.5, 0.2, 0.6, 0.3, 0.15],
        'deep': [0.2, 1, 0.4, 0.1, 0.5, 0.3, 0.1, 0.2, 0.1, 0.05],
        'neural': [0.3, 0.9, 0.5, 0.2, 0.6, 0.4, 0.2, 0.3, 0.1, 0.1],
        'transformers': [0.1, 0.3, 1, 0.7, 0.8, 0.2, 0.1, 0.1, 0.2, 0.3],
        'nlp': [0.2, 0.2, 0.8, 1, 0.7, 0.3, 0.2, 0.1, 0.3, 0.2],
        'embeddings': [0.4, 0.5, 0.6, 0.7, 1, 0.3, 0.8, 0.2, 0.9, 0.4],
        'rag': [0.3, 0.4, 0.2, 0.1, 0.3, 1, 0.2, 0.1, 0.4, 0.5],
        'vector': [0.3, 0.3, 0.2, 0.1, 0.9, 0.3, 1, 0.1, 0.7, 0.3],
        'data': [0.6, 0.4, 0.1, 0.1, 0.3, 0.2, 0.2, 1, 0.1, 0.2],
        'semantic': [0.3, 0.3, 0.4, 0.5, 0.8, 0.3, 0.7, 0.2, 1, 0.6],
        'knowledge': [0.2, 0.2, 0.3, 0.2, 0.4, 0.3, 0.3, 0.4, 0.5, 1],
    }
    
    for chunk in chunks:
        text = chunk['chunk_text'].lower()
        embedding = np.zeros(embedding_dim)
        
        for i, (keyword, weights) in enumerate(keywords.items()):
            if keyword in text:
                for j, weight in enumerate(weights):
                    embedding[j % embedding_dim] += weight
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        embeddings.append(embedding)
    
    return np.array(embeddings)

# ============================================================================
# STEP 3: LOAD SAMPLE DATA
# ============================================================================

print("📚 Loading sample documents...\n")

SAMPLE_CHUNKS = [
    {
        "chunk_id": "chunk_001",
        "chunk_text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data and make decisions.",
        "document_id": "doc_001"
    },
    {
        "chunk_id": "chunk_002",
        "chunk_text": "Deep learning is a method of machine learning based on artificial neural networks where the learning process is deep, involving multiple layers of abstraction and representation.",
        "document_id": "doc_002"
    },
    {
        "chunk_id": "chunk_003",
        "chunk_text": "Transformers are neural network models that use attention mechanisms to process sequential data more efficiently than recurrent neural networks. They form the basis of modern NLP models like BERT and GPT.",
        "document_id": "doc_003"
    },
    {
        "chunk_id": "chunk_004",
        "chunk_text": "Natural Language Processing (NLP) is a field of artificial intelligence focused on the interaction between computers and human language. It enables machines to understand and generate human language.",
        "document_id": "doc_004"
    },
    {
        "chunk_id": "chunk_005",
        "chunk_text": "Embeddings are numerical representations of text that capture semantic meaning, allowing similar texts to have similar vectors in high-dimensional space. They are fundamental to modern NLP systems.",
        "document_id": "doc_005"
    },
    {
        "chunk_id": "chunk_006",
        "chunk_text": "Retrieval-Augmented Generation (RAG) combines information retrieval with generative models to provide more accurate and contextual responses. It improves accuracy by grounding responses in retrieved documents.",
        "document_id": "doc_006"
    },
    {
        "chunk_id": "chunk_007",
        "chunk_text": "Vector databases store and search high-dimensional vector embeddings efficiently. They enable semantic search and similarity matching at scale for AI applications and machine learning systems.",
        "document_id": "doc_007"
    },
    {
        "chunk_id": "chunk_008",
        "chunk_text": "Data engineering is the practice of designing and building systems for collecting, storing, and analyzing large amounts of data. It forms the foundation for all AI and ML systems.",
        "document_id": "doc_008"
    },
    {
        "chunk_id": "chunk_009",
        "chunk_text": "Semantic search uses embeddings to find documents by meaning rather than keywords. It understands the context and intent behind queries, providing more relevant results than traditional keyword-based search.",
        "document_id": "doc_009"
    },
    {
        "chunk_id": "chunk_010",
        "chunk_text": "Knowledge graphs represent information as interconnected nodes and relationships. They enable systems to understand complex relationships between entities and improve reasoning capabilities in AI applications.",
        "document_id": "doc_010"
    },
]

print(f"✅ Loaded {len(SAMPLE_CHUNKS)} documents\n")

# ============================================================================
# STEP 4: CREATE EMBEDDINGS (now function is defined)
# ============================================================================

print("🧠 Setting up embeddings...\n")

if HAS_TRANSFORMERS:
    try:
        print("Loading 'all-MiniLM-L6-v2' model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings_array = np.array([model.encode(chunk['chunk_text']) for chunk in SAMPLE_CHUNKS])
        print(f"✅ Created {len(embeddings_array)} real embeddings from model\n")
    except Exception as e:
        print(f"⚠️  Failed to load real embeddings: {e}")
        print("Using fallback embeddings...\n")
        HAS_TRANSFORMERS = False
        embeddings_array = create_keyword_embeddings(SAMPLE_CHUNKS)
else:
    print("Using fallback: Creating semantic-aware embeddings...\n")
    embeddings_array = create_keyword_embeddings(SAMPLE_CHUNKS)

# ============================================================================
# STEP 5: RAG CORE FUNCTIONS
# ============================================================================

def retrieve_similar_chunks(query: str, top_k: int = 5) -> list:
    """Retrieve chunks similar to query"""
    try:
        top_k = min(top_k, len(embeddings_array))
        
        # Encode query
        if HAS_TRANSFORMERS and model:
            query_embedding = model.encode(query, convert_to_tensor=False)
        else:
            # Fallback query embedding
            query_text = query.lower()
            query_embedding = np.zeros(embeddings_array.shape[1])
            
            keywords = ['machine', 'learning', 'deep', 'neural', 'transformers', 
                       'nlp', 'embeddings', 'rag', 'vector', 'data', 'semantic', 'knowledge']
            
            for keyword in keywords:
                if keyword in query_text:
                    query_embedding[hash(keyword) % len(query_embedding)] += 0.5
            
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
        
        # Calculate similarity
        similarities = np.dot(embeddings_array, query_embedding)
        
        # Normalize to [0, 1]
        if similarities.max() > similarities.min():
            similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-8)
        else:
            similarities = np.ones_like(similarities) * 0.5
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(SAMPLE_CHUNKS):
                chunk = SAMPLE_CHUNKS[idx]
                results.append({
                    'chunk_id': chunk['chunk_id'],
                    'chunk_text': chunk['chunk_text'],
                    'similarity_score': float(similarities[idx]),
                    'document_id': chunk['document_id']
                })
        
        return results
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        return []

def generate_answer(query: str, chunks: list) -> str:
    """Generate answer from retrieved chunks"""
    if not chunks:
        return "No relevant documents found."
    
    context = "RELEVANT INFORMATION FROM DOCUMENTS:\n" + "="*80 + "\n"
    
    for i, chunk in enumerate(chunks, 1):
        text_preview = chunk['chunk_text'][:300]
        context += f"\n[Source {i}] (Match: {chunk['similarity_score']:.1%})\n{text_preview}...\n"
    
    context += "="*80 + "\n"
    
    answer = f"""Based on the retrieved documents:

{context}

ANSWER TO: "{query}"

The passages above provide relevant information about your query. 
Refer to the sources listed above for detailed answers.

📊 Retrieval Stats:
   • Documents searched: {len(SAMPLE_CHUNKS)}
   • Relevant results: {len(chunks)}
   • Best match: {chunks[0]['similarity_score']:.1%}"""
    
    return answer

# ============================================================================
# STEP 6: FASTAPI SETUP (if available)
# ============================================================================

if FASTAPI_AVAILABLE:
    print("🚀 Setting up FastAPI...\n")
    
    # Define Pydantic models
    class QueryRequest(BaseModel):
        query: str
        top_k: int = 5
    
    class QueryResponse(BaseModel):
        query: str
        retrieved_chunks: list
        answer: str
        timestamp: str
    
    # Create FastAPI app
    app = FastAPI(
        title="RAG Query API",
        description="REST API for RAG-based document retrieval and Q&A",
        version="1.0.0"
    )
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "RAG Query API - Document Intelligence System",
            "version": "1.0.0",
            "status": "active"
        }
    
    @app.get("/health")
    async def health():
        """Health check"""
        return {
            "status": "healthy",
            "documents": len(SAMPLE_CHUNKS),
            "embeddings": len(embeddings_array),
            "model": "all-MiniLM-L6-v2" if HAS_TRANSFORMERS else "fallback",
            "timestamp": datetime.now().isoformat()
        }
    
    @app.post("/query")
    async def query_rag(request: QueryRequest):
        """Main query endpoint"""
        
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if request.top_k < 1 or request.top_k > 10:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")
        
        chunks = retrieve_similar_chunks(request.query, top_k=request.top_k)
        
        if not chunks:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        answer = generate_answer(request.query, chunks)
        
        return QueryResponse(
            query=request.query,
            retrieved_chunks=chunks,
            answer=answer,
            timestamp=datetime.now().isoformat()
        )

else:
    print("⚠️  FastAPI not available - using simple API mode\n")
    
    class SimpleRAGAPI:
        """Simple RAG API for environments without FastAPI"""
        
        def query(self, query_text: str, top_k: int = 5) -> dict:
            """Process query"""
            if not query_text:
                return {"error": "Query cannot be empty"}
            
            chunks = retrieve_similar_chunks(query_text, top_k=top_k)
            if not chunks:
                return {"error": "No relevant documents found"}
            
            answer = generate_answer(query_text, chunks)
            
            return {
                "query": query_text,
                "retrieved_chunks": chunks,
                "answer": answer,
                "timestamp": datetime.now().isoformat()
            }
        
        def health(self) -> dict:
            """Health check"""
            return {
                "status": "healthy",
                "documents": len(SAMPLE_CHUNKS),
                "model": "all-MiniLM-L6-v2" if HAS_TRANSFORMERS else "fallback"
            }
    
    rag_api = SimpleRAGAPI()

# ============================================================================
# INITIALIZATION COMPLETE
# ============================================================================

print("="*80)
print("✅ RAG BACKEND READY")
print("="*80)
print(f"\n📊 System Configuration:")
print(f"   • Total Documents: {len(SAMPLE_CHUNKS)}")
print(f"   • Embedding Type: {'Real' if HAS_TRANSFORMERS else 'Fallback'}")
print(f"   • Embedding Dimension: {embeddings_array.shape[1]}")
print(f"   • FastAPI Available: {'Yes' if FASTAPI_AVAILABLE else 'No'}")
print("="*80 + "\n")

# ============================================================================
# RUN SERVER (if FastAPI is available)
# ============================================================================

if FASTAPI_AVAILABLE and __name__ == "__main__":
    print("🚀 Starting FastAPI Server...\n")
    print("📍 Server URL: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔍 Redoc: http://localhost:8000/redoc")
    print("\n⏸️  Press Ctrl+C to stop\n")
    print("="*80 + "\n")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

else:
    print("✨ RAG API is ready! Use rag_api.query(text) to process queries")
    
    # Test queries
    print("\n" + "="*80)
    print("🧪 TESTING RAG SYSTEM")
    print("="*80 + "\n")
    
    test_queries = [
        "What is machine learning?",
        "Explain transformers",
        "What is RAG?"
    ]
    
    for test_query in test_queries:
        print(f"Query: {test_query}")
        result = rag_api.query(test_query, top_k=3)
        print(result['answer'][:300] + "...\n")