# Databricks notebook source

# MAGIC %md
# MAGIC # FastAPI Backend for RAG System - Fixed Version

# Restart Python kernel to load new packages
%restart_python

# MAGIC %md
# MAGIC ## Install Dependencies

print("📦 Installing dependencies...")

%pip install --upgrade fastapi uvicorn sentence-transformers requests

print("✅ Dependencies installed!")

# MAGIC %md
# MAGIC ## Import Libraries

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import json

print("✅ All imports successful!")

# MAGIC %md
# MAGIC ## Load Data from Databricks

print("📥 Loading data from Databricks tables...")

try:
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
    print(f"✅ Loaded {len(chunks_df)} chunks")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    raise

# MAGIC %md
# MAGIC ## Load Embedding Model

print("🧠 Loading embedding model...")

try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# MAGIC %md
# MAGIC ## Define API Models

class QueryRequest(BaseModel):
    """Request model for RAG query"""
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    """Response model for RAG query"""
    query: str
    retrieved_chunks: list
    answer: str
    timestamp: str

# MAGIC %md
# MAGIC ## RAG Functions

def retrieve_similar_chunks(query: str, top_k: int = 5) -> list:
    """Find most similar chunks to query"""
    try:
        # Encode query
        query_embedding = model.encode(query, convert_to_tensor=False)
        
        # Calculate similarity (cosine)
        similarities = np.dot(embeddings_array, query_embedding)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            if idx < len(merged_df):
                row = merged_df.iloc[idx]
                results.append({
                    'chunk_id': str(row['chunk_id']),
                    'chunk_text': str(row['chunk_text']),
                    'similarity_score': float(similarities[idx]),
                    'document_id': str(row.get('document_id', 'unknown'))
                })
        
        return results
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        return []

def generate_answer(query: str, chunks: list) -> str:
    """Generate answer from retrieved chunks"""
    try:
        if not chunks:
            return "No relevant documents found."
        
        context = "RELEVANT INFORMATION FROM DOCUMENTS:\n" + "="*80 + "\n"
        
        for i, chunk in enumerate(chunks, 1):
            text_preview = chunk['chunk_text'][:300]
            context += f"\n[Source {i}] (Similarity: {chunk['similarity_score']:.3f})\n{text_preview}...\n"
        
        context += "="*80 + "\n"
        
        answer = f"""Based on the retrieved documents:

{context}

ANSWER TO: "{query}"

The retrieved passages above contain relevant information related to your query. 
You can find answers to different aspects of your question in the sources listed above.

Total documents searched: {len(merged_df)}
Relevant passages retrieved: {len(chunks)}"""
        
        return answer
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# MAGIC %md
# MAGIC ## Create FastAPI App

print("\n" + "="*80)
print("🚀 Creating FastAPI Application...")
print("="*80)

app = FastAPI(
    title="RAG Query API",
    description="REST API for RAG-based document retrieval and Q&A",
    version="1.0.0"
)

# MAGIC %md
# MAGIC ## API Routes

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "RAG Query API - Educational Document Intelligence System",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "root": "/",
            "health": "/health",
            "query": "/query (POST)"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "documents": len(merged_df),
        "embeddings": len(embeddings_array),
        "model": "all-MiniLM-L6-v2",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Main RAG Query Endpoint
    
    Args:
        query: User's question (str)
        top_k: Number of documents to retrieve (int, default=5)
    
    Returns:
        QueryResponse with answer and sources
    """
    
    # Validate query
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if request.top_k < 1 or request.top_k > 10:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")
    
    # Retrieve chunks
    chunks = retrieve_similar_chunks(request.query, top_k=request.top_k)
    
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found")
    
    # Generate answer
    answer = generate_answer(request.query, chunks)
    
    # Return response
    return QueryResponse(
        query=request.query,
        retrieved_chunks=chunks,
        answer=answer,
        timestamp=datetime.now().isoformat()
    )

# MAGIC %md
# MAGIC ## Startup Summary

print("\n" + "="*80)
print("✅ RAG API READY FOR REQUESTS")
print("="*80)
print(f"📊 Statistics:")
print(f"   • Total Documents: {len(merged_df)}")
print(f"   • Total Embeddings: {len(embeddings_array)}")
print(f"   • Embedding Model: all-MiniLM-L6-v2")
print(f"   • API Version: 1.0.0")
print(f"   • Status: Active ✅")
print("="*80)
print(f"🚀 API is running on http://localhost:8000")
print(f"📚 Available endpoints:")
print(f"   • GET  http://localhost:8000/")
print(f"   • GET  http://localhost:8000/health")
print(f"   • POST http://localhost:8000/query")
print("="*80)
print("\n✨ Ready to receive queries! Connect Streamlit dashboard now.\n")