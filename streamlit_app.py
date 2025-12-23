import streamlit as st
import requests
from datetime import datetime

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="RAG Document Query System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chunk-container {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .answer-box {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TITLE & DESCRIPTION
# ============================================================================

st.title("🤖 RAG Document Query System")
st.markdown("Ask questions about your documents using AI-powered semantic search")

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Configuration
    st.subheader("API Configuration")
    api_url = st.text_input(
        "FastAPI Backend URL",
        value="http://localhost:8000",
        help="URL where your FastAPI backend is running"
    )
    
    st.divider()
    
    # Query Settings
    st.subheader("Query Settings")
    top_k = st.slider(
        "Number of documents to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="Higher = more context, slower response"
    )
    
    st.divider()
    
    # System Info
    st.subheader("ℹ️ About")
    st.info("""
    **RAG Document Query System**
    
    - 📚 Uses semantic search
    - 🧠 Powered by embeddings
    - ⚡ Fast retrieval
    - 🔄 Connected to FastAPI backend
    """)

# ============================================================================
# CHECK API HEALTH
# ============================================================================

st.divider()

st.subheader("📡 System Status")

try:
    health_response = requests.get(f"{api_url}/health", timeout=3)
    if health_response.status_code == 200:
        health_data = health_response.json()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📚 Documents", health_data.get('documents', 'N/A'))
        with col2:
            st.metric("🧠 Model Type", health_data.get('model', 'unknown'))
        with col3:
            st.metric("✅ API Status", "Connected")
        
        st.success("✅ Successfully connected to FastAPI backend!")
    else:
        st.error(f"❌ API Error: Status code {health_response.status_code}")
        st.info("**Solution:** Make sure your FastAPI backend is running:\n```\npython fastapi_rag_backend_fixed.py\n```")

except requests.exceptions.ConnectionError:
    st.error("❌ Cannot connect to FastAPI backend")
    st.warning(f"**Make sure FastAPI is running at: {api_url}**")
    st.info("""
    **To start the backend, run in a terminal:**
    ```
    python fastapi_rag_backend_fixed.py
    ```
    
    It should show:
    ```
    📍 Server URL: http://localhost:8000
    📚 API Docs: http://localhost:8000/docs
    ```
    """)

except Exception as e:
    st.error(f"❌ Error: {str(e)}")

st.divider()

# ============================================================================
# QUERY SECTION
# ============================================================================

st.subheader("🔍 Ask a Question")

# Input text area
user_query = st.text_area(
    "Enter your question:",
    placeholder="e.g., What is machine learning? How do transformers work? What is RAG?",
    height=100,
    label_visibility="collapsed"
)

# Search button
col1, col2 = st.columns([4, 1])

with col2:
    submit_button = st.button("🔍 Search", use_container_width=True, type="primary")

# ============================================================================
# RESULTS SECTION
# ============================================================================

if submit_button:
    if not user_query or len(user_query.strip()) == 0:
        st.warning("⚠️ Please enter a question")
    else:
        with st.spinner("🔄 Retrieving relevant documents..."):
            try:
                # Make request to FastAPI backend
                response = requests.post(
                    f"{api_url}/query",
                    json={
                        "query": user_query,
                        "top_k": top_k
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display Answer
                    st.subheader("📖 Answer")
                    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', 
                               unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Display Retrieved Chunks
                    st.subheader("📚 Retrieved Documents")
                    
                    retrieved_chunks = result.get('retrieved_chunks', [])
                    
                    if retrieved_chunks:
                        for i, chunk in enumerate(retrieved_chunks, 1):
                            similarity_pct = chunk['similarity_score'] * 100
                            
                            with st.expander(
                                f"📄 Source {i} - Similarity: {similarity_pct:.1f}%",
                                expanded=(i == 1)
                            ):
                                # Display chunk text
                                st.markdown(f'<div class="chunk-container">{chunk["chunk_text"]}</div>', 
                                          unsafe_allow_html=True)
                                
                                # Display metadata
                                col_meta1, col_meta2 = st.columns(2)
                                with col_meta1:
                                    st.caption(f"📌 Chunk ID: {chunk['chunk_id']}")
                                with col_meta2:
                                    st.caption(f"📄 Document ID: {chunk['document_id']}")
                    else:
                        st.warning("No relevant documents found")
                    
                    st.divider()
                    
                    # Display Statistics
                    st.subheader("📊 Retrieval Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "📊 Chunks Retrieved",
                            len(retrieved_chunks)
                        )
                    
                    with col2:
                        if retrieved_chunks:
                            best_match = retrieved_chunks[0]['similarity_score'] * 100
                            st.metric(
                                "🎯 Best Match",
                                f"{best_match:.1f}%"
                            )
                        else:
                            st.metric("🎯 Best Match", "N/A")
                    
                    with col3:
                        st.metric(
                            "⏱️ Retrieved At",
                            result.get('timestamp', 'N/A').split('T')[1][:8] if 'timestamp' in result else 'N/A'
                        )
                
                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    st.write("**Response:**")
                    st.code(response.text)
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to FastAPI backend")
                st.error(f"**Make sure FastAPI is running at: {api_url}**")
                st.info("""
                **Steps to fix:**
                1. Open a new terminal
                2. Run: `python fastapi_rag_backend_fixed.py`
                3. Wait for: "📍 Server URL: http://localhost:8000"
                4. Come back here and try your query again
                """)
            
            except requests.exceptions.Timeout:
                st.error("❌ Request timeout - backend took too long to respond")
                st.info("Try again or check if backend is running smoothly")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.divider()

# ============================================================================
# EXAMPLE QUERIES
# ============================================================================

st.subheader("💡 Quick Start - Try These Queries")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("What is machine learning?", use_container_width=True, key="q1"):
        st.session_state.query = "What is machine learning?"
        st.rerun()

with col2:
    if st.button("Explain transformers in NLP", use_container_width=True, key="q2"):
        st.session_state.query = "Explain transformers in NLP"
        st.rerun()

with col3:
    if st.button("What is RAG?", use_container_width=True, key="q3"):
        st.session_state.query = "What is RAG?"
        st.rerun()

st.divider()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
---

**RAG Document Query System**

Built with:
- 🖥️ **Frontend:** Streamlit (this file)
- 🔧 **Backend:** FastAPI (fastapi_rag_backend_fixed.py)
- 🧠 **Search:** Semantic search with embeddings
- 📚 **Data:** 10 sample documents (customizable)

**Architecture:**
```
Streamlit App (Port 8501)
    ↓ (HTTP POST /query)
FastAPI Backend (Port 8000)
    ↓ (Search + Generate)
Answer + Sources
```

**To run both together:**
```bash
# Terminal 1 - Start Backend
python fastapi_rag_backend_fixed.py

# Terminal 2 - Start Frontend
streamlit run streamlit_app.py

# Browser
# Go to: http://localhost:8501
```
""")s