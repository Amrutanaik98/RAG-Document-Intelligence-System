import streamlit as st
import requests
import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title="RAG Document Query System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 0rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Header
# ============================================================================

st.title("🤖 RAG Document Query System")
st.markdown("Ask questions about your documents using AI-powered retrieval")

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Configuration
    st.subheader("API Configuration")
    api_url = st.text_input(
        "API URL",
        value="http://localhost:8000",
        help="Base URL of FastAPI backend"
    )
    
    # Query Settings
    st.subheader("Query Settings")
    top_k = st.slider(
        "Number of relevant documents to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="Higher = more context, slower response"
    )
    
    # Info
    st.divider()
    st.subheader("ℹ️ About")
    st.info("""
    This RAG system:
    - 📚 Contains 109+ documents
    - 🔍 Uses semantic search
    - 🧠 Powered by AI embeddings
    - ⚡ Fast retrieval (< 1 second)
    """)


try:
    health_response = requests.get(f"{api_url}/health", timeout=5)
    if health_response.status_code == 200:
        health_data = health_response.json()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📚 Documents", health_data['documents'])
        with col2:
            st.metric("🔢 Embeddings", health_data['embeddings'])
        with col3:
            st.metric("✅ API Status", "Connected")
    else:
        st.error("❌ API Connection Failed")
except:
    st.error("❌ Cannot connect to API. Make sure FastAPI is running on " + api_url)

st.divider()


st.subheader("🔍 Ask a Question")

# Input
user_query = st.text_area(
    "Enter your question:",
    placeholder="e.g., What is machine learning? How do transformers work?",
    height=100,
    label_visibility="collapsed"
)

# Submit button
col1, col2 = st.columns([4, 1])
with col2:
    submit_button = st.button("🔍 Search", use_container_width=True)

# ============================================================================
# Results Section
# ============================================================================

if submit_button:
    if not user_query or len(user_query.strip()) == 0:
        st.warning("Please enter a question")
    else:
        with st.spinner("🔄 Retrieving relevant documents..."):
            try:
                # Query API
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
                    st.markdown(result['answer'])
                    
                    st.divider()
                    
                    # Display Retrieved Chunks
                    st.subheader("📚 Retrieved Documents")
                    
                    for i, chunk in enumerate(result['retrieved_chunks'], 1):
                        with st.expander(
                            f"📄 Source {i} - Similarity: {chunk['similarity_score']:.3f}",
                            expanded=(i == 1)
                        ):
                            st.markdown(chunk['chunk_text'])
                            st.caption(f"Chunk ID: {chunk['chunk_id']}")
                    
                    # Metadata
                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📊 Chunks Retrieved", len(result['retrieved_chunks']))
                    with col2:
                        st.metric("⏱️ Retrieved At", result['timestamp'])
                    
                    # Save Query
                    if st.button("💾 Save Query"):
                        st.success("Query saved!")
                
                else:
                    st.error(f"API Error: {response.status_code}")
                    st.write(response.text)
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure it's running!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ============================================================================
# Example Queries
# ============================================================================

st.divider()
st.subheader("💡 Example Queries")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("What is machine learning?"):
        st.session_state.query = "What is machine learning?"

with col2:
    if st.button("Explain transformers"):
        st.session_state.query = "Explain transformers in NLP"

with col3:
    if st.button("What is deep learning?"):
        st.session_state.query = "What is deep learning?"

# ============================================================================
# Footer
# ============================================================================

st.divider()
st.markdown("""
---
**RAG Document Query System** | Powered by FastAPI + Streamlit + Databricks
""")