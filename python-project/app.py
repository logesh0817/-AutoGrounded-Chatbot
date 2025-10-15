import streamlit as st
import os
from pathlib import Path
from rag_pipeline import RAGPipeline
from document_processor import DocumentProcessor
from vector_store import VectorStore
from verification import ResponseVerifier
import config

# Page configuration
st.set_page_config(
    page_title="AutoGrounded Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .confidence-high {
        color: #10b981;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f59e0b;
        font-weight: bold;
    }
    .confidence-low {
        color: #ef4444;
        font-weight: bold;
    }
    .source-citation {
        background-color: #f3f4f6;
        color: #1f2937;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .stNumberInput {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()

if "document_processor" not in st.session_state:
    st.session_state.document_processor = DocumentProcessor()

if "verifier" not in st.session_state:
    st.session_state.verifier = ResponseVerifier()

# Initialize settings
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = config.DEFAULT_CHUNK_SIZE

if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = config.DEFAULT_CHUNK_OVERLAP

if "top_k" not in st.session_state:
    st.session_state.top_k = config.DEFAULT_TOP_K_RESULTS

# Initialize directories
os.makedirs(config.DOCUMENTS_DIR, exist_ok=True)
os.makedirs(config.VECTOR_DB_DIR, exist_ok=True)

# Auto-initialize RAG pipeline if documents exist
if st.session_state.rag_pipeline is None and st.session_state.vector_store.get_document_count() > 0:
    st.session_state.rag_pipeline = RAGPipeline(st.session_state.vector_store)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Advanced settings in expander
    with st.expander("Advanced Settings", expanded=False):
        # Chunk size control
        new_chunk_size = st.slider(
            "Chunk Size (words)",
            min_value=config.MIN_CHUNK_SIZE,
            max_value=config.MAX_CHUNK_SIZE,
            value=st.session_state.chunk_size,
            step=50,
            help="Number of words per chunk. Larger chunks = more context, smaller = more precise"
        )
        
        # Chunk overlap control
        new_chunk_overlap = st.slider(
            "Chunk Overlap (words)",
            min_value=config.MIN_OVERLAP,
            max_value=min(config.MAX_OVERLAP, new_chunk_size - 10),
            value=min(st.session_state.chunk_overlap, new_chunk_size - 10),
            step=10,
            help="Overlap between chunks. More overlap = better context continuity"
        )
        
        # Top K control
        new_top_k = st.slider(
            "Search Results (Top K)",
            min_value=config.MIN_TOP_K,
            max_value=config.MAX_TOP_K,
            value=st.session_state.top_k,
            step=1,
            help="Number of relevant chunks to retrieve. More = more context but slower"
        )
        
        # Update settings
        settings_changed = (
            new_chunk_size != st.session_state.chunk_size or
            new_chunk_overlap != st.session_state.chunk_overlap or
            new_top_k != st.session_state.top_k
        )
        
        if settings_changed:
            st.session_state.chunk_size = new_chunk_size
            st.session_state.chunk_overlap = new_chunk_overlap
            st.session_state.top_k = new_top_k
            # Update document processor with new settings
            st.session_state.document_processor = DocumentProcessor(
                chunk_size=new_chunk_size,
                chunk_overlap=new_chunk_overlap
            )
            st.info("⚠️ Settings updated! Reprocess documents to apply changes.")
    
    st.markdown("---")
    st.markdown("### 📚 Document Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT, MD)",
        type=['pdf', 'txt', 'md'],
        accept_multiple_files=True,
        help="Upload documents to build your knowledge base"
    )
    
    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                processed_count = 0
                total_chunks = 0
                
                for uploaded_file in uploaded_files:
                    try:
                        # Save file
                        file_path = os.path.join(config.DOCUMENTS_DIR, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process document with current settings
                        chunks = st.session_state.document_processor.process_document(file_path)
                        
                        # Add to vector store
                        st.session_state.vector_store.add_documents(chunks, uploaded_file.name)
                        
                        processed_count += 1
                        total_chunks += len(chunks)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                if processed_count > 0:
                    st.success(f"✅ Processed {processed_count} document(s) into {total_chunks} chunks!")
                    
                    # Initialize RAG pipeline
                    st.session_state.rag_pipeline = RAGPipeline(st.session_state.vector_store)
                    st.rerun()
    
    # Display loaded documents
    st.markdown("### 📄 Loaded Documents")
    if st.session_state.vector_store.index is not None:
        doc_count = st.session_state.vector_store.get_document_count()
        
        # Get unique document names
        unique_docs = set()
        for doc in st.session_state.vector_store.documents:
            unique_docs.add(doc['document'])
        
        st.metric("Total Chunks", doc_count)
        st.metric("Documents", len(unique_docs))
        
        # Show document list
        if unique_docs:
            with st.expander("View Documents", expanded=False):
                for doc_name in sorted(unique_docs):
                    st.markdown(f"📄 {doc_name}")
    else:
        st.warning("No documents loaded")
    
    # Clear conversation
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
    
    # System info
    st.markdown("---")
    st.markdown("### 📊 Current Settings")
    st.markdown(f"**Model:** {config.MODEL_NAME}")
    st.markdown(f"**Chunk Size:** {st.session_state.chunk_size} words")
    st.markdown(f"**Overlap:** {st.session_state.chunk_overlap} words")
    st.markdown(f"**Top K:** {st.session_state.top_k}")
    st.markdown(f"**Similarity Threshold:** {config.SIMILARITY_THRESHOLD}")

# Main content
st.markdown('<p class="main-header">🤖 AutoGrounded Chatbot</p>', unsafe_allow_html=True)
st.markdown("*Self-Verifying, Document-Aware AI Assistant with Enhanced RAG*")

# Show welcome message if no documents
if st.session_state.vector_store.get_document_count() == 0:
    st.info("👋 Welcome! Upload documents in the sidebar to get started. The chatbot will answer questions based on your documents.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display metadata for assistant messages
        if message["role"] == "assistant" and "metadata" in message:
            metadata = message["metadata"]
            
            # Confidence score
            confidence = metadata.get("confidence", 0)
            if confidence >= 80:
                confidence_class = "confidence-high"
            elif confidence >= 50:
                confidence_class = "confidence-medium"
            else:
                confidence_class = "confidence-low"
            
            st.markdown(f'<p class="{confidence_class}">Confidence: {confidence}%</p>', unsafe_allow_html=True)
            
            # Verification details
            if metadata.get("verification"):
                with st.expander("🔍 Verification Details"):
                    st.json(metadata["verification"])
            
            # Source citations
            if metadata.get("sources"):
                with st.expander("📚 Source Citations"):
                    for i, source in enumerate(metadata["sources"], 1):
                        st.markdown(f"**Source {i}:** {source['document']} (Score: {source.get('final_score', source.get('similarity_score', 0)):.2f})")
                        st.markdown(f'<div class="source-citation">{source["text"][:300]}...</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask me anything about your documents..."):
    # Check if RAG pipeline is initialized
    if st.session_state.rag_pipeline is None:
        st.error("⚠️ Please upload and process documents first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response from RAG pipeline with current top_k setting
                    response_data = st.session_state.rag_pipeline.generate_response(
                        prompt,
                        top_k=st.session_state.top_k
                    )
                    
                    # Verify response
                    verification = st.session_state.verifier.verify_response(
                        response_data["response"],
                        response_data["sources"]
                    )
                    
                    # Display response
                    st.markdown(response_data["response"])
                    
                    # Display confidence
                    confidence = verification["confidence"]
                    if confidence >= 80:
                        confidence_class = "confidence-high"
                    elif confidence >= 50:
                        confidence_class = "confidence-medium"
                    else:
                        confidence_class = "confidence-low"
                    
                    st.markdown(f'<p class="{confidence_class}">Confidence: {confidence}%</p>', unsafe_allow_html=True)
                    
                    # Verification details
                    with st.expander("🔍 Verification Details"):
                        st.json(verification)
                    
                    # Source citations
                    with st.expander("📚 Source Citations"):
                        for i, source in enumerate(response_data["sources"], 1):
                            st.markdown(f"**Source {i}:** {source['document']} (Score: {source.get('final_score', source.get('similarity_score', 0)):.2f})")
                            st.markdown(f'<div class="source-citation">{source["text"][:300]}...</div>', unsafe_allow_html=True)
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_data["response"],
                        "metadata": {
                            "confidence": confidence,
                            "verification": verification,
                            "sources": response_data["sources"]
                        }
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Make sure Ollama is running with TinyLlama installed: `ollama run tinyllama`")
