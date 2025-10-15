"""Configuration settings for AutoGrounded Chatbot"""

# Ollama Settings
MODEL_NAME = "tinyllama"  # TinyLlama model via Ollama
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K_SAMPLING = 40

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence Transformers model

# Document Processing (Default values - can be overridden)
DEFAULT_CHUNK_SIZE = 300  # Words per chunk
DEFAULT_CHUNK_OVERLAP = 50  # Overlap between chunks
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 1000
MIN_OVERLAP = 20
MAX_OVERLAP = 200

# RAG Settings
DEFAULT_TOP_K_RESULTS = 5  # Number of relevant chunks to retrieve
MIN_TOP_K = 1
MAX_TOP_K = 20

# Search Settings
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score to include results
RERANK_TOP_N = 10  # Number of results to retrieve before reranking

# Storage Directories
DOCUMENTS_DIR = "documents"
VECTOR_DB_DIR = "vector_db"

# Verification Settings
MIN_CONFIDENCE_THRESHOLD = 50  # Minimum confidence score to show warnings
