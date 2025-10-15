import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import pickle
import os
import config

class VectorStore:
    """FAISS vector store for document embeddings"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.index = None
        self.documents = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        # Load existing index if available
        self._load_index()
    
    def add_documents(self, chunks: List[Dict[str, Any]], document_name: str):
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of document chunks
            document_name: Name of the source document
        """
        if not chunks:
            return
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Initialize or update FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store document metadata
        for i, chunk in enumerate(chunks):
            self.documents.append({
                'text': chunk['text'],
                'document': document_name,
                'chunk_index': chunk['metadata']['chunk_index'],
                'embedding_index': len(self.documents)
            })
        
        # Save index
        self._save_index()
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Enhanced search with similarity filtering and better scoring
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant document chunks
        """
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search with more candidates for reranking
        search_k = min(k * 3, len(self.documents))  # Get 3x results for better filtering
        distances, indices = self.index.search(query_embedding, search_k)
        
        # Retrieve documents with better similarity scoring
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                # Better similarity calculation using exponential decay
                similarity = float(np.exp(-distance / 10))  # Exponential similarity
                doc['similarity_score'] = similarity
                doc['distance'] = float(distance)
                
                # Only include results above threshold
                if similarity >= config.SIMILARITY_THRESHOLD:
                    results.append(doc)
        
        # Sort by similarity and keyword overlap
        results = self._rerank_results(query, results)
        
        # Return top k results
        return results[:k]
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results using keyword matching for better relevance
        
        Args:
            query: Search query
            results: Initial search results
            
        Returns:
            Reranked results
        """
        query_words = set(query.lower().split())
        
        for result in results:
            text_words = set(result['text'].lower().split())
            keyword_overlap = len(query_words & text_words) / len(query_words) if query_words else 0
            
            # Combine semantic similarity with keyword matching
            result['final_score'] = (result['similarity_score'] * 0.7) + (keyword_overlap * 0.3)
        
        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results
    
    def get_document_count(self) -> int:
        """Get total number of document chunks"""
        return len(self.documents)
    
    def _save_index(self):
        """Save FAISS index and documents to disk"""
        if self.index is not None:
            # Save FAISS index
            faiss.write_index(self.index, os.path.join(config.VECTOR_DB_DIR, 'faiss.index'))
            
            # Save documents metadata
            with open(os.path.join(config.VECTOR_DB_DIR, 'documents.pkl'), 'wb') as f:
                pickle.dump(self.documents, f)
    
    def _load_index(self):
        """Load FAISS index and documents from disk"""
        index_path = os.path.join(config.VECTOR_DB_DIR, 'faiss.index')
        docs_path = os.path.join(config.VECTOR_DB_DIR, 'documents.pkl')
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
            except Exception as e:
                print(f"Error loading index: {e}")
                self.index = None
                self.documents = []
