import ollama
from typing import List, Dict, Any
import config

class RAGPipeline:
    """RAG Pipeline using TinyLlama via Ollama"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.conversation_history = []
        
    def generate_response(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Generate response using RAG pipeline
        
        Args:
            query: User query
            top_k: Number of sources to retrieve (uses config default if None)
            
        Returns:
            Dictionary containing response and sources
        """
        # Retrieve relevant documents
        k = top_k or config.DEFAULT_TOP_K_RESULTS
        sources = self.vector_store.search(query, k=k)
        
        if not sources:
            return {
                "response": "I don't have enough information in the loaded documents to answer this question. Please upload relevant documents.",
                "sources": []
            }
        
        # Build context from sources
        context = self._build_context(sources)
        
        # Create prompt
        prompt = self._create_prompt(query, context)
        
        # Generate response using Ollama
        try:
            response = ollama.generate(
                model=config.MODEL_NAME,
                prompt=prompt,
                options={
                    'temperature': config.TEMPERATURE,
                    'top_p': config.TOP_P,
                    'top_k': config.TOP_K_SAMPLING,
                }
            )
            
            answer = response['response'].strip()
            
            return {
                "response": answer,
                "sources": sources
            }
            
        except Exception as e:
            raise Exception(f"Error generating response with Ollama: {str(e)}. Make sure Ollama is running and TinyLlama is installed.")
    
    def _build_context(self, sources: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved sources"""
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(f"[Document {i}]: {source['text']}")
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create optimized prompt for the language model"""
        prompt = f"""You are a precise AI assistant. Answer based ONLY on the provided documents.

RULES:
1. Use ONLY information from documents below
2. Cite document numbers [Doc 1], [Doc 2], etc.
3. Be concise and direct
4. If insufficient info, state clearly
5. No speculation or external knowledge

DOCUMENTS:
{context}

QUESTION: {query}

ANSWER:"""
        
        return prompt
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
