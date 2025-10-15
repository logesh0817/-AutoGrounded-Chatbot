import os
from typing import List, Dict, Any
from pathlib import Path
import config

class DocumentProcessor:
    """Process and chunk documents"""
    
    def __init__(self, chunk_size=None, chunk_overlap=None):
        self.chunk_size = chunk_size or config.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.DEFAULT_CHUNK_OVERLAP
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a document and return chunks
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of document chunks with metadata
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            text = self._process_pdf(file_path)
        elif file_extension in ['.txt', '.md']:
            text = self._process_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Chunk the text
        chunks = self._chunk_text(text, file_path)
        
        return chunks
    
    def _process_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            raise ImportError("PyPDF2 is required to process PDF files. Install it with: pip install PyPDF2")
    
    def _process_text(self, file_path: str) -> str:
        """Read text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _chunk_text(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Text to chunk
            source_file: Source file path
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        words = text.split()
        
        chunk_size = self.chunk_size
        overlap = self.chunk_overlap
        
        # Ensure we don't have zero or negative step
        step = max(1, chunk_size - overlap)
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'source': os.path.basename(source_file),
                        'chunk_index': len(chunks),
                        'word_count': len(chunk_words)
                    }
                })
        
        return chunks
