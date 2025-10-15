# 🤖 AutoGrounded Chatbot

A Self-Verifying, Document-Aware AI Assistant using TinyLlama via Ollama and free resources only.

## 🌟 Features

- **100% Free & Local**: Uses TinyLlama via Ollama - no API keys required
- **Retrieval-Augmented Generation (RAG)**: Answers based on your uploaded documents
- **Self-Verification**: Cross-checks responses against source documents
- **Confidence Scoring**: Shows reliability scores (0-100%) for each response
- **Source Citations**: Displays which documents support each answer
- **Document Support**: Upload PDF, TXT, and Markdown files
- **Modern UI**: Clean Streamlit interface with real-time chat
- **Persistent Storage**: FAISS vector database saves your documents

## 🏗️ Architecture

```
┌─────────────────┐
│  User Interface │  (Streamlit)
└────────┬────────┘
         │
┌────────▼────────┐
│  RAG Pipeline   │  (TinyLlama via Ollama)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────┐
│Vector │ │Document │
│Store  │ │Processor│
│(FAISS)│ │         │
└───┬───┘ └──┬──────┘
    │        │
┌───▼────────▼───┐
│  Verification  │
│     Module     │
└────────────────┘
```

## 📋 Prerequisites

- **Python 3.8+**
- **Ollama** (for running TinyLlama locally)
- **4GB+ RAM** recommended
- **2GB+ disk space** for models and documents

## 🚀 Installation

### Step 1: Install Ollama

**Linux/Mac:**
```bash
curl https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [https://ollama.ai/download](https://ollama.ai/download)

### Step 2: Clone and Setup Project

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup (installs dependencies and downloads TinyLlama)
./setup.sh
```

The setup script will:
- Create a Python virtual environment
- Install all required packages
- Download TinyLlama model (~600MB)
- Create necessary directories

## 🎮 Usage

### 1. Start Ollama Server

In one terminal:
```bash
ollama serve
```

### 2. Run the Application

In another terminal:
```bash
# Activate virtual environment
source venv/bin/activate

# Run Streamlit app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

### 3. Upload Documents

1. Click **"Browse files"** in the sidebar
2. Select PDF, TXT, or MD files
3. Click **"Process Documents"**
4. Wait for processing to complete

### 4. Chat with Your Documents

- Type questions in the chat input
- View responses with confidence scores
- Check source citations to verify answers
- Expand verification details for transparency

## 📁 Project Structure

```
autogrounded_chatbot/
├── app.py                 # Main Streamlit application
├── rag_pipeline.py        # RAG implementation with TinyLlama
├── document_processor.py  # Document loading and chunking
├── vector_store.py        # FAISS vector database management
├── verification.py        # Self-verification logic
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── setup.sh              # Installation script
├── README.md             # This file
├── documents/            # Uploaded documents storage
└── vector_db/           # FAISS index storage
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Model Settings
MODEL_NAME = "tinyllama"
TEMPERATURE = 0.7         # Creativity (0.0-1.0)
TOP_K_RESULTS = 5         # Retrieved chunks per query

# Document Processing
CHUNK_SIZE = 300          # Words per chunk
CHUNK_OVERLAP = 50        # Overlap between chunks

# Verification
MIN_CONFIDENCE_THRESHOLD = 50  # Warning threshold
```

## 🎯 How It Works

### 1. Document Processing
- Documents are split into overlapping chunks
- Each chunk is embedded using Sentence Transformers
- Embeddings are stored in FAISS vector database

### 2. Query Processing
- User query is embedded
- Top-K most similar chunks are retrieved
- Context is built from retrieved chunks

### 3. Response Generation
- TinyLlama generates response using context
- Response is based only on provided documents
- Model explicitly cites document numbers

### 4. Self-Verification
- Response is split into individual claims
- Each claim is checked against source documents
- Confidence score is calculated (0-100%)
- Source citations are provided

## 📊 Confidence Score Interpretation

- **🟢 80-100%**: High confidence - strong document support
- **🟡 50-79%**: Medium confidence - partial support
- **🔴 0-49%**: Low confidence - weak or no support

## 🔧 Troubleshooting

### Ollama Connection Error
```bash
# Make sure Ollama is running
ollama serve

# Test if TinyLlama is available
ollama list
```

### Model Not Found
```bash
# Pull TinyLlama model
ollama pull tinyllama
```

### Memory Issues
- Reduce `TOP_K_RESULTS` in config.py
- Process documents in smaller batches
- Close other applications

### Slow Processing
- TinyLlama is optimized for speed but smaller
- Consider using faster hardware
- Reduce document chunk size

## 🧪 Testing

Test with sample documents:

1. Create a test document `test.txt`:
```
The capital of France is Paris.
Python is a programming language created by Guido van Rossum.
Water boils at 100 degrees Celsius at sea level.
```

2. Upload and process it
3. Ask: "What is the capital of France?"
4. Verify the confidence score and citations

## 📚 Dependencies

- **streamlit**: Web interface
- **ollama**: TinyLlama integration
- **faiss-cpu**: Vector similarity search
- **sentence-transformers**: Text embeddings
- **PyPDF2**: PDF processing
- **numpy**: Numerical operations

## 🔒 Privacy & Security

- **100% Local**: All processing happens on your machine
- **No API Calls**: No data sent to external servers
- **No Tracking**: No telemetry or analytics
- **Your Data**: Documents stay on your computer

## 🎓 Academic Use

This project demonstrates:
- Retrieval-Augmented Generation (RAG)
- Vector similarity search with FAISS
- Self-verification techniques
- Local LLM deployment
- Document chunking strategies
- Confidence scoring algorithms

## 📝 Citation

If using for academic work, consider citing:
- TinyLlama: https://github.com/jzhang38/TinyLlama
- Ollama: https://ollama.ai
- FAISS: https://github.com/facebookresearch/faiss
- Sentence Transformers: https://www.sbert.net

## 🤝 Contributing

This is an academic project template. Feel free to:
- Modify configurations
- Add new features
- Improve verification logic
- Extend document support

## ⚠️ Limitations

- **Model Size**: TinyLlama (1.1B) is smaller than GPT-3/4
- **Context Length**: Limited context window
- **Speed**: Local processing may be slower than cloud APIs
- **Accuracy**: Self-verification helps but isn't perfect

## 🚀 Future Enhancements

Potential improvements:
- [ ] Support for images in PDFs
- [ ] Multi-document conversation threads
- [ ] Export chat history
- [ ] Advanced query reformulation
- [ ] Multiple language support
- [ ] Custom embedding models

## 📞 Support

For issues:
1. Check Ollama is running: `ollama serve`
2. Verify TinyLlama is installed: `ollama list`
3. Check Python version: `python3 --version`
4. Reinstall dependencies: `pip install -r requirements.txt`

## 📄 License

This project is provided as-is for academic and educational purposes.

## 🙏 Acknowledgments

- TinyLlama team for the efficient language model
- Ollama for easy local LLM deployment
- Meta AI for FAISS vector database
- Sentence Transformers for embeddings
- Streamlit for the web framework

---

**Built with ❤️ for academic research and learning**

🌟 **Remember**: This is a 100% free, local, and privacy-respecting AI assistant!
