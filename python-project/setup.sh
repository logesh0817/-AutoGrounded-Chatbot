#!/bin/bash

echo "🚀 Setting up AutoGrounded Chatbot..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed."
    echo "📥 Please install Ollama from: https://ollama.ai/download"
    echo "   - Linux/Mac: curl https://ollama.ai/install.sh | sh"
    echo "   - Or visit the website for other platforms"
    exit 1
fi

echo "✅ Ollama found: $(ollama --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p documents
mkdir -p vector_db

# Pull TinyLlama model
echo "🤖 Pulling TinyLlama model (this may take a few minutes)..."
ollama pull tinyllama

echo ""
echo "✅ Setup complete!"
echo ""
echo "📖 To run the application:"
echo "   1. Make sure Ollama is running: ollama serve"
echo "   2. Activate the virtual environment: source venv/bin/activate"
echo "   3. Run the app: streamlit run app.py"
echo ""
echo "🌐 The app will open in your browser at http://localhost:8501"
