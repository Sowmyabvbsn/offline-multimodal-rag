# Offline AI Agent

A fully offline AI-powered document processing and question-answering system that works with PDFs, images, and audio files without requiring cloud services.

## Features

- **Multimodal Processing**: PDFs, images (OCR), and audio files (speech-to-text)
- **Fully Offline**: No cloud dependencies, runs entirely on your local machine
- **Local LLM Integration**: Uses Ollama for local language model inference
- **Vector Search**: FAISS-powered semantic search across your documents
- **Web Interface**: User-friendly Gradio-based web UI
- **Terminal Interface**: Command-line chat interface
- **Real-time Processing**: Upload and process files through the web interface

## Tech Stack

- **Language Models**: Ollama (Mistral, Phi3:mini, LLaMA)
- **Embeddings**: Sentence Transformers
- **Vector Database**: FAISS
- **Document Processing**: PyMuPDF
- **OCR**: Tesseract
- **Speech-to-Text**: OpenAI Whisper
- **Web Interface**: Gradio
- **Framework**: LangChain

## Quick Start

### Prerequisites

- Python 3.8+
- 8GB+ RAM (16GB recommended)
- 10GB+ free disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/offline-ai-agent.git
cd offline-ai-agent
```

2. **Create virtual environment**
```bash
python -m venv offline_ai_env
source offline_ai_env/bin/activate  # On Windows: offline_ai_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Tesseract OCR**
- **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`
- **Ubuntu/Debian**: `sudo apt install tesseract-ocr`

5. **Install Ollama and models**
```bash
# Install Ollama from https://ollama.ai
ollama pull phi3:mini  # Fast model (recommended)
ollama pull mistral    # Alternative model
```

### Usage

1. **Test your setup**
```bash
python main.py test
```

2. **Start the web interface**
```bash
python main.py gradio
```

3. **Upload files and start chatting!**
   - Open http://127.0.0.1:7860
   - Upload PDFs, images, or audio files
   - Click "Process Documents"
   - Ask questions about your content

## Alternative Interfaces

### Terminal Interface
```bash
python main.py terminal
```

### Direct Processing
```bash
# Add files to data/ folders, then:
python main.py process
```

## Project Structure

```
offline-ai-agent/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/                   # Source code modules
│   ├── __init__.py
│   ├── document_processor.py  # PDF processing
│   ├── audio_processor.py     # Audio transcription
│   ├── image_processor.py     # OCR processing
│   ├── vector_store.py        # Vector database
│   ├── qa_chain.py           # Question-answering
│   └── ui.py                 # Web interface
├── data/                  # Input files
│   ├── pdfs/             # Place PDF files here
│   ├── images/           # Place images here
│   └── audio/            # Place audio files here
├── models/               # Vector indices and models
└── logs/                 # Application logs
```

## Supported File Formats

- **Documents**: PDF
- **Images**: PNG, JPG, JPEG, BMP, TIFF, GIF, WEBP (with OCR)
- **Audio**: WAV, MP3, M4A, FLAC, AAC, OGG (with speech-to-text)

## Performance Tips

- Use `phi3:mini` model for faster responses
- Limit PDF size to < 100MB for optimal performance
- For large document collections, allow extra processing time
- 16GB RAM recommended for processing large files

## Commands

```bash
python main.py test           # Test system setup
python main.py gradio         # Start web interface
python main.py terminal       # Start terminal chat
python main.py process        # Process files in data/ folders
python main.py stats          # Show system statistics
python main.py switch-phi3    # Switch to phi3:mini model
python main.py switch-mistral # Switch to mistral model
```

## Configuration

### Switching Models
```bash
python main.py switch-phi3    # Faster responses
python main.py switch-mistral # Higher quality responses
```

### Custom Model
Edit `main.py` and change the default model:
```python
agent = OfflineAIAgent(model_name="your-model-name")
```

## Troubleshooting

### Common Issues

1. **Timeout errors**: Switch to phi3:mini model for faster responses
2. **Tesseract not found**: Install Tesseract OCR and add to PATH
3. **Ollama connection failed**: Start Ollama service and pull models
4. **Out of memory**: Use smaller models or reduce batch sizes

### Model Performance

- **phi3:mini**: Fast responses, good for most tasks
- **mistral**: Higher quality, slower responses
- **llama3**: Best quality, requires more resources

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ollama](https://ollama.ai) for local LLM serving
- [LangChain](https://langchain.com) for RAG framework
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Gradio](https://gradio.app) for web interface
- [Whisper](https://openai.com/research/whisper) for speech recognition

## Star History

If this project helped you, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/offline-ai-agent&type=Date)](https://star-history.com/#yourusername/offline-ai-agent&Date)