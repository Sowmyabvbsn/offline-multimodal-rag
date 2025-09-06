# main.py - Complete Offline AI Agent (Optimized for phi3:mini)
import os
import sys
from pathlib import Path
import sqlite3
from datetime import datetime
import threading

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from document_processor import DocumentProcessor
    from audio_processor import AudioProcessor
    from image_processor import ImageProcessor
    from vector_store import VectorStore
    from qa_chain import QAChain
    from ui import launch_gradio_interface, launch_terminal_interface
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all src files are created and contain the code")
    sys.exit(1)

class ThreadSafeDatabase:
    """Thread-safe database wrapper for SQLite"""
    def __init__(self, db_path):
        self.db_path = db_path
        self.local = threading.local()
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                query TEXT,
                response TEXT,
                sources TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def get_connection(self):
        """Get a thread-local database connection"""
        if not hasattr(self.local, 'connection'):
            self.local.connection = sqlite3.connect(
                self.db_path, 
                check_same_thread=False,
                timeout=20
            )
        return self.local.connection
    
    def log_conversation(self, query, response, sources):
        """Log a conversation to the database"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            conn.execute("""
                INSERT INTO conversations (timestamp, query, response, sources)
                VALUES (?, ?, ?, ?)
            """, (datetime.now().isoformat(), query, response, str(sources)))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Warning: Could not log conversation: {e}")

class OfflineAIAgent:
    def __init__(self, data_dir="data", models_dir="models", model_name="phi3:mini"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.model_name = model_name  # Default to phi3:mini for speed
        self.setup_directories()
        self.setup_database()
        
        print("🚀 Offline AI Agent initialized!")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"🤖 Models directory: {self.models_dir}")
        print(f"⚡ Using model: {self.model_name}")
        
        # Initialize processors (lazy loading)
        self.doc_processor = None
        self.audio_processor = None
        self.image_processor = None
        self.vector_store = None
        self.qa_chain = None
        
    def setup_directories(self):
        """Create necessary directories"""
        for subdir in ['pdfs', 'images', 'audio', 'processed']:
            (self.data_dir / subdir).mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
    
    def setup_database(self):
        """Initialize thread-safe database for logging"""
        self.db = ThreadSafeDatabase("logs/chat_history.db")
        print("💾 Database initialized")
    
    def _initialize_components(self):
        """Lazy initialization of components"""
        if self.doc_processor is None:
            self.doc_processor = DocumentProcessor()
        if self.audio_processor is None:
            self.audio_processor = AudioProcessor()
        if self.image_processor is None:
            self.image_processor = ImageProcessor()
        if self.vector_store is None:
            self.vector_store = VectorStore(str(self.models_dir / "faiss_index"))
        if self.qa_chain is None:
            self.qa_chain = QAChain(self.vector_store, model_name=self.model_name)
    
    def process_documents(self):
        """Process all documents in data directories"""
        self._initialize_components()
        
        print("\n🔄 Processing documents...")
        all_chunks = []
        
        # Process PDFs
        pdf_dir = self.data_dir / "pdfs"
        if pdf_dir.exists():
            pdf_chunks = self.doc_processor.process_all_pdfs(str(pdf_dir))
            all_chunks.extend(pdf_chunks)
            if pdf_chunks:
                self.vector_store.add_documents(pdf_chunks, "pdf")
        
        # Process Images
        img_dir = self.data_dir / "images"
        if img_dir.exists():
            img_chunks = self.image_processor.process_all_images(str(img_dir))
            all_chunks.extend(img_chunks)
            if img_chunks:
                self.vector_store.add_documents(img_chunks, "image")
        
        # Process Audio
        audio_dir = self.data_dir / "audio"
        if audio_dir.exists():
            audio_chunks = self.audio_processor.process_all_audio(str(audio_dir))
            all_chunks.extend(audio_chunks)
            if audio_chunks:
                self.vector_store.add_documents(audio_chunks, "audio")
        
        # Save the vector store
        if all_chunks:
            self.vector_store.save()
            print(f"✅ Processing complete! Created {len(all_chunks)} chunks total")
        else:
            print("⚠️  No documents found to process. Add files to data/ folders")
    
    def ask_question(self, question, quick_mode=False):
        """Ask a question with optional quick mode for faster responses"""
        self._initialize_components()
        
        try:
            if quick_mode and hasattr(self.qa_chain, 'quick_ask'):
                # Use ultra-fast mode for simple questions
                response = self.qa_chain.quick_ask(question)
                sources = ["Quick response mode - limited context"]
            else:
                # Standard mode
                response, sources, metadata = self.qa_chain.ask(question)
            
            # Log the conversation (thread-safe)
            self.db.log_conversation(question, response, sources)
            
            return response, sources
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(error_msg)
            return error_msg, []
    
    def test_dependencies(self):
        """Test if all required packages are installed"""
        print("\n🔍 Testing dependencies...")
        
        required_packages = [
            ("torch", "PyTorch for ML operations"),
            ("requests", "HTTP requests for Ollama"),
            ("transformers", "HuggingFace transformers"),
            ("sentence_transformers", "Sentence embeddings"),
            ("langchain", "LangChain framework"),
            ("faiss", "Vector similarity search"),
            ("fitz", "PyMuPDF for PDF processing"),
            ("pytesseract", "OCR for images"),
            ("whisper", "Speech-to-text"),
            ("gradio", "Web interface"),
        ]
        
        missing_packages = []
        
        for package, description in required_packages:
            try:
                if package == "fitz":
                    import fitz
                else:
                    __import__(package)
                print(f"✅ {package} - {description}")
            except ImportError:
                print(f"❌ {package} - MISSING - {description}")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            return False
        else:
            print("\n🎉 All dependencies installed!")
            return True
    
    def test_ollama(self):
        """Test Ollama connection and phi3:mini availability"""
        print(f"\n🤖 Testing Ollama connection and {self.model_name} model...")
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                print("✅ Ollama is running")
                print(f"📦 Available models: {available_models}")
                
                # Check if our target model is available
                if any(self.model_name in model for model in available_models):
                    print(f"✅ {self.model_name} is available")
                    return True
                else:
                    print(f"⚠️  {self.model_name} not found")
                    print(f"💡 Run: ollama pull {self.model_name}")
                    return False
            else:
                print("❌ Ollama server error")
                return False
        except Exception as e:
            print("❌ Ollama not running")
            print("💡 Start Ollama: ollama serve")
            print(f"💡 Install model: ollama pull {self.model_name}")
            return False
    
    def get_stats(self):
        """Get system statistics"""
        try:
            # Get file counts
            pdf_count = len(list((self.data_dir / "pdfs").glob("*.pdf"))) if (self.data_dir / "pdfs").exists() else 0
            img_count = len([f for f in (self.data_dir / "images").glob("*") 
                           if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp']]) if (self.data_dir / "images").exists() else 0
            audio_count = len([f for f in (self.data_dir / "audio").glob("*") 
                             if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg']]) if (self.data_dir / "audio").exists() else 0
            
            # Get vector store stats
            if self.vector_store:
                vs_stats = self.vector_store.get_stats()
                doc_chunks = vs_stats.get('total_documents', 0)
                sources = len(vs_stats.get('sources', []))
            else:
                doc_chunks = 0
                sources = 0
            
            return {
                'pdf_count': pdf_count,
                'img_count': img_count,
                'audio_count': audio_count,
                'doc_chunks': doc_chunks,
                'sources': sources,
                'model': self.model_name
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {
                'pdf_count': 0,
                'img_count': 0,
                'audio_count': 0,
                'doc_chunks': 0,
                'sources': 0,
                'model': self.model_name
            }
    
    def switch_model(self, new_model="mistral"):
        """Switch to a different model"""
        old_model = self.model_name
        self.model_name = new_model
        
        # Reset QA chain to use new model
        if self.qa_chain:
            self.qa_chain.set_model(new_model)
        
        print(f"🔄 Switched from {old_model} to {new_model}")
        return self.test_ollama()
    
    def launch_interface(self, interface_type="gradio"):
        """Launch the user interface"""
        self._initialize_components()
        
        if interface_type == "gradio":
            print("🌐 Starting Gradio web interface...")
            print(f"⚡ Optimized for {self.model_name}")
            demo = launch_gradio_interface(self)
            demo.launch(
                server_name="127.0.0.1", 
                server_port=7860, 
                share=False,
                show_error=True,
                quiet=False
            )
        elif interface_type == "terminal":
            launch_terminal_interface(self)

def main():
    print("🚀 Starting Offline AI Agent...")
    print("⚡ Optimized for phi3:mini (fast responses)")
    print("=" * 60)
    
    # Initialize agent with phi3:mini as default
    agent = OfflineAIAgent(model_name="phi3:mini")
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            print("\n📊 Testing system...")
            deps_ok = agent.test_dependencies()
            ollama_ok = agent.test_ollama()
            print(f"\nTest Results:")
            print(f"Dependencies: {'✅ OK' if deps_ok else '❌ MISSING'}")
            print(f"Ollama + phi3:mini: {'✅ OK' if ollama_ok else '❌ NOT READY'}")
            
            if not ollama_ok:
                print(f"\n💡 To install phi3:mini, run:")
                print(f"   ollama pull phi3:mini")
            
        elif command == "process":
            agent.process_documents()
            
        elif command == "gradio":
            agent.launch_interface("gradio")
            
        elif command == "terminal":
            agent.launch_interface("terminal")
            
        elif command == "chat":
            agent.launch_interface("terminal")
            
        elif command == "stats":
            stats = agent.get_stats()
            print(f"\n📊 System Statistics:")
            print(f"🤖 Model: {stats['model']}")
            print(f"📄 PDFs: {stats['pdf_count']}")
            print(f"🖼️  Images: {stats['img_count']}")
            print(f"🎵 Audio files: {stats['audio_count']}")
            print(f"📚 Document chunks: {stats['doc_chunks']}")
            print(f"📁 Unique sources: {stats['sources']}")
            
        elif command == "switch-mistral":
            print("🔄 Switching to Mistral model...")
            success = agent.switch_model("mistral")
            if success:
                print("✅ Switched to Mistral successfully")
            else:
                print("❌ Failed to switch to Mistral")
                
        elif command == "switch-phi3":
            print("🔄 Switching to phi3:mini model...")
            success = agent.switch_model("phi3:mini")
            if success:
                print("✅ Switched to phi3:mini successfully")
            else:
                print("❌ Failed to switch to phi3:mini")
            
    else:
        print("\n🎯 Usage:")
        print("python main.py test           - Test if everything is set up")
        print("python main.py process        - Process documents in data/ folders")
        print("python main.py gradio         - Start web interface (recommended)")
        print("python main.py terminal       - Start terminal chat")
        print("python main.py chat           - Start terminal chat")
        print("python main.py stats          - Show system statistics")
        print("python main.py switch-mistral - Switch to Mistral model")
        print("python main.py switch-phi3    - Switch to phi3:mini model")
        print("\n⚡ Default model: phi3:mini (optimized for speed)")
        print("💡 Switch models anytime with switch commands")

if __name__ == "__main__":
    main()