# src/document_processor.py
import fitz  # PyMuPDF
from typing import List, Dict
import os

class DocumentProcessor:
    def __init__(self, chunk_size=1000, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        print("📄 Document processor initialized")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                # Add page metadata
                full_text += f"[Page {page_num + 1}]\n{page_text}\n"
            
            doc.close()
            print(f"✅ Extracted text from {os.path.basename(pdf_path)}")
            return full_text
            
        except Exception as e:
            print(f"❌ Error processing PDF {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, source_info: str) -> List[Dict]:
        """Split text into chunks with metadata"""
        if not text.strip():
            return []
            
        chunks = []
        
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size]
            
            # Extract page number if present
            page_num = self._extract_page_number(chunk_text)
            
            chunk = {
                'text': chunk_text.strip(),
                'source': source_info,
                'chunk_id': len(chunks),
                'page': page_num,
                'type': 'pdf'
            }
            chunks.append(chunk)
        
        print(f"📝 Created {len(chunks)} chunks from {os.path.basename(source_info)}")
        return chunks
    
    def _extract_page_number(self, text: str) -> int:
        """Extract page number from text chunk"""
        lines = text.split('\n')
        for line in lines:
            if line.startswith('[Page ') and ']' in line:
                try:
                    return int(line.split('[Page ')[1].split(']')[0])
                except:
                    pass
        return 1
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Process a PDF file and return chunks"""
        print(f"📖 Processing PDF: {os.path.basename(pdf_path)}")
        text = self.extract_text_from_pdf(pdf_path)
        if text:
            return self.chunk_text(text, pdf_path)
        return []
    
    def process_all_pdfs(self, pdf_directory: str) -> List[Dict]:
        """Process all PDFs in a directory"""
        all_chunks = []
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"⚠️  No PDF files found in {pdf_directory}")
            return all_chunks
        
        print(f"📚 Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            chunks = self.process_pdf(pdf_path)
            all_chunks.extend(chunks)
        
        print(f"✅ Processed {len(pdf_files)} PDFs, created {len(all_chunks)} chunks total")
        return all_chunks

# Test the document processor
if __name__ == "__main__":
    print("🧪 Testing Document Processor...")
    
    processor = DocumentProcessor()
    
    # Test with data/pdfs directory
    pdf_dir = "../data/pdfs"
    if os.path.exists(pdf_dir):
        chunks = processor.process_all_pdfs(pdf_dir)
        print(f"📊 Result: {len(chunks)} chunks created")
        
        if chunks:
            print("\n📝 Sample chunk:")
            print(f"Source: {chunks[0]['source']}")
            print(f"Type: {chunks[0]['type']}")
            print(f"Page: {chunks[0]['page']}")
            print(f"Text preview: {chunks[0]['text'][:200]}...")
    else:
        print(f"📁 Directory {pdf_dir} not found. Add some PDFs to test!")