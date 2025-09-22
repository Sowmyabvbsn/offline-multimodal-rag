# src/document_processor.py
import fitz  # PyMuPDF
from typing import List, Dict
import os
import base64
from PIL import Image
import io

class DocumentProcessor:
    def __init__(self, chunk_size=1000, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.images_dir = "data/extracted_images"
        os.makedirs(self.images_dir, exist_ok=True)
        print("📄 Document processor initialized")
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            page_images = {}
            page_texts = {}
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                page_texts[page_num + 1] = page_text
                
                # Extract images from page
                image_list = page.get_images()
                if image_list:
                    page_images[page_num + 1] = []
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)
                            if pix.n - pix.alpha < 4:  # GRAY or RGB
                                img_data = pix.tobytes("png")
                                img_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page{page_num+1}_img{img_index+1}.png"
                                img_path = os.path.join(self.images_dir, img_filename)
                                
                                with open(img_path, "wb") as img_file:
                                    img_file.write(img_data)
                                
                                page_images[page_num + 1].append({
                                    'path': img_path,
                                    'filename': img_filename,
                                    'size': (pix.width, pix.height)
                                })
                            pix = None
                        except Exception as e:
                            print(f"⚠️ Could not extract image {img_index} from page {page_num + 1}: {e}")
                # Add page metadata
                full_text += f"[Page {page_num + 1}]\n{page_text}\n"
            
            doc.close()
            print(f"✅ Extracted text from {os.path.basename(pdf_path)}")
            if page_images:
                print(f"🖼️ Extracted {sum(len(imgs) for imgs in page_images.values())} images")
            
            return {
                'text': full_text,
                'page_texts': page_texts,
                'page_images': page_images
            }
            
        except Exception as e:
            print(f"❌ Error processing PDF {pdf_path}: {e}")
            return {'text': '', 'page_texts': {}, 'page_images': {}}
    
    def chunk_text(self, extraction_result: Dict, source_info: str) -> List[Dict]:
        """Split text into chunks with metadata"""
        text = extraction_result['text']
        page_texts = extraction_result['page_texts']
        page_images = extraction_result['page_images']
        
        if not text.strip():
            return []
            
        chunks = []
        
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size]
            
            # Extract page number if present
            page_num = self._extract_page_number(chunk_text)
            
            # Get original page text for better citation
            original_page_text = page_texts.get(page_num, "")
            
            # Get images for this page
            chunk_images = page_images.get(page_num, [])
            
            chunk = {
                'text': chunk_text.strip(),
                'original_page_text': original_page_text,
                'source': source_info,
                'chunk_id': len(chunks),
                'page': page_num,
                'type': 'pdf',
                'images': chunk_images,
                'has_images': len(chunk_images) > 0
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
        extraction_result = self.extract_text_from_pdf(pdf_path)
        if extraction_result['text']:
            return self.chunk_text(extraction_result, pdf_path)
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