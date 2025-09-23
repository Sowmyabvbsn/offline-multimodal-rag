# src/document_processor.py
import fitz  # PyMuPDF
from typing import List, Dict, Optional
import os
from pathlib import Path
import hashlib
import base64
from io import BytesIO
from PIL import Image

class DocumentProcessor:
    def __init__(self):
        self.supported_formats = ['.pdf']
        print("📄 Document processor initialized")
        print(f"   Supported formats: {', '.join(self.supported_formats)}")
        
        # Test if PyMuPDF is available
        self._test_pymupdf()
    
    def _test_pymupdf(self):
        """Test if PyMuPDF is available"""
        try:
            doc = fitz.open()  # Create empty document
            doc.close()
            print("✅ PyMuPDF (fitz) is available")
            return True
        except Exception as e:
            print(f"❌ PyMuPDF not found: {e}")
            print("💡 Install with: pip install PyMuPDF")
            return False
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """Extract text and metadata from PDF"""
        try:
            print(f"📖 Processing PDF: {os.path.basename(pdf_path)}")
            
            doc = fitz.open(pdf_path)
            
            # Get document metadata
            metadata = doc.metadata
            page_count = len(doc)
            
            print(f"   Pages: {page_count}")
            print(f"   Title: {metadata.get('title', 'N/A')}")
            print(f"   Author: {metadata.get('author', 'N/A')}")
            
            # Extract text from all pages
            full_text = ""
            pages_data = []
            total_images = 0
            
            for page_num in range(page_count):
                page = doc[page_num]
                # Try multiple text extraction methods
                page_text = page.get_text()
                
                # If no text found, try different extraction methods
                if not page_text.strip():
                    # Try extracting text blocks
                    text_blocks = page.get_text("blocks")
                    page_text = " ".join([block[4] for block in text_blocks if len(block) > 4])
                
                # If still no text, try extracting from text dict
                if not page_text.strip():
                    text_dict = page.get_text("dict")
                    page_text = self._extract_text_from_dict(text_dict)
                
                print(f"   Page {page_num + 1}: {len(page_text)} characters extracted")
                
                # Extract images from this page
                page_images = self._extract_images_from_page(page, pdf_path, page_num + 1)
                total_images += len(page_images)
                
                if len(page_images) > 0:
                    print(f"   Page {page_num + 1}: {len(page_images)} images found")
                
                pages_data.append({
                    'page_number': page_num + 1,
                    'text': page_text,
                    'char_count': len(page_text),
                    'images': page_images
                })
                
                full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            doc.close()
            
            print(f"✅ PDF processed: {len(full_text)} characters, {total_images} images")
            
            # Debug: Show sample of extracted text
            if full_text.strip():
                sample_text = full_text.strip()[:200]
                print(f"📄 Sample text: '{sample_text}...'")
            else:
                print("⚠️ No text was extracted from any page!")
            
            return {
                'text': full_text,
                'pages': pages_data,
                'metadata': metadata,
                'page_count': page_count,
                'char_count': len(full_text),
                'word_count': len(full_text.split()) if full_text else 0,
                'total_images': total_images
            }
            
        except Exception as e:
            print(f"❌ Error processing PDF {pdf_path}: {e}")
            return {
                'text': '',
                'pages': [],
                'metadata': {},
                'page_count': 0,
                'char_count': 0,
                'word_count': 0,
                'total_images': 0
            }
    
    def _extract_images_from_page(self, page, pdf_path: str, page_num: int) -> List[Dict]:
        """Extract images from a PDF page"""
        images = []
        
        # Create images directory if it doesn't exist
        images_dir = Path(pdf_path).parent / "extracted_images" / Path(pdf_path).stem
        images_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    # Skip if image is too small or has no data
                    if pix.width < 50 or pix.height < 50:
                        pix = None
                        continue
                    
                    # Convert to PIL Image for processing
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        img_filename = f"page_{page_num}_img_{img_index + 1}.png"
                        img_file_path = images_dir / img_filename
                        
                        # Save image to disk for display
                        with open(img_file_path, "wb") as img_file:
                            img_file.write(img_data)
                        
                        # Extract text from image using OCR for context
                        context_text = self._extract_image_text(img_file_path)
                        
                        # Create image info
                        image_info = {
                            'filename': img_filename,
                            'page': page_num,
                            'size': (pix.width, pix.height),
                            'format': 'PNG',
                            'data': base64.b64encode(img_data).decode('utf-8'),
                            'path': str(img_file_path),
                            'context_text': context_text,
                            'file_size': len(img_data)
                        }
                        
                        images.append(image_info)
                    
                    pix = None  # Clean up
                    
                except Exception as e:
                    print(f"⚠️ Could not extract image {img_index + 1} from page {page_num}: {e}")
                    continue
        
        except Exception as e:
            print(f"⚠️ Could not extract images from page {page_num}: {e}")
        
        return images
    
    def _extract_text_from_dict(self, text_dict: dict) -> str:
        """Extract text from PyMuPDF text dictionary"""
        text_content = []
        
        try:
            if 'blocks' in text_dict:
                for block in text_dict['blocks']:
                    if 'lines' in block:
                        for line in block['lines']:
                            if 'spans' in line:
                                line_text = ""
                                for span in line['spans']:
                                    if 'text' in span:
                                        line_text += span['text']
                                if line_text.strip():
                                    text_content.append(line_text.strip())
            
            return " ".join(text_content)
        except Exception as e:
            print(f"⚠️ Error extracting text from dict: {e}")
            return ""
    
    def _extract_image_text(self, image_path: Path) -> str:
        """Extract text from image for context using OCR"""
        try:
            from PIL import Image
            import pytesseract
            
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Extract text with basic OCR
                text = pytesseract.image_to_string(img, config='--psm 6')
                return text.strip()
        except Exception as e:
            print(f"⚠️ Could not extract text from image {image_path}: {e}")
            return ""
    
    def create_chunks(self, pdf_data: Dict, pdf_path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Create text chunks from PDF data with metadata"""
        text = pdf_data['text']
        pages_data = pdf_data['pages']
        
        if not text.strip():
            print(f"⚠️ No text found in {os.path.basename(pdf_path)}")
            return []
        
        print(f"🔍 Debug: Total text length: {len(text)} characters")
        print(f"🔍 Debug: Number of pages with data: {len(pages_data)}")
        
        chunks = []
        
        # Create chunks by pages first, then split large pages
        for page_data in pages_data:
            page_text = page_data['text'].strip()
            page_num = page_data['page_number']
            page_images = page_data['images']
            
            print(f"🔍 Debug: Page {page_num} - Text length: {len(page_text)} chars")
            
            # Be more lenient - accept pages with minimal text or images
            if not page_text and len(page_images) == 0:
                print(f"⚠️ Skipping page {page_num} - no text or images")
                continue
            
            # If page has no text but has images, create a minimal chunk for the images
            if not page_text and len(page_images) > 0:
                page_text = f"[Page {page_num} contains {len(page_images)} image(s)]"
                print(f"📄 Created image-only chunk for page {page_num}")
            
            # If page text is small enough, create single chunk
            if len(page_text) <= chunk_size:
                chunk = {
                    'text': page_text,
                    'source': pdf_path,
                    'chunk_id': len(chunks),
                    'type': 'pdf',
                    'page': page_num,
                    'char_count': len(page_text),
                    'word_count': len(page_text.split()),
                    'has_images': len(page_images) > 0,
                    'images': page_images,
                    'all_document_images': self._get_all_document_images(pdf_data),
                    'pdf_metadata': pdf_data['metadata']
                }
                chunks.append(chunk)
                print(f"✅ Created chunk {len(chunks)} for page {page_num} ({len(page_text)} chars)")
            else:
                # Split large pages into smaller chunks
                page_chunks = self._split_text_with_overlap(page_text, chunk_size, overlap)
                print(f"📄 Split page {page_num} into {len(page_chunks)} chunks")
                
                for i, chunk_text in enumerate(page_chunks):
                    chunk = {
                        'text': chunk_text,
                        'source': pdf_path,
                        'chunk_id': len(chunks),
                        'type': 'pdf',
                        'page': page_num,
                        'page_chunk': i + 1,
                        'char_count': len(chunk_text),
                        'word_count': len(chunk_text.split()),
                        'has_images': len(page_images) > 0 and i == 0,  # Only first chunk gets page images
                        'images': page_images if i == 0 else [],
                        'all_document_images': self._get_all_document_images(pdf_data),
                        'pdf_metadata': pdf_data['metadata']
                    }
                    chunks.append(chunk)
        
        print(f"📝 Created {len(chunks)} chunks from {os.path.basename(pdf_path)}")
        
        # Additional debugging if no chunks were created
        if len(chunks) == 0:
            print(f"❌ DEBUG: No chunks created!")
            print(f"   Total text: '{text[:200]}...' ({len(text)} chars)")
            print(f"   Pages data: {len(pages_data)} pages")
            for i, page_data in enumerate(pages_data[:3]):  # Show first 3 pages
                print(f"   Page {i+1}: '{page_data['text'][:100]}...' ({len(page_data['text'])} chars)")
        
        return chunks
    
    def _get_all_document_images(self, pdf_data: Dict) -> List[Dict]:
        """Get all images from the entire document"""
        all_images = []
        for page_data in pdf_data['pages']:
            all_images.extend(page_data['images'])
        return all_images
    
    def _split_text_with_overlap(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this is not the last chunk, try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence endings
                sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
                best_break = -1
                
                for i in range(end - 100, end):  # Look back up to 100 chars
                    if i > start and any(text[i:i+2] == ending for ending in sentence_endings):
                        best_break = i + 1
                        break
                
                # If no sentence break found, look for word boundary
                if best_break == -1:
                    for i in range(end - 50, end):  # Look back up to 50 chars
                        if i > start and text[i] == ' ':
                            best_break = i
                            break
                
                if best_break != -1:
                    end = best_break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def process_pdf(self, pdf_path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Process a single PDF file"""
        if not os.path.exists(pdf_path):
            print(f"❌ PDF file not found: {pdf_path}")
            return []
        
        if not pdf_path.lower().endswith('.pdf'):
            print(f"❌ Not a PDF file: {pdf_path}")
            return []
        
        # Extract text and metadata
        pdf_data = self.extract_text_from_pdf(pdf_path)
        
        if pdf_data['text']:
            return self.create_chunks(pdf_data, pdf_path, chunk_size, overlap)
        else:
            print(f"❌ No text extracted from {os.path.basename(pdf_path)}")
            return []
    
    def process_all_pdfs(self, pdf_directory: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Process all PDF files in a directory"""
        all_chunks = []
        
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"⚠️ No PDF files found in {pdf_directory}")
            return all_chunks
        
        print(f"📚 Found {len(pdf_files)} PDF files to process")
        
        # Track processing statistics
        total_pages = 0
        total_images = 0
        total_words = 0
        successful_pdfs = 0
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            print(f"\n📁 Processing {pdf_file}")
            
            chunks = self.process_pdf(pdf_path, chunk_size, overlap)
            
            if chunks:
                successful_pdfs += 1
                # Get stats from first chunk (contains document metadata)
                if chunks[0].get('all_document_images'):
                    total_images += len(chunks[0]['all_document_images'])
                
                file_words = sum(chunk['word_count'] for chunk in chunks)
                total_words += file_words
                
                # Count unique pages
                unique_pages = len(set(chunk['page'] for chunk in chunks))
                total_pages += unique_pages
                
                print(f"✅ Success: {len(chunks)} chunks, {file_words} words, {unique_pages} pages")
            else:
                print(f"❌ Failed to process {pdf_file}")
            
            all_chunks.extend(chunks)
        
        # Print summary
        print(f"\n📊 Processing Summary:")
        print(f"   📁 PDFs processed: {successful_pdfs}/{len(pdf_files)}")
        print(f"   📝 Total chunks: {len(all_chunks)}")
        print(f"   📖 Total words: {total_words}")
        print(f"   📄 Total pages: {total_pages}")
        print(f"   🖼️ Total images: {total_images}")
        
        return all_chunks
    
    def get_pdf_info(self, pdf_path: str) -> Dict:
        """Get basic information about a PDF file"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            page_count = len(doc)
            
            # Calculate file size
            file_size = os.path.getsize(pdf_path)
            
            info = {
                'filename': os.path.basename(pdf_path),
                'page_count': page_count,
                'file_size': file_size,
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', '')
            }
            
            doc.close()
            return info
            
        except Exception as e:
            print(f"❌ Error getting PDF info for {pdf_path}: {e}")
            return {}

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
            sample = chunks[0]
            print(f"Source: {os.path.basename(sample['source'])}")
            print(f"Type: {sample['type']}")
            print(f"Page: {sample['page']}")
            print(f"Words: {sample['word_count']}")
            print(f"Has images: {sample['has_images']}")
            print(f"Text preview: {sample['text'][:200]}...")
        else:
            print("💡 No text found in PDFs. Try adding PDFs with text content.")
    else:
        print(f"📁 Directory {pdf_dir} not found. Add some PDFs to test!")
        print("💡 Supported formats: .pdf")