# src/image_processor.py
import pytesseract
from PIL import Image
from typing import List, Dict, Optional
import os

class ImageProcessor:
    def __init__(self, tesseract_path: Optional[str] = None):
        self.tesseract_path = tesseract_path
        print("🖼️  Image processor initialized")
        
        # Set tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Test if tesseract is available
        self._test_tesseract()
    
    def _test_tesseract(self):
        """Test if Tesseract OCR is available"""
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract OCR found - Version: {version}")
            return True
        except Exception as e:
            print(f"❌ Tesseract OCR not found: {e}")
            print("💡 Install Tesseract OCR:")
            print("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
            print("   Mac: brew install tesseract")
            print("   Linux: sudo apt install tesseract-ocr")
            return False
    
    def ocr_image(self, image_path: str, lang: str = 'eng') -> Dict:
        """Extract text from image using OCR"""
        try:
            print(f"🔍 Processing image: {os.path.basename(image_path)}")
            
            # Open and process image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text with confidence scores
            data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
            
            # Filter out low-confidence text
            texts = []
            confidences = []
            
            for i, conf in enumerate(data['conf']):
                if int(conf) > 30:  # Only keep text with >30% confidence
                    text = data['text'][i].strip()
                    if text:
                        texts.append(text)
                        confidences.append(int(conf))
            
            # Combine all text
            full_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            print(f"✅ OCR complete: {len(full_text)} characters, avg confidence: {avg_confidence:.1f}%")
            
            return {
                'text': full_text,
                'confidence': avg_confidence,
                'word_count': len(texts),
                'language': lang,
                'image_size': image.size
            }
            
        except Exception as e:
            print(f"❌ Error processing image {image_path}: {e}")
            return {
                'text': '',
                'confidence': 0,
                'word_count': 0,
                'language': lang,
                'image_size': (0, 0)
            }
    
    def process_image(self, image_path: str, lang: str = 'eng') -> List[Dict]:
        """Process image and return text chunks"""
        ocr_result = self.ocr_image(image_path, lang)
        
        text = ocr_result['text']
        if not text.strip():
            print(f"⚠️  No text found in {os.path.basename(image_path)}")
            return []
        
        # Create chunks (split long text if needed)
        chunks = self._create_image_chunks(text, image_path, ocr_result)
        
        print(f"📝 Created {len(chunks)} chunks from {os.path.basename(image_path)}")
        return chunks
    
    def _create_image_chunks(self, text: str, image_path: str, ocr_result: Dict, max_chunk_size: int = 1000) -> List[Dict]:
        """Create chunks from image text"""
        chunks = []
        
        # If text is short enough, create single chunk
        if len(text) <= max_chunk_size:
            chunk = {
                'text': text,
                'source': image_path,
                'chunk_id': 0,
                'type': 'image',
                'confidence': ocr_result['confidence'],
                'word_count': ocr_result['word_count'],
                'language': ocr_result['language'],
                'image_size': ocr_result['image_size']
            }
            chunks.append(chunk)
        else:
            # Split into multiple chunks
            words = text.split()
            current_chunk = ""
            chunk_id = 0
            
            for word in words:
                if len(current_chunk + " " + word) <= max_chunk_size:
                    current_chunk += " " + word if current_chunk else word
                else:
                    if current_chunk:
                        chunk = {
                            'text': current_chunk,
                            'source': image_path,
                            'chunk_id': chunk_id,
                            'type': 'image',
                            'confidence': ocr_result['confidence'],
                            'word_count': len(current_chunk.split()),
                            'language': ocr_result['language'],
                            'image_size': ocr_result['image_size']
                        }
                        chunks.append(chunk)
                        chunk_id += 1
                    current_chunk = word
            
            # Add final chunk
            if current_chunk:
                chunk = {
                    'text': current_chunk,
                    'source': image_path,
                    'chunk_id': chunk_id,
                    'type': 'image',
                    'confidence': ocr_result['confidence'],
                    'word_count': len(current_chunk.split()),
                    'language': ocr_result['language'],
                    'image_size': ocr_result['image_size']
                }
                chunks.append(chunk)
        
        return chunks
    
    def process_all_images(self, image_directory: str, lang: str = 'eng') -> List[Dict]:
        """Process all images in a directory"""
        all_chunks = []
        supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp']
        
        image_files = [f for f in os.listdir(image_directory) 
                      if any(f.lower().endswith(fmt) for fmt in supported_formats)]
        
        if not image_files:
            print(f"⚠️  No image files found in {image_directory}")
            print(f"   Supported formats: {', '.join(supported_formats)}")
            return all_chunks
        
        print(f"🖼️  Found {len(image_files)} image files to process")
        
        # Track OCR statistics
        total_text_found = 0
        successful_ocr = 0
        
        for image_file in image_files:
            image_path = os.path.join(image_directory, image_file)
            chunks = self.process_image(image_path, lang)
            
            if chunks and chunks[0]['text'].strip():
                successful_ocr += 1
                total_text_found += sum(len(chunk['text']) for chunk in chunks)
            
            all_chunks.extend(chunks)
        
        print(f"✅ Processed {len(image_files)} images:")
        print(f"   📄 {successful_ocr} images contained text")
        print(f"   📝 {len(all_chunks)} chunks created")
        print(f"   📊 {total_text_found} total characters extracted")
        
        return all_chunks
    
    def get_image_info(self, image_path: str) -> Dict:
        """Get basic information about an image"""
        try:
            with Image.open(image_path) as img:
                return {
                    'filename': os.path.basename(image_path),
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format,
                    'file_size': os.path.getsize(image_path)
                }
        except Exception as e:
            print(f"❌ Error getting image info for {image_path}: {e}")
            return {}

# Test the image processor
if __name__ == "__main__":
    print("🧪 Testing Image Processor...")
    
    processor = ImageProcessor()
    
    # Test with data/images directory
    image_dir = "../data/images"
    if os.path.exists(image_dir):
        chunks = processor.process_all_images(image_dir)
        print(f"📊 Result: {len(chunks)} chunks created")
        
        if chunks:
            print("\n📝 Sample chunk:")
            sample = chunks[0]
            print(f"Source: {sample['source']}")
            print(f"Type: {sample['type']}")
            print(f"Confidence: {sample['confidence']:.1f}%")
            print(f"Image size: {sample['image_size']}")
            print(f"Text preview: {sample['text'][:200]}...")
        else:
            print("💡 No text found in images. Try adding images with text content.")
    else:
        print(f"📁 Directory {image_dir} not found. Add some images to test!")
        print("💡 Supported formats: .png, .jpg, .jpeg, .bmp, .tiff, .gif, .webp")