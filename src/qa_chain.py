# src/qa_chain.py - Optimized for Speed
import requests
import json
from typing import List, Tuple, Dict, Optional
import time
import os
import base64
import numpy as np
from sentence_transformers import SentenceTransformer

class QAChain:
    def __init__(self, vector_store, model_name="mistral", ollama_url="http://localhost:11434"):
        self.vector_store = vector_store
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.conversation_history = []
        self.offline_mode = True
        self.image_embedding_model = None
        self.citation_cache = {}  # Cache for citation formatting
        
        print(f"🤖 QA Chain initialized with model: {model_name}")
        print(f"🔗 Ollama URL: {ollama_url}")
        
        # Test connection
        self._test_connection()
    
    def _load_image_embedding_model(self):
        """Load model for image relevance scoring"""
        if self.image_embedding_model is None:
            print("📥 Loading image relevance model...")
            self.image_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Image relevance model loaded")
    
    def _test_connection(self):
        """Test connection to Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                
                if self.model_name in available_models or any(self.model_name in m for m in available_models):
                    print(f"✅ Ollama connected, model '{self.model_name}' available")
                else:
                    print(f"⚠️  Model '{self.model_name}' not found. Available: {available_models}")
                    print(f"💡 Run: ollama pull {self.model_name}")
            else:
                print(f"❌ Ollama connection error: {response.status_code}")
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            print("💡 Make sure Ollama is running: ollama serve")
    
    def generate_response(self, prompt: str, temperature: float = 0.3, max_tokens: int = 800) -> str:
        """Generate response using local Ollama model with optimized settings"""
        try:
            # Optimized payload for faster responses
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_k": 20,
                    "top_p": 0.8,
                    "repeat_penalty": 1.1,
                    "num_ctx": 2048  # Reduce context window for speed
                }
            }
            
            print(f"🤔 Generating response with {self.model_name}...")
            start_time = time.time()
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=180  # Increased timeout to 3 minutes
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                # Get generation stats
                total_duration = result.get('total_duration', 0) / 1e9  # Convert to seconds
                eval_count = result.get('eval_count', 0)
                
                print(f"✅ Response generated in {end_time - start_time:.1f}s")
                if eval_count > 0 and total_duration > 0:
                    print(f"📊 Tokens: {eval_count}, Speed: {eval_count/total_duration:.1f} tokens/s")
                
                return generated_text
            else:
                error_msg = f"Ollama API error: {response.status_code}"
                print(f"❌ {error_msg}")
                return error_msg
        
        except requests.exceptions.Timeout:
            return "⏰ Request timed out. Try asking a simpler question or switch to a faster model like 'phi3:mini'."
        except Exception as e:
            error_msg = f"Error connecting to Ollama: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def create_prompt(self, question: str, contexts: List[str], include_history: bool = False) -> str:
        """Create an optimized prompt for faster processing"""
        
        # Intelligently select contexts based on relevance and diversity
        limited_contexts = self._select_diverse_contexts(contexts, max_contexts=4)
        context_text = ""
        if limited_contexts:
            # Truncate long contexts but preserve important information
            truncated_contexts = []
            for i, ctx in enumerate(limited_contexts):
                if len(ctx) > 600:  # Slightly longer for better context
                    # Try to break at sentence boundary
                    sentences = ctx.split('. ')
                    truncated = ""
                    for sentence in sentences:
                        if len(truncated + sentence) < 600:
                            truncated += sentence + ". "
                        else:
                            break
                    ctx = truncated.strip() + "..." if truncated else ctx[:600] + "..."
                truncated_contexts.append(f"[Source {i+1}] {ctx}")
            context_text = "\n\n".join(truncated_contexts)
        
        # Enhanced prompt with better citation instructions
        prompt_parts = []
        prompt_parts.append("Answer the question based on the provided context. Be accurate and cite your sources using [Source X] format. Include specific page numbers, timestamps, or image details when available.")
        
        if context_text:
            prompt_parts.append(f"Available Context:\n{context_text}")
        else:
            prompt_parts.append("No relevant context found.")
        
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("Answer (cite sources as [Source X] and include specific details like page numbers, timestamps, or confidence scores):")
        
        return "\n".join(prompt_parts)
    
    def _select_diverse_contexts(self, contexts: List[str], max_contexts: int = 4) -> List[str]:
        """Select diverse contexts to avoid redundancy"""
        if len(contexts) <= max_contexts:
            return contexts
        
        # Simple diversity selection - take top context and then skip similar ones
        selected = [contexts[0]]  # Always include most relevant
        
        for ctx in contexts[1:]:
            if len(selected) >= max_contexts:
                break
            
            # Simple similarity check - avoid very similar contexts
            is_diverse = True
            for selected_ctx in selected:
                # Check if contexts are too similar (simple word overlap)
                ctx_words = set(ctx.lower().split())
                selected_words = set(selected_ctx.lower().split())
                overlap = len(ctx_words.intersection(selected_words))
                similarity = overlap / min(len(ctx_words), len(selected_words)) if min(len(ctx_words), len(selected_words)) > 0 else 0
                
                if similarity > 0.7:  # Too similar
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(ctx)
        
        return selected
    
    def format_citations(self, response: str, metadata: List[Dict], contexts: List[str]) -> str:
        """Format response with stronger citations"""
        if not metadata:
            return response
        
        # Create comprehensive citation information with context matching
        citations = []
        source_details = {}
        
        for i, meta in enumerate(metadata, 1):
            source_name = os.path.basename(meta['source'])
            page = meta.get('page', 1)
            doc_type = meta.get('type', 'document')
            confidence = meta.get('confidence', 0)
            distance = meta.get('distance', 0)
            
            # Create detailed citation based on document type
            citation_parts = [f"[{i}]"]
            
            if doc_type == 'pdf':
                citation_parts.append(f"{source_name}")
                citation_parts.append(f"Page {page}")
                if meta.get('page_chunk'):
                    citation_parts.append(f"Section {meta['page_chunk']}")
            elif doc_type == 'image':
                citation_parts.append(f"{source_name}")
                citation_parts.append(f"OCR")
                if confidence > 0:
                    citation_parts.append(f"{confidence:.1f}% confidence")
                image_size = meta.get('image_size')
                if image_size:
                    citation_parts.append(f"{image_size[0]}x{image_size[1]}px")
            elif doc_type == 'audio':
                citation_parts.append(f"{source_name}")
                citation_parts.append("Audio")
                if meta.get('start_time') is not None and meta.get('end_time') is not None:
                    citation_parts.append(f"{meta.get('formatted_time', 'N/A')}")
                if meta.get('speaker'):
                    citation_parts.append(f"Speaker {meta['speaker']}")
                if confidence > 0:
                    citation_parts.append(f"{confidence:.1f}% confidence")
            else:
                citation_parts.append(f"{source_name}")
            
            # Add relevance score
            relevance = max(0, min(100, int((1 - distance) * 100))) if distance > 0 else 100
            citation_parts.append(f"Relevance: {relevance}%")
            
            citation = " - ".join(citation_parts)
            citations.append(citation)
            
            # Store source details for summary
            if source_name not in source_details:
                source_details[source_name] = {
                    'type': doc_type,
                    'pages': set(),
                    'confidence': [],
                    'relevance': []
                }
            source_details[source_name]['pages'].add(page)
            if confidence > 0:
                source_details[source_name]['confidence'].append(confidence)
            source_details[source_name]['relevance'].append(relevance)
        
        # Enhanced citation formatting
        citation_text = "\n\n**📚 Detailed Sources:**\n" + "\n".join(citations)
        
        # Add source summary
        if len(source_details) > 1:
            citation_text += f"\n\n**📊 Source Summary:**\n"
            citation_text += f"• {len(source_details)} unique documents referenced\n"
            citation_text += f"• {len(metadata)} total context chunks used\n"
            
            # Add per-source summary
            for source, details in source_details.items():
                pages_str = f"Pages {min(details['pages'])}-{max(details['pages'])}" if len(details['pages']) > 1 else f"Page {list(details['pages'])[0]}"
                avg_conf = sum(details['confidence']) / len(details['confidence']) if details['confidence'] else 0
                avg_rel = sum(details['relevance']) / len(details['relevance']) if details['relevance'] else 0
                
                summary_parts = [f"• {source}"]
                summary_parts.append(pages_str)
                if avg_conf > 0:
                    summary_parts.append(f"Avg confidence: {avg_conf:.1f}%")
                summary_parts.append(f"Avg relevance: {avg_rel:.1f}%")
                
                citation_text += " - ".join(summary_parts) + "\n"
        
        return response + citation_text
    
    def get_relevant_images(self, query: str, metadata: List[Dict], max_images: int = 5) -> List[Dict]:
        """Extract and rank relevant images from metadata based on query"""
        relevant_images = []
        all_images = []
        
        # Collect all images from all chunks
        for meta in metadata:
            if meta.get('has_images', False):
                images = meta.get('images', [])
                for img in images:
                    img_info = {
                        'path': img['path'],
                        'filename': img['filename'],
                        'source': os.path.basename(meta['source']),
                        'page': meta.get('page', 1),
                        'size': img.get('size', (0, 0)),
                        'context_text': img.get('context_text', ''),
                        'chunk_relevance_score': 1.0 / (meta.get('distance', 1) + 0.1),  # Higher score for more relevant chunks
                        'data': img.get('data', ''),  # Base64 encoded image data
                        'file_size': img.get('file_size', 0)
                    }
                    all_images.append(img_info)
            
            # Also check for all document images in case we need broader context
            all_doc_images = meta.get('all_document_images', [])
            for img in all_doc_images:
                if not any(existing['path'] == img['path'] for existing in all_images):
                    img_info = {
                        'path': img['path'],
                        'filename': img['filename'],
                        'source': os.path.basename(meta['source']),
                        'page': img.get('page', 1),
                        'size': img.get('size', (0, 0)),
                        'context_text': img.get('context_text', ''),
                        'chunk_relevance_score': 0.5,  # Lower score for non-chunk images
                        'data': img.get('data', ''),
                        'file_size': img.get('file_size', 0)
                    }
                    all_images.append(img_info)
        
        if not all_images:
            return []
        
        # Rank images by relevance to query
        try:
            self._load_image_embedding_model()
            
            # Create embeddings for query and image contexts
            query_embedding = self.image_embedding_model.encode([query])
            
            image_contexts = []
            for img in all_images:
                # Combine filename and context for relevance scoring
                context = f"{img['filename']} {img['context_text']}"
                image_contexts.append(context)
            
            if image_contexts:
                context_embeddings = self.image_embedding_model.encode(image_contexts)
                
                # Calculate similarity scores
                similarities = np.dot(query_embedding, context_embeddings.T)[0]
                
                # Combine with chunk relevance scores
                for i, img in enumerate(all_images):
                    img['text_similarity'] = float(similarities[i])
                    img['combined_score'] = (
                        img['text_similarity'] * 0.7 + 
                        img['chunk_relevance_score'] * 0.3
                    )
                    
                    # Add display-ready information
                    img['display_info'] = {
                        'title': f"{img['filename']} (Page {img['page']})",
                        'subtitle': f"From: {img['source']}",
                        'relevance': f"{img['combined_score']:.1%}",
                        'size_info': f"{img['size'][0]}×{img['size'][1]}px" if img['size'][0] > 0 else "Unknown size"
                    }
                
                # Sort by combined score
                all_images.sort(key=lambda x: x['combined_score'], reverse=True)
                
                print(f"🖼️ Ranked {len(all_images)} images by relevance to query")
                
                # Return top images
                relevant_images = all_images[:max_images]
                
                # Log top matches
                for i, img in enumerate(relevant_images[:3]):
                    print(f"  {i+1}. {img['filename']} (score: {img['combined_score']:.3f})")
            
        except Exception as e:
            print(f"⚠️ Could not rank images by relevance: {e}")
            # Fallback: return images from most relevant chunks
            relevant_images = sorted(all_images, key=lambda x: x['chunk_relevance_score'], reverse=True)[:max_images]
            
            # Add basic display info for fallback
            for img in relevant_images:
                img['display_info'] = {
                    'title': f"{img['filename']} (Page {img['page']})",
                    'subtitle': f"From: {img['source']}",
                    'relevance': "High",
                    'size_info': f"{img['size'][0]}×{img['size'][1]}px" if img['size'][0] > 0 else "Unknown size"
                }
        
        return relevant_images
    
    def ask(self, question: str, k: int = 3, include_history: bool = False, temperature: float = 0.3) -> Tuple[str, List[str], List[Dict], List[Dict]]:
        """Ask a question with optimized settings for speed"""
        print(f"\n❓ Question: {question}")
        
        # Check if vector store has documents
        if not hasattr(self.vector_store, 'index') or self.vector_store.index is None or self.vector_store.index.ntotal == 0:
            return "❌ No documents available. Please upload and process some documents first.", [], [], []
        
        # Retrieve fewer contexts for faster processing
        contexts, metadata = self.vector_store.search(question, k)
        
        if not contexts:
            response = "I don't have any relevant documents to answer this question. Please add some documents to the knowledge base first."
            return response, [], [], []
        
        # Generate response with optimized prompt
        prompt = self.create_prompt(question, contexts, include_history)
        response = self.generate_response(prompt, temperature, max_tokens=800)  # Allow longer for better citations
        
        # Format response with stronger citations
        response = self.format_citations(response, metadata, contexts)
        
        # Extract sources
        sources = [meta['source'] for meta in metadata]
        unique_sources = list(dict.fromkeys(sources))  # Remove duplicates while preserving order
        
        # Get relevant images
        relevant_images = self.get_relevant_images(question, metadata, max_images=5)
        
        # Add to conversation history
        self.conversation_history.append({
            'question': question,
            'answer': response,
            'sources': unique_sources,
            'context_count': len(contexts),
            'images': relevant_images,
            'metadata_summary': self._create_metadata_summary(metadata)
        })
        
        # Keep only last 5 conversations in history (reduced from 10)
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]
        
        print(f"✅ Answer generated using {len(contexts)} context chunks from {len(unique_sources)} sources")
        if relevant_images:
            print(f"🖼️ Found {len(relevant_images)} relevant images")
        
        return response, unique_sources, metadata, relevant_images
    
    def _create_metadata_summary(self, metadata: List[Dict]) -> Dict:
        """Create a summary of metadata for analysis"""
        summary = {
            'total_chunks': len(metadata),
            'document_types': {},
            'avg_relevance': 0,
            'page_range': {'min': float('inf'), 'max': 0},
            'confidence_range': {'min': 100, 'max': 0}
        }
        
        relevance_scores = []
        for meta in metadata:
            doc_type = meta.get('type', 'unknown')
            summary['document_types'][doc_type] = summary['document_types'].get(doc_type, 0) + 1
            
            # Calculate relevance from distance
            distance = meta.get('distance', 0)
            relevance = max(0, min(100, int((1 - distance) * 100))) if distance > 0 else 100
            relevance_scores.append(relevance)
            
            # Track page ranges
            page = meta.get('page', 1)
            summary['page_range']['min'] = min(summary['page_range']['min'], page)
            summary['page_range']['max'] = max(summary['page_range']['max'], page)
            
            # Track confidence ranges
            confidence = meta.get('confidence', 0)
            if confidence > 0:
                summary['confidence_range']['min'] = min(summary['confidence_range']['min'], confidence)
                summary['confidence_range']['max'] = max(summary['confidence_range']['max'], confidence)
        
        summary['avg_relevance'] = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        return summary
    
    def quick_ask(self, question: str) -> str:
        """Ultra-fast question answering with minimal context"""
        contexts, metadata = self.vector_store.search(question, k=1)  # Only 1 context
        
        if not contexts:
            return "No relevant information found."
        
        # Ultra-simple prompt
        prompt = f"Context: {contexts[0][:300]}...\nQuestion: {question}\nBrief answer:"
        
        return self.generate_response(prompt, temperature=0.1, max_tokens=200)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🗑️  Conversation history cleared")
    
    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def set_model(self, model_name: str):
        """Change the model being used"""
        old_model = self.model_name
        self.model_name = model_name
        print(f"🔄 Model changed from {old_model} to {model_name}")
        self._test_connection()

# Test the optimized QA chain
if __name__ == "__main__":
    print("🧪 Testing Optimized QA Chain...")
    print("💡 This version is optimized for speed with:")
    print("   • Reduced context size")
    print("   • Faster model parameters") 
    print("   • Shorter responses")
    print("   • Increased timeout handling")