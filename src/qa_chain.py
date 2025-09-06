# src/qa_chain.py - Optimized for Speed
import requests
import json
from typing import List, Tuple, Dict, Optional
import time

class QAChain:
    def __init__(self, vector_store, model_name="mistral", ollama_url="http://localhost:11434"):
        self.vector_store = vector_store
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.conversation_history = []
        
        print(f"🤖 QA Chain initialized with model: {model_name}")
        print(f"🔗 Ollama URL: {ollama_url}")
        
        # Test connection
        self._test_connection()
    
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
        
        # Limit context to most relevant chunks and shorter length
        limited_contexts = contexts[:3]  # Only use top 3 most relevant
        context_text = ""
        if limited_contexts:
            # Truncate long contexts
            truncated_contexts = []
            for i, ctx in enumerate(limited_contexts):
                if len(ctx) > 500:  # Limit context length
                    ctx = ctx[:500] + "..."
                truncated_contexts.append(f"Context {i+1}: {ctx}")
            context_text = "\n\n".join(truncated_contexts)
        
        # Shorter, more direct prompt
        prompt_parts = []
        prompt_parts.append("Answer the question based on the provided context. Be concise and direct.")
        
        if context_text:
            prompt_parts.append(f"Context:\n{context_text}")
        else:
            prompt_parts.append("No relevant context found.")
        
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("Answer:")
        
        return "\n".join(prompt_parts)
    
    def ask(self, question: str, k: int = 3, include_history: bool = False, temperature: float = 0.3) -> Tuple[str, List[str], List[Dict]]:
        """Ask a question with optimized settings for speed"""
        print(f"\n❓ Question: {question}")
        
        # Retrieve fewer contexts for faster processing
        contexts, metadata = self.vector_store.search(question, k)
        
        if not contexts:
            response = "I don't have any relevant documents to answer this question. Please add some documents to the knowledge base first."
            return response, [], []
        
        # Generate response with optimized prompt
        prompt = self.create_prompt(question, contexts, include_history)
        response = self.generate_response(prompt, temperature, max_tokens=500)  # Shorter responses
        
        # Extract sources
        sources = [meta['source'] for meta in metadata]
        unique_sources = list(dict.fromkeys(sources))  # Remove duplicates while preserving order
        
        # Add to conversation history
        self.conversation_history.append({
            'question': question,
            'answer': response,
            'sources': unique_sources,
            'context_count': len(contexts)
        })
        
        # Keep only last 5 conversations in history (reduced from 10)
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]
        
        print(f"✅ Answer generated using {len(contexts)} context chunks from {len(unique_sources)} sources")
        
        return response, unique_sources, metadata
    
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