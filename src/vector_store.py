# src/vector_store.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from typing import List, Dict, Tuple
import os

class VectorStore:
    def __init__(self, index_path: str, embedding_model="all-MiniLM-L6-v2"):
        self.index_path = index_path
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.metadata = []
        
        print(f"🔍 Vector store initialized with path: {index_path}")
        print(f"🤖 Embedding model: {embedding_model}")
        
        # Try to load existing index
        self.load()
    
    def _load_embedding_model(self):
        """Lazy load the embedding model"""
        if self.embedding_model is None:
            print(f"📥 Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print("✅ Embedding model loaded")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        self._load_embedding_model()
        print(f"🧮 Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        print(f"✅ Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def add_documents(self, docs: List[Dict], source_prefix: str = ""):
        """Add documents to the vector store"""
        if not docs:
            print("⚠️  No documents to add")
            return
        
        print(f"📚 Adding {len(docs)} documents to vector store...")
        
        texts = [doc['text'] for doc in docs]
        embeddings = self.embed_texts(texts)
        
        # Initialize index if not exists
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            print(f"🏗️  Created new FAISS index with dimension: {dimension}")
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        for doc in docs:
            self.documents.append(doc['text'])
            metadata = {
                'source': f"{source_prefix}:{doc.get('source', '')}" if source_prefix else doc.get('source', ''),
                'type': doc.get('type', 'unknown'),
                'chunk_id': doc.get('chunk_id', 0),
                'page': doc.get('page', 1)
            }
            self.metadata.append(metadata)
        
        print(f"✅ Added {len(docs)} documents. Total documents: {len(self.documents)}")
    
    def search(self, query: str, k: int = 5) -> Tuple[List[str], List[Dict]]:
        """Search for similar documents"""
        if self.index is None or self.index.ntotal == 0:
            print("⚠️  No documents in vector store")
            return [], []
        
        print(f"🔍 Searching for: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        
        # Generate query embedding
        query_embedding = self.embed_texts([query])
        
        # Search with more candidates for better diversity
        search_k = min(k * 2, self.index.ntotal)  # Search more candidates
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        # Filter and diversify results
        results = []
        result_metadata = []
        seen_sources = set()
        source_count = {}
        
        print(f"📊 Found {len(indices[0])} candidates, selecting best {k}")
        
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:  # Valid index
                metadata = self.metadata[idx].copy()
                metadata['distance'] = float(distances[0][i])
                metadata['rank'] = len(results) + 1
                
                source = metadata['source']
                
                # Limit results per source for diversity (max 2 chunks per source)
                source_count[source] = source_count.get(source, 0)
                if source_count[source] >= 2:
                    continue
                
                results.append(self.documents[idx])
                result_metadata.append(metadata)
                source_count[source] += 1
                
                print(f"  {len(results)}. Source: {os.path.basename(metadata['source'])} (distance: {metadata['distance']:.3f})")
                
                # Stop when we have enough results
                if len(results) >= k:
                    break
        
        return results, result_metadata
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        stats = {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.index.d if self.index else 0,
            'sources': list(set([meta['source'] for meta in self.metadata])),
            'types': list(set([meta['type'] for meta in self.metadata]))
        }
        return stats
    
    def save(self):
        """Save the vector store"""
        if self.index is not None and len(self.documents) > 0:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{self.index_path}.faiss")
            
            # Save documents and metadata
            with open(f"{self.index_path}.pkl", 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata,
                    'embedding_model': self.embedding_model_name
                }, f)
            
            print(f"💾 Vector store saved to {self.index_path}")
            
            # Print stats
            stats = self.get_stats()
            print(f"📊 Stats: {stats['total_documents']} docs, {len(stats['sources'])} sources")
        else:
            print("⚠️  Nothing to save")
    
    def load(self):
        """Load existing vector store"""
        try:
            # Load FAISS index
            if os.path.exists(f"{self.index_path}.faiss"):
                self.index = faiss.read_index(f"{self.index_path}.faiss")
                
                # Load documents and metadata
                with open(f"{self.index_path}.pkl", 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadata = data['metadata']
                    # Update embedding model if saved
                    if 'embedding_model' in data:
                        self.embedding_model_name = data['embedding_model']
                
                print(f"📥 Vector store loaded from {self.index_path}")
                stats = self.get_stats()
                print(f"📊 Loaded: {stats['total_documents']} docs, {len(stats['sources'])} sources")
                
        except Exception as e:
            print(f"⚠️  Could not load existing vector store: {e}")
            print("🆕 Will create a new vector store")

# Test the vector store
if __name__ == "__main__":
    print("🧪 Testing Vector Store...")
    
    # Create test vector store
    vs = VectorStore("../models/test_index")
    
    # Test with sample documents
    test_docs = [
        {
            'text': 'The quick brown fox jumps over the lazy dog.',
            'source': 'test.txt',
            'type': 'text',
            'chunk_id': 0,
            'page': 1
        },
        {
            'text': 'Machine learning is a subset of artificial intelligence.',
            'source': 'ai.txt', 
            'type': 'text',
            'chunk_id': 1,
            'page': 1
        },
        {
            'text': 'Python is a popular programming language for data science.',
            'source': 'programming.txt',
            'type': 'text', 
            'chunk_id': 2,
            'page': 1
        }
    ]
    
    # Add documents
    vs.add_documents(test_docs, "test")
    
    # Test search
    results, metadata = vs.search("artificial intelligence", k=2)
    
    print(f"\n🔍 Search results:")
    for i, (result, meta) in enumerate(zip(results, metadata)):
        print(f"{i+1}. {result}")
        print(f"   Source: {meta['source']}, Distance: {meta['distance']:.3f}")
    
    # Save for testing
    vs.save()
    
    # Show stats
    print(f"\n📊 Final stats: {vs.get_stats()}")