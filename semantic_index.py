"""
Advanced semantic indexing system optimized for slang understanding
Uses multiple strategies and better FAISS indexing
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Set
import faiss

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install: pip install sentence-transformers")
    exit(1)

class AdvancedSemanticIndexer:
    """Enhanced semantic indexing with multiple embedding strategies"""
    
    def __init__(self):
        # Use multiple models for better semantic understanding
        print("🤖 Loading semantic models...")
        
        # Primary model - good for general semantic similarity
        self.primary_model = SentenceTransformer('all-mpnet-base-v2')  # Better than MiniLM
        
        # Secondary model - good for short text and slang
        self.secondary_model = SentenceTransformer('all-MiniLM-L12-v2')  # Larger than L6
        
        print("✅ Models loaded successfully")
        
        self.index = None
        self.metadata = None
        self.slang_terms = set()
    
    def create_rich_semantic_contexts(self, slang_dict: Dict) -> Tuple[List[str], List[Dict]]:
        """Create comprehensive semantic contexts for better matching"""
        texts = []
        metadata = []
        
        for slang_term, data in slang_dict.items():
            definition = data['definition']
            example = data.get('example', '')
            notes = data.get('notes', '')
            
            # Create comprehensive context variations
            contexts = [
                # Direct term contexts
                slang_term,
                f"{slang_term} slang",
                f"{slang_term} meaning",
                
                # Definition contexts
                definition,
                f"{slang_term} means {definition}",
                f"the meaning of {slang_term} is {definition}",
                
                # Query variations  
                f"what does {slang_term} mean",
                f"define {slang_term}",
                f"meaning of {slang_term}",
                f"{slang_term} definition",
                f"what is {slang_term}",
                
                # Usage contexts
                f"saying {slang_term}",
                f"using {slang_term}",
                f"word {slang_term}",
                
                # Conversational contexts
                f"hey {slang_term}",
                f"that's {slang_term}",
                f"so {slang_term}",
                f"really {slang_term}",
                f"very {slang_term}",
                f"totally {slang_term}",
            ]
            
            # Add example-based contexts if available
            if example and len(example.strip()) > 0:
                contexts.extend([
                    example,
                    f"example: {example}",
                    f"{slang_term} example: {example}",
                    # Extract key phrases from example
                    *self._extract_example_phrases(example, slang_term)
                ])
            
            # Add note-based contexts
            if notes and len(notes.strip()) > 0:
                contexts.extend([
                    notes,
                    f"{slang_term} context: {notes}"
                ])
            
            # Add morphological variations
            morphological_variants = self._generate_morphological_contexts(slang_term, definition)
            contexts.extend(morphological_variants)
            
            # Create metadata for each context
            base_metadata = {
                'slang_term': slang_term,
                'definition': definition,
                'example': example,
                'notes': notes
            }
            
            for context in contexts:
                texts.append(context.lower().strip())  # Normalize
                metadata.append(base_metadata.copy())
        
        print(f"📚 Created {len(texts)} semantic contexts for {len(slang_dict)} slang terms")
        return texts, metadata
    
    def _extract_example_phrases(self, example: str, slang_term: str) -> List[str]:
        """Extract useful phrases from examples"""
        phrases = []
        
        # Split example into phrases
        example_lower = example.lower()
        sentences = example_lower.replace('.', '!').replace('?', '!').split('!')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if slang_term in sentence and len(sentence.split()) >= 2:
                phrases.append(sentence)
                
                # Create variations without the slang term for broader matching
                without_slang = sentence.replace(slang_term, '').strip()
                if len(without_slang) > 0:
                    phrases.append(without_slang)
        
        return phrases
    
    def _generate_morphological_contexts(self, slang_term: str, definition: str) -> List[str]:
        """Generate morphological variations with semantic context"""
        contexts = []
        
        # Common morphological patterns for slang
        if len(slang_term) > 2:
            variants = [
                f"{slang_term}ed",     # past tense
                f"{slang_term}ing",    # gerund
                f"{slang_term}s",      # plural
                f"{slang_term}er",     # agent
                f"{slang_term}y",      # adjective
            ]
            
            # Handle words ending in 'e'
            if slang_term.endswith('e'):
                variants.extend([
                    f"{slang_term}d",           # vibed
                    f"{slang_term[:-1]}ing"     # vibing
                ])
            
            # Create contexts for each variant
            for variant in variants:
                contexts.extend([
                    variant,
                    f"{variant} means {definition}",
                    f"what does {variant} mean",
                    f"define {variant}",
                    f"that's {variant}",
                    f"so {variant}",
                    f"really {variant}"
                ])
        
        return contexts
    
    def create_ensemble_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create ensemble embeddings using multiple models"""
        print("🧠 Creating ensemble embeddings...")
        
        # Get embeddings from both models
        primary_embeddings = self.primary_model.encode(texts, show_progress_bar=True, batch_size=32)
        secondary_embeddings = self.secondary_model.encode(texts, show_progress_bar=False, batch_size=32)
        
        # Normalize embeddings
        primary_embeddings = primary_embeddings / np.linalg.norm(primary_embeddings, axis=1, keepdims=True)
        secondary_embeddings = secondary_embeddings / np.linalg.norm(secondary_embeddings, axis=1, keepdims=True)
        
        # Create ensemble by concatenation (better than averaging for this use case)
        ensemble_embeddings = np.concatenate([primary_embeddings, secondary_embeddings], axis=1)
        
        print(f"✅ Created ensemble embeddings: {ensemble_embeddings.shape}")
        return ensemble_embeddings.astype('float32')
    
    def build_advanced_index(self, slang_dict: Dict, index_path: str, mapping_path: str):
        """Build advanced FAISS index with optimizations"""
        if not slang_dict:
            return
        
        self.slang_terms = set(slang_dict.keys())
        
        # Create rich semantic contexts
        texts, metadata = self.create_rich_semantic_contexts(slang_dict)
        
        # Create ensemble embeddings
        embeddings = self.create_ensemble_embeddings(texts)
        
        # Build optimized FAISS index
        dimension = embeddings.shape[1]
        
        # Use IVF index for better performance with large datasets
        if len(embeddings) > 1000:
            print("🏗️  Building IVF index for large dataset...")
            nlist = min(100, len(embeddings) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            # Train the index
            self.index.train(embeddings)
            self.index.add(embeddings)
            
            # Set search parameters for better recall
            self.index.nprobe = min(20, nlist // 2)
            
        else:
            print("🏗️  Building flat index for small dataset...")
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
        
        self.metadata = metadata
        
        # Save index and metadata
        self._save_index(index_path, mapping_path)
        
        print(f"✅ Built semantic index with {len(texts)} entries")
    
    def _save_index(self, index_path: str, mapping_path: str):
        """Save index and metadata"""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'slang_terms': self.slang_terms
            }, f)
        
        print(f"💾 Saved index to {index_path}")
    
    def load_index(self, index_path: str, mapping_path: str) -> bool:
        """Load existing index"""
        try:
            self.index = faiss.read_index(index_path)
            
            with open(mapping_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.slang_terms = data['slang_terms']
            
            print(f"📂 Loaded existing index with {len(self.metadata)} entries")
            return True
            
        except (FileNotFoundError, Exception) as e:
            print(f"⚠️  Could not load index: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Advanced semantic search with multiple strategies"""
        if not self.index or not self.metadata:
            return []
        
        print(f"🔍 Searching: '{query}'")
        
        # Create ensemble embedding for query
        query_primary = self.primary_model.encode([query])
        query_secondary = self.secondary_model.encode([query])
        
        # Normalize
        query_primary = query_primary / np.linalg.norm(query_primary, axis=1, keepdims=True)
        query_secondary = query_secondary / np.linalg.norm(query_secondary, axis=1, keepdims=True)
        
        # Ensemble query embedding
        query_embedding = np.concatenate([query_primary, query_secondary], axis=1).astype('float32')
        
        # Search with more candidates for better results
        search_k = min(k * 10, len(self.metadata))
        distances, indices = self.index.search(query_embedding, search_k)
        
        # Process results with advanced filtering
        results = []
        seen_terms = set()
        term_scores = {}
        
        # Collect all candidates with scores
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                item = self.metadata[idx]
                slang_term = item['slang_term']
                distance = distances[0][i]
                
                # Convert distance to similarity score
                similarity = 1 / (1 + distance)
                
                # Aggregate scores for the same slang term (take best score)
                if slang_term not in term_scores or similarity > term_scores[slang_term]['similarity']:
                    term_scores[slang_term] = {
                        'item': item,
                        'similarity': similarity,
                        'distance': distance
                    }
        
        # Sort by similarity and apply adaptive threshold
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1]['similarity'], reverse=True)
        
        print(f"📊 Top candidates:")
        for i, (term, data) in enumerate(sorted_terms[:10]):
            print(f"  {i+1}. '{term}' - similarity: {data['similarity']:.3f}")
        
        # Adaptive threshold based on top result
        if sorted_terms:
            top_similarity = sorted_terms[0][1]['similarity']
            # More lenient threshold for slang
            threshold = max(0.3, top_similarity * 0.6)  # At least 60% of top score
            
            print(f"🎯 Using adaptive threshold: {threshold:.3f}")
            
            for term, data in sorted_terms:
                if data['similarity'] >= threshold and len(results) < k:
                    results.append({
                        'slang_term': term,
                        'definition': data['item']['definition'],
                        'example': data['item']['example'],
                        'notes': data['item']['notes'],
                        'similarity': data['similarity']
                    })
                elif len(results) >= k:
                    break
        
        print(f"✅ Found {len(results)} relevant results")
        return results
