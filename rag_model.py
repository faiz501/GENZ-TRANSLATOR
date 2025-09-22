"""
Simple, stable RAG system that avoids memory issues
"""

import pandas as pd
import requests
import json
import numpy as np
from typing import Set, Dict, List, Tuple
from morphology import MorphologyCorrector

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install: pip install faiss-cpu sentence-transformers")
    exit(1)

class StableRAG:
    """Memory-efficient, stable RAG system"""
    
    def __init__(self, rag_data_path: str, ollama_url: str = "http://localhost:11434/api/chat"):
        self.ollama_url = ollama_url
        
        # Load slang data
        print("📚 Loading slang database...")
        self.slang_dict, self.slang_terms = self._load_slang_data(rag_data_path)
        
        # Initialize morphology corrector
        print("🔧 Initializing morphology corrector...")
        self.morphology = MorphologyCorrector(self.slang_terms)
        
        # Build simple semantic index (avoid memory issues)
        print("🧠 Building simple semantic index...")
        self._build_simple_index()
        
        print("✅ Stable RAG system ready!")
    
    def _load_slang_data(self, path: str) -> Tuple[Dict, Set]:
        """Load slang data"""
        try:
            df = pd.read_csv(path)
            slang_dict = {}
            slang_terms = set()
            
            for _, row in df.iterrows():
                slang = str(row['slang']).lower().strip()
                slang_dict[slang] = {
                    'definition': str(row['definition']).strip(),
                    'example': str(row.get('example', '')).strip(),
                    'notes': str(row.get('notes', '')).strip()
                }
                slang_terms.add(slang)
            
            print(f"📊 Loaded {len(slang_dict)} slang terms")
            return slang_dict, slang_terms
            
        except Exception as e:
            print(f"❌ Error loading slang data: {e}")
            return {}, set()
    
    def _build_simple_index(self):
        """Build simple, memory-efficient semantic index"""
        if not self.slang_dict:
            return
        
        # Use lightweight model to avoid memory issues
        print("🤖 Loading lightweight model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create minimal contexts (max 3 per slang term) [web:97]
        texts = []
        metadata = []
        
        for slang_term, data in self.slang_dict.items():
            definition = data['definition']
            
            # Only 3 essential contexts per term to avoid memory issues
            contexts = [
                slang_term,
                f"{slang_term} means {definition}",
                definition
            ]
            
            base_metadata = {
                'slang_term': slang_term,
                'definition': definition,
                'example': data['example'],
                'notes': data['notes']
            }
            
            for context in contexts:
                texts.append(context.lower().strip())
                metadata.append(base_metadata.copy())
        
        print(f"📚 Created {len(texts)} simple contexts")
        
        # Create embeddings with small batch size to avoid segfault [web:97][web:98]
        print("🧠 Creating embeddings with small batches...")
        embeddings = self.model.encode(
            texts, 
            batch_size=8,  # Very small batch size to avoid memory issues [web:97]
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Build simple FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # Simple flat index to avoid segfaults [web:102]
        self.index.add(embeddings.astype('float32'))
        self.metadata = metadata
        
        print(f"✅ Built stable index with {len(texts)} entries")
    
    def semantic_search(self, query: str, k: int = 3) -> List[Dict]:
        """Simple semantic search"""
        if not hasattr(self, 'index'):
            return []
        
        print(f"🔍 Searching: '{query}'")
        
        # Encode query with single batch to avoid issues
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        
        # Search with small k to avoid issues
        search_k = min(k * 2, len(self.metadata))
        distances, indices = self.index.search(query_embedding, search_k)
        
        results = []
        seen_terms = set()
        
        print(f"📊 Raw search results:")
        for i, idx in enumerate(indices[0][:10]):
            if idx < len(self.metadata):
                item = self.metadata[idx]
                slang_term = item['slang_term']
                distance = distances[0][i]
                similarity = 1 / (1 + distance)
                
                print(f"  {i+1}. '{slang_term}' - distance: {distance:.3f}, similarity: {similarity:.3f}")
                
                # Relaxed threshold for better results
                if distance < 1.5 and slang_term not in seen_terms:
                    results.append({
                        'slang_term': slang_term,
                        'definition': item['definition'],
                        'example': item['example'],
                        'notes': item['notes'],
                        'similarity': similarity
                    })
                    seen_terms.add(slang_term)
                    
                    if len(results) >= k:
                        break
        
        print(f"✅ Found {len(results)} relevant results")
        return results
    
    def direct_slang_check(self, text: str) -> List[Dict]:
        """Direct check for slang terms in text"""
        words = text.lower().split()
        found_slang = []
        
        for word in words:
            # Remove punctuation
            clean_word = word.strip('.,!?;:)"\'')
            if clean_word in self.slang_terms:
                found_slang.append({
                    'slang_term': clean_word,
                    'definition': self.slang_dict[clean_word]['definition'],
                    'example': self.slang_dict[clean_word]['example'],
                    'notes': self.slang_dict[clean_word]['notes'],
                    'similarity': 1.0
                })
                print(f"📍 Direct match: '{clean_word}'")
        
        return found_slang
    
    def generate_normalized_text(self, query: str, context: List[Dict]) -> str:
        """Generate normalized text with natural, concise replacements"""
        if not context:
            return query
        
        # Build simple, direct mappings
        context_str = ""
        for item in context:
            # Use just the core meaning, not the full definition
            definition = item['definition']
            # Extract the shortest, most natural equivalent
            if 'friend' in definition.lower():
                simple_def = 'friend'
            elif 'family' in definition.lower():
                simple_def = 'friend'  # "fam" is casual for friend
            else:
                # Take the first few words of definition
                simple_def = definition.split(',')[0].split(' or ')[0].strip()
            
            context_str += f"'{item['slang_term']}' = {simple_def}\n"
        
        prompt = f"""Convert slang to natural standard English. Use the shortest, most natural replacement.

    REPLACEMENTS:
    {context_str.strip()}

    RULES:
    - Replace slang with the SIMPLEST equivalent 
    - Keep the same casual tone
    - Don't add extra words
    - Make it sound natural

    INPUT: {query}
    OUTPUT:"""

        try:
            response = requests.post(self.ollama_url, json={
                "model": "nemotron-mini:4b",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.05,  # Very low for consistent results
                    "top_p": 0.7,
                    "max_tokens": 50  # Limit output length
                }
            }, timeout=30)
            
            response.raise_for_status()
            result = response.json()['message']['content'].strip()
            
            # Clean result
            if result.lower().startswith('output:'):
                result = result[7:].strip()
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            
            return result if result else query
            
        except Exception as e:
            print(f"⚠️  Generation error: {e}")
            return query

    
    def normalize(self, user_input: str) -> str:
        """Main normalization pipeline"""
        print(f"\n🔄 Processing: '{user_input}'")
        
        # Step 1: Morphology correction
        corrected_input, corrections = self.morphology.correct_sentence(user_input)
        
        if corrections:
            correction_summary = [f"'{c['original']}' → '{c['corrected']}'" for c in corrections]
            print(f"🔧 Corrections: {', '.join(correction_summary)}")
            print(f"✅ After morphology: '{corrected_input}'")
            query_to_use = corrected_input
        else:
            print("✅ No corrections needed")
            query_to_use = user_input
        
        # Step 2: Try direct matching first
        direct_matches = self.direct_slang_check(query_to_use)
        
        if direct_matches:
            print(f"🎯 Direct matches: {[m['slang_term'] for m in direct_matches]}")
            context = direct_matches
        else:
            # Step 3: Semantic search as fallback
            context = self.semantic_search(query_to_use)
        
        # Step 4: Generate result
        if context:
            result = self.generate_normalized_text(query_to_use, context)
            print(f"🤖 Generated: '{result}'")
            return result
        else:
            print("ℹ️  No slang detected")
            return query_to_use


SimpleRAG = StableRAG
