import pandas as pd
import os
from datasets import load_dataset
import re

def preprocess_slang_data(output_path: str):
    """
    Preprocess slang database for optimal RAG and semantic understanding
    Makes data structured for both semantic search and LLM understanding
    """
    print("📥 Loading slang dataset...")
    
    try:
        ds = load_dataset("MLBtrio/genz-slang-dataset")
        df = ds['train'].to_pandas()
        
        print(f"📊 Original dataset: {df.shape[0]} entries")
        
        # Map columns to standard format
        column_mapping = {
            'Slang': 'slang',
            'Description': 'definition', 
            'Example': 'example',
            'Context': 'notes'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure all required columns exist
        required_cols = ['slang', 'definition', 'example', 'notes']
        for col in required_cols:
            if col not in df.columns:
                df[col] = ''
        
        df = df[required_cols].copy()
        
        # Advanced preprocessing for better RAG performance
        df = df.dropna(subset=['slang', 'definition'])
        
        # Clean and normalize slang terms
        df['slang'] = df['slang'].astype(str).str.lower().str.strip()
        df['slang'] = df['slang'].str.replace(r'[^\w\s\'-]', '', regex=True)  # Remove special chars except apostrophes
        
        # Clean definitions for better LLM understanding
        df['definition'] = df['definition'].astype(str).str.strip()
        df['definition'] = df['definition'].str.replace(r'\s+', ' ', regex=True)  # Normalize whitespace
        
        # Process examples for semantic context
        df['example'] = df['example'].astype(str).str.strip()
        df['notes'] = df['notes'].astype(str).str.strip()
        
        # Remove duplicates and invalid entries
        df = df[df['slang'].str.len() > 0]  # Remove empty slang
        df = df[df['definition'].str.len() > 3]  # Remove very short definitions
        df = df.drop_duplicates(subset=['slang'], keep='first')
        
        # Create enhanced context for RAG (semantic understanding)
        df['rag_context'] = df.apply(_create_rag_context, axis=1)
        
        # Create variations for better morphological understanding
        df['variations'] = df['slang'].apply(_generate_slang_variations)
        
        print(f"✅ Processed dataset: {df.shape[0]} clean entries")
        print(f"📋 Sample entries:")
        print(df[['slang', 'definition', 'rag_context']].head(3))
        
        # Save processed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"💾 Saved to: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return None

def _create_rag_context(row):
    """Create rich context for RAG semantic understanding"""
    context_parts = []
    
    # Add the slang term itself
    context_parts.append(f"The slang term '{row['slang']}' means {row['definition']}")
    
    # Add example usage if available
    if row['example'] and len(str(row['example']).strip()) > 0:
        context_parts.append(f"Example usage: {row['example']}")
    
    # Add contextual notes if available
    if row['notes'] and len(str(row['notes']).strip()) > 0:
        context_parts.append(f"Context: {row['notes']}")
    
    # Add query variations for better semantic matching
    slang = row['slang']
    variations = [
        f"What does {slang} mean?",
        f"Define {slang}",
        f"Meaning of {slang}",
        f"{slang} definition"
    ]
    
    context_parts.extend(variations)
    
    return " | ".join(context_parts)

def _generate_slang_variations(slang_term):
    """Generate morphological variations of slang terms"""
    variations = [slang_term]
    
    # Common morphological patterns
    if len(slang_term) > 2:
        variations.extend([
            slang_term + 'ed',    # past tense: goat -> goated
            slang_term + 'ing',   # gerund: flex -> flexing
            slang_term + 's',     # plural: vibe -> vibes
            slang_term + 'er',    # agent: hate -> hater
            slang_term + 'y'      # adjective: salt -> salty
        ])
        
        # Handle words ending in 'e'
        if slang_term.endswith('e'):
            variations.extend([
                slang_term + 'd',           # vibe -> vibed
                slang_term[:-1] + 'ing'     # vibe -> vibing
            ])
    
    return ", ".join(variations)

if __name__ == '__main__':
    preprocess_slang_data("data/rag_slang_dataset.csv")
