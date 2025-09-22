# Gen Z Slang Normalization Engine

This repository contains a **Python-based software utility** designed for the translation and normalization of contemporary slang into standard English. As digital communication evolves, new linguistic forms emerge rapidly, often creating a gap in comprehension. This tool bridges that gap using a **Retrieval-Augmented Generation (RAG) model**, combined with semantic search and morphological analysis, to deliver precise and contextually appropriate interpretations of modern expressions.  

***

## ✨ Core Functionalities

- **Retrieval-Augmented Generation (RAG):**  
  Provides accurate definitions and standard English normalizations for slang terminology, grounding responses in a curated knowledge base.

- **Advanced Semantic Search:**  
  Employs sentence transformer models with a FAISS index for lightning-fast retrieval of relevant slang definitions, even for nuanced or obscure terms.

- **Morphological Analysis and Correction:**  
  Automatically corrects typographical errors and morphological variants (e.g., *“yeeting” → “yeet”*), improving accuracy in search and interpretation.

- **Systematic Data Preprocessing:**  
  Cleans, enriches, and structures slang data from public sources, optimizing RAG model performance.

- **Interactive Command-Line Interface (CLI):**  
  A simple, user-friendly CLI allows immediate slang translation and analysis without complex setup.

- **Modular Architecture:**  
  Decoupled modules for data processing, semantic indexing, morphology, and generation ensure easier maintenance and future extensibility.

***

## 🧠 NLP Techniques

- **RAG Framework:**  
  Combines information retrieval with text generation by first retrieving context from the slang knowledge base, then generating accurate and grounded outputs.

- **Semantic Search & Vector Embeddings:**  
  Converts slang terms into high-dimensional vectors using Sentence Transformers, enabling similarity search via FAISS.

- **Text Normalization & Preprocessing:**  
  Lowercasing, cleaning, and structuring ensure high-quality data pipelines for consistent model accuracy.

- **Morphological Analysis:**  
  Handles suffixes (*-ing, -ed*) and misspellings by identifying base forms, ensuring better matching and search recall.

***

## 🚀 Potential Applications

- **Content Moderation:** Detect hidden or coded language on social platforms.  
- **Market Research & Brand Monitoring:** Analyze Gen Z conversations for trends and sentiment insights.  
- **Chatbots & Virtual Assistants:** Enhance conversational AI with slang understanding.  
- **Digital Parenting & Safety:** Help parents decode online slang for safer communication.  
- **Linguistic Research & Education:** Support sociolinguistics research and non-native speakers learning modern slang.  

***

## ⚙️ System Architecture & Workflow

1. **Data Ingestion & Preprocessing (`data_process.py`):**  
   - Downloads dataset from Hugging Face  
   - Cleans and structures slang definitions  

2. **Morphological Correction (`morphology.py`):**  
   - Corrects spelling errors and normalizes morphological forms  

3. **Semantic Indexing & Search (`semantic_index.py`):**  
   - Uses Sentence Transformers + FAISS to find semantically similar slang terms  

4. **Contextual Retrieval & Text Generation (`rag_model.py`):**  
   - Retrieves context and generates normalized explanations using Ollama models  

5. **Application Entry Point (`main.py`):**  
   - Orchestrates modules and provides an interactive CLI  

***

## 🛠️ Setup & Execution

### Prerequisites
- Python **3.8+**  
- [Ollama](https://ollama.com/) installed locally with at least one model (e.g., `nemotron-mini:4b`, `Llama3`, `Mistral`)  

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/genz-slang-normalizer.git
cd genz-slang-normalizer

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application
```bash
python main.py
```

On first run, the system:  
- Downloads and preprocesses the slang dataset  
- Builds a FAISS semantic index  
- Creates `data/rag_slang_dataset.csv`  

***

## 💻 Usage Example

```text
--- Gen Z Slang Normalizer ---
Enter a slang term or sentence to normalize.
Type 'exit' to quit.

Enter slang > That new album slaps
🔄 Processing: 'That new album slaps'
✅ No corrections needed
🎯 Direct matches: ['slaps']
🤖 Generated: 'The term "slaps" means that the new album is excellent or very good.'
Normalized: The new album is excellent or very good.

Enter slang > he's goated
🔄 Processing: 'he's goated'
🔧 Corrections: 'goated' → 'goat'
✅ After morphology: 'he's goat'
🎯 Direct matches: ['goat']
🤖 Generated: '"Goated" is derived from "GOAT" (Greatest Of All Time). It means he's the absolute best in his field.'
Normalized: He is considered the greatest of all time.
```

***

## 📂 Project Structure

```
.
├── main.py              # Application entry point
├── rag_model.py         # RAG model logic
├── data_process.py      # Data ingestion & preprocessing
├── semantic_index.py    # FAISS semantic search index
├── morphology.py        # Typo & morphological correction
├── requirements.txt     # Dependencies
└── data/                # (Generated on first run)
    └── rag_slang_dataset.csv
```

***

## 🙌 Acknowledgements

Dataset sourced from [GenZ Slang Dataset](https://huggingface.co/datasets/MLBtrio/genz-slang-dataset) on Hugging Face.  
