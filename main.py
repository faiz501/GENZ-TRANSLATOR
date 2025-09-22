import os
from data_process import preprocess_slang_data
from rag_model import SimpleRAG

def main():
    """
    Main function to run the Gen Z Slang Normalizer.
    """
    # Setup paths
    DATA_DIR = "data"
    RAG_DATA_PATH = os.path.join(DATA_DIR, "rag_slang_dataset.csv")
    
    # Check if data needs preprocessing
    if not os.path.exists(RAG_DATA_PATH):
        print("RAG data file not found. Starting preprocessing...")
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        # Preprocess the data
        result = preprocess_slang_data(RAG_DATA_PATH)
        
        if result is None:
            print("Could not process data. Exiting.")
            return
        print("Preprocessing completed successfully!")
    else:
        print("RAG data file found. Skipping preprocessing.")
    
    # Initialize the RAG model
    try:
        rag_model = SimpleRAG(rag_data_path=RAG_DATA_PATH)
    except Exception as e:
        print(f"Error initializing RAG model: {e}")
        return
    
    # Interactive loop
    print("\n--- Gen Z Slang Normalizer ---")
    print("Enter a slang term or sentence to normalize.")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nEnter slang > ")
        if user_input.lower() == 'exit':
            break
        
        try:
            response = rag_model.normalize(user_input)
            print(f"Normalized: {response}")
        except Exception as e:
            print(f"Error processing input: {e}")

if __name__ == '__main__':
    main()
