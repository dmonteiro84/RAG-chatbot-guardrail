from src.retrieval import FaissRetriever
from src.llm import OpenAIGPTWithGuardrails
from src.guardrail import Guardrails
from src.evaluation import compute_bertscore, compute_cosine_similarity_bert, validate_language  # Import validate_language
from bert_score import score as bertscore

# Function to preload BERTScore model and tokenizer for evaluation metrics
def preload_resources():
    print("Preloading BERTScore model and tokenizer...")
    # This will load the BERTScore model in advance
    bertscore(["preload"], ["preload"], lang="en", verbose=False)  # Dummy data to trigger model loading
    print("Resources preloaded.")

def main():
    # Preload libraries
    preload_resources()

    # Start of the program
    print("Starting program...")

    # Initialize FAISS retriever
    try:
        print("Initializing FAISS retriever...")
        retriever = FaissRetriever("data/products.json")
        print("FAISS retriever initialized successfully.")
    except Exception as e:
        print(f"Error initializing FAISS retriever: {e}")
        return

    # Set up Guardrails
    try:
        print("Setting up Guardrails...")
        guardrails = Guardrails("specs/guardrail_spec.xml")
        print("Guardrails setup complete.")
    except Exception as e:
        print(f"Error setting up Guardrails: {e}")
        return

    # Initialize the LLM with Guardrails
    print("Initializing LLM with Guardrails...")
    try:
        gpt_with_guardrails = OpenAIGPTWithGuardrails(retriever, guardrails)
        print("LLM with Guardrails initialized.")
    except Exception as e:
        print(f"Error initializing LLM with Guardrails: {e}")
        return

    # Start conversation loop
    while True:
        user_query = input("Please enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Exiting the program...")
            break

        # Validate the language of the query
        is_valid_language, error_message = validate_language(user_query)
        if not is_valid_language:
            print(error_message)
            continue

        try:
            print("Generating response with Guardrails...")
            response_with_guardrails = gpt_with_guardrails.retrieve_and_generate_response(user_query)
            print(f"Response with Guardrails:\n{response_with_guardrails}")

            response_facts = [response_with_guardrails]
            knowledge_base_facts = ["The Bank offers home loans with competitive interest rates and flexible repayment options.", 
                                    "The Bank offers flexible personal loans with fixed rates up to $50,000."]

            # Compute BERTScore
            bertscore_result = compute_bertscore(response_with_guardrails, knowledge_base_facts[0])
            print("BERTScore:", bertscore_result)

            # Compute Cosine Similarity using BERT embeddings
            cosine_similarity_result = compute_cosine_similarity_bert(response_with_guardrails, knowledge_base_facts[0])
            print("Cosine Similarity:", cosine_similarity_result)

        except Exception as e:
            print(f"Error generating response with Guardrails: {e}")
            continue

if __name__ == "__main__":
    main()
