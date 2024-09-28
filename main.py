from src.retrieval import FaissRetriever
from src.llm import OpenAIGPTWithGuardrails
from src.guardrail import Guardrails

def main():
    # Start of the program
    print("Starting program...")

    # 1. Initialize the FAISS retriever (Knowledge Base)
    try:
        print("Initializing FAISS retriever...")
        retriever = FaissRetriever("data/products.json")
        print("FAISS retriever initialized successfully.")
    except Exception as e:
        print(f"Error initializing FAISS retriever: {e}")
        return

    # 2. Set up Guardrails (validation rules)
    try:
        print("Setting up Guardrails...")
        guardrails = Guardrails("specs/guardrail_spec.xml")
        print("Guardrails setup complete.")
    except Exception as e:
        print(f"Error setting up Guardrails: {e}")
        return

    # 3. Initialize the LLM with Guardrails
    print("Initializing LLM with Guardrails...")
    try:
        gpt_with_guardrails = OpenAIGPTWithGuardrails(retriever, guardrails)
        print("LLM with Guardrails initialized.")
    except Exception as e:
        print(f"Error initializing LLM with Guardrails: {e}")
        return

    # 4. Take user input for a query
    user_query = input("Please enter your query: ")

    # 5. Generate response with guardrails
    try:
        print("Generating response with Guardrails...")
        response_with_guardrails = gpt_with_guardrails.retrieve_and_generate_response(user_query)
        print(f"Response with Guardrails:\n{response_with_guardrails}")
    except Exception as e:
        print(f"Error generating response with Guardrails: {e}")
        return

    # Skipping evaluation for now
    print("Skipping evaluation...")

if __name__ == "__main__":
    main()
