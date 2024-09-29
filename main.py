from langdetect import detect, LangDetectException  # Import langdetect for language detection
from src.retrieval import FaissRetriever
from src.llm import OpenAIGPTWithGuardrails
from src.guardrail import Guardrails

# Function to validate the input language
def validate_language(user_query):
    try:
        # Detect the language of the input
        language = detect(user_query)
        if language != 'en':
            # If the language is not English, return False with an error message
            return False, "Please provide your query in English."
        else:
            return True, None
    except LangDetectException:
        return False, "Could not detect the language. Please try again."

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

    # Start conversation loop
    while True:
        # 4. Take user input for a query
        user_query = input("Please enter your query (or type 'exit' to quit): ")

        # Check if the user wants to exit
        if user_query.lower() == "exit":
            print("Exiting the program...")
            break

        # Step 4.1: Validate the input language
        is_valid_language, error_message = validate_language(user_query)
        if not is_valid_language:
            print(error_message)
            continue  # Ask for a new query

        # 5. Generate response with guardrails
        try:
            print("Generating response with Guardrails...")
            response_with_guardrails = gpt_with_guardrails.retrieve_and_generate_response(user_query)
            print(f"Response with Guardrails:\n{response_with_guardrails}")
        except Exception as e:
            print(f"Error generating response with Guardrails: {e}")
            continue  # Continue the loop and ask for another query

    print("Conversation ended.")

if __name__ == "__main__":
    main()
