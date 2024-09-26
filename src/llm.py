import openai
import os
from src.retrieval import FaissRetriever
from src.guardrail import Guardrails

class OpenAIGPTWithGuardrails:
    def __init__(self, retriever, guardrails):
        """
        Initialize the OpenAIGPTWithGuardrails class using the FAISS retriever and Guardrails for validation.
        :param retriever: An instance of the FaissRetriever class.
        :param guardrails: An instance of the Guardrails class for validating responses.
        """
        self.retriever = retriever
        self.guardrails = guardrails

    def retrieve_and_generate_response(self, user_query, k=3):
        """
        Retrieve relevant products using FAISS and generate a response with GPT-4. Validate the response using Guardrails.

        :param user_query: The user's query (e.g., "Tell me about credit cards").
        :param k: The number of products to retrieve.
        :return: A GPT-4 generated and validated response.
        """
        # Step 1: Retrieve relevant products from the knowledge base using FAISS
        retrieved_products = self.retriever.search(user_query, k)

        # Step 2: Format the retrieved products into a text prompt for GPT-4
        product_info = "\n".join([
            f"Product: {p['product_name']}\nDescription: {p['description']}\n" for p in retrieved_products
        ])

        # Create the final prompt for GPT-4
        prompt = f"""
           The user asked: '{user_query}'. Based on the following product information, provide a helpful response in JSON format:
           {product_info}
           """

        # Step 3: Use OpenAI GPT-4 to generate a response based on the prompt
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )

        # Step 4: Extract the generated response
        generated_response = response.choices[0].message.content.strip()

        # Step 5: Validate the response using Guardrails
        validation_outcome = self.guardrails.validate_response(generated_response)

        # Step 6: Build the formatted response
        formatted_response = self.format_validation_outcome(validation_outcome)

        return formatted_response
    
    def format_validation_outcome(self, validation_outcome):
        """
        Formats the validation outcome in a user-friendly and structured manner, similar to the format specified.

        :param validation_outcome: The validation outcome object from Guardrails.
        :return: A formatted string representation of the validation outcome.
        """
        result = []

        # Formatting raw_llm_output
        result.append(f"# raw_llm_output:\n{validation_outcome.raw_llm_output}")

        # Formatting validation_summaries
        result.append(f"# validation_summaries: {validation_outcome.validation_summaries}")

        # Formatting validated_output
        result.append(f"# validated_output: {validation_outcome.validated_output}")

        # Formatting reask
        if validation_outcome.reask:
            result.append(f"# reask - incorrect_value: {validation_outcome.reask.incorrect_value}")
            for i, fail_result in enumerate(validation_outcome.reask.fail_results, start=1):
                result.append(f"# reask - fail_results[{i}] - outcome: {fail_result.outcome}")
                result.append(f"# reask - fail_results[{i}] - error_message: {fail_result.error_message}")
                result.append(f"# reask - fail_results[{i}] - fix_value: {fail_result.fix_value}")
                result.append(f"# reask - fail_results[{i}] - error_spans: {fail_result.error_spans}")
                result.append(f"# reask - fail_results[{i}] - metadata: {fail_result.metadata}")
                result.append(f"# reask - fail_results[{i}] - validated_chunk: {fail_result.validated_chunk}")
        else:
            result.append(f"# reask: None")

        # Formatting validation_passed
        result.append(f"# validation_passed: {validation_outcome.validation_passed}")

        # Formatting error
        result.append(f"# error: {validation_outcome.error}")

        return "\n".join(result)

# Example usage for testing (Optional):
if __name__ == "__main__":
    # Initialize the FAISS retriever
    retriever = FaissRetriever('data/products.json')  # Ensure you have the right knowledge base path

    # Example user query
    user_query = "Tell me about the bank's credit cards."

    # Initialize the OpenAIGPTWithGuardrails class
    guardrails = Guardrails()  # Initialize guardrails instance
    gpt_with_guardrails = OpenAIGPTWithGuardrails(retriever, guardrails)

    # Generate and print the response with Guardrails
    validated_response = gpt_with_guardrails.retrieve_and_generate_response(user_query)
    print("Response with Guardrails:", validated_response)
