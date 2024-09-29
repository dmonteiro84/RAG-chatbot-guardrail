import openai
import os
from src.retrieval import FaissRetriever
from src.guardrail import Guardrails
import json

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

        # Step 3: Create a controlled prompt for GPT-4, including explicit instructions and an example of the expected format
        prompt = f"""
            You are a helpful assistant providing information about bank products.
            The user asked: '{user_query}'.

            If the user query is related to personal loans, home loans, credit cards, or other banking products, provide a helpful response in the following JSON format:

            {{
                "response": "Your helpful response text based on the products retrieved."
            }}

            Example output:
            {{
                "response": "Sure. Here are some details about [Product Name]: 
                            [Description].
                            And about [Product Name]:
                            [Description]."
            }},
            {{
                "response": "Absolutely! The Bank provides a range of products. Our Credit Cards come with great rewards programs and 0% interest on balance transfers for the first year. We also offer Personal Loans with fixed interest rates and customizable repayment options. And for savers, our Savings Accounts offer 1.25% interest with no monthly fees. Feel free to ask for more details on any of these."
            }}

            If the user query is not related to these products, respond with:
            {{
                "response": "I'm sorry, I can only provide information related to personal loans, home loans, credit cards, or other banking products."
            }}

            Product information:
            {product_info}
            """

        # Step 4: Use OpenAI GPT-4 to generate a response based on the prompt
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        # Step 5: Extract the generated response
        raw_response = response.choices[0].message.content.strip()

        # Step 6: Pass the string response to Guardrails for validation
        validation_outcome = self.guardrails.validate_response(raw_response)

        # Step 7: Build the formatted response for output
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
