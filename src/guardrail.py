import guardrails as gd
import os

class Guardrails:
    def __init__(self, spec_path="specs/guardrail_spec.xml"):
        """
        Initialize the Guardrails class with a spec file for grounded responses.
        :param spec_path: The path to the guardrail XML file.
        """
        # Check if the XML spec file exists
        if not os.path.exists(spec_path):
            raise FileNotFoundError(f"Guardrail specification file not found: {spec_path}")
        
        # Read the specification content from the file
        with open(spec_path, 'r') as f:
            spec_content = f.read()

        # Use from_rail_string to load the specification as a string
        self.guard = gd.Guard.from_rail_string(spec_content)

    def validate_response(self, response):
        """
        Validate the LLM response using Guardrails.
        :param response: The chatbot response to validate
        :return: Validated response or raise an error if validation fails
        """
        validated_response = self.guard.parse(response)
        return validated_response
