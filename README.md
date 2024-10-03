
# LLM Chatbot with Guardrails

## Installation

### Prerequisites
- **Python 3.8+** installed
- **OpenAI API Key** (for using GPT-4 via OpenAI)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/LLM-Chatbot-Guardrails.git
   cd LLM-Chatbot-Guardrails
   ```

2. **Install dependencies**:
   Set up a virtual environment and install the dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key**:
   Add your OpenAI API key to the environment variables:
   
   ```bash
   export OPENAI_API_KEY='your_openai_api_key'   # On Windows: set OPENAI_API_KEY=your_openai_api_key
   ```

---

## Usage

### 1. **Running the LLM-Powered Chatbot with Guardrails**

After setting up the environment, you can run the chatbot with the LLM guardrails to validate responses.

```bash
python src/llm.py
```

This will prompt for a user query. The chatbot will respond, and the guardrails will validate the response for groundedness, topic relevancy, length, and language.

### Example:

```
Please enter your query: What is a home loan?
Generating response with Guardrails...
Response with Guardrails:
# raw_llm_output:
"A home loan is a type of loan where you borrow money to purchase a house. The loan is secured against your property, and you can repay it over time with interest."
# validation_passed: True
# ...
```

### 2. **Running Evaluations**

The evaluation process tests the chatbot’s response groundedness, topic relevancy, and language check using datasets. Here are the commands to run different evaluations:

- Language Check:
```bash
python -m src.evaluation -t language_check
```

- Topic Filter:
```bash
python -m src.evaluation -t topic_filter
```

- Groundedness Check:
```bash
python -m src.evaluation -t groundedness
```

### 3. **Running Analysis on Evaluation Results**
After running evaluations, you can analyze the results using the analysis tool. Here’s an example command to analyze a groundedness evaluation JSON file:
```bash
python -m src.analysis --json_file evaluation_results_groundedness_20241002_225734.json --type groundedness
```

---

## Configuration

### Guardrails XML

The guardrail configuration is defined in the `src/guardrail.xml` file, which contains the following validations:
- **Response length**: Ensures responses don't exceed 300 characters.
- **Whitelist topic validation**: The chatbot is restricted to discussing only the following products:
  - Credit cards
  - Home loans
  - Personal loans
  - Savings accounts
- **English language validation**: Ensures both the input and output are in English.

You can customize the guardrails by modifying this XML file to add or change allowed products, adjust the response length, or add more validation rules.

---

## Performance Evaluation

To ensure the guardrails work effectively, we have provided an evaluation dataset in `dataset/evaluation_set.json`. This dataset includes:
- Allowed queries related to the bank's products.
- Disallowed queries about unrelated topics (e.g., food, travel, etc.).

---

## Design Considerations

1. **Groundedness**:
   - Responses are grounded in factual product information from the knowledge base. Only allowed topics (credit cards, home loans, etc.) are permitted.
   
2. **Language Control**:
   - The chatbot restricts communication to English only, preventing any multi-language confusion.

3. **Reliability**:
   - Tests are provided to ensure consistent behavior across various input queries.
   
4. **Modularity**:
   - The guardrails system is modular, with XML-based configurations for easy future updates.

---

## Limitations

- **Hallucinations**: While the guardrails block off-topic responses, they might not catch subtle hallucinations that still mention the allowed topics but with inaccurate information.
- **Complex Queries**: Highly complex or ambiguous queries might pass through the guardrails without sufficient validation if the product knowledge base lacks depth.
- **English Only**: Currently, the chatbot only handles English queries and responses.
- **No context/chat history**: Currently, the chatbot can not remember what the user said before. Hence, contextual questions can not be tested.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.