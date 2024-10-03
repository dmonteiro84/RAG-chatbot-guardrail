import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bert_score import score as bertscore
from datasets import load_dataset
from datetime import datetime
import json
import os
from tqdm import tqdm
from src.llm import OpenAIGPTWithGuardrails
from src.guardrail import Guardrails
from src.retrieval import FaissRetriever
import numpy as np
import argparse
from langdetect import detect, LangDetectException 

# Load the SentenceTransformer model for embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to compute BERTScore
def compute_bertscore(response, fact):
    P, R, F1 = bertscore([response], [fact], lang="en", verbose=False)
    return {"Precision": P.mean().item(), "Recall": R.mean().item(), "F1": F1.mean().item()}

# Function to compute cosine similarity using BERT embeddings
def compute_cosine_similarity_bert(response, fact):
    response_embedding = model.encode([response], convert_to_tensor=True)
    fact_embedding = model.encode([fact], convert_to_tensor=True)
    cosine_sim = cosine_similarity(response_embedding.cpu().numpy(), fact_embedding.cpu().numpy())
    return cosine_sim[0][0]

# Function to append results to a JSON file, line by line
def append_result_to_json(result, output_file):
    result = json.loads(json.dumps(result, default=lambda o: float(o) if isinstance(o, (torch.Tensor, np.float32, np.float64)) else o))
    
    with open(output_file, 'a') as f:
        json.dump(result, f)
        f.write('\n')

# Load or download datasets and cache them locally
def load_or_download_dataset(name, cache_dir='data'):
    dataset_path = os.path.join(cache_dir, name)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    if not os.path.exists(dataset_path):
        print(f"Downloading {name} dataset...")
        dataset = load_dataset(name, cache_dir=cache_dir)
    else:
        print(f"Loading {name} dataset from cache...")
        dataset = load_dataset(name, cache_dir=cache_dir)
    return dataset

# Load non-English questions from the specified JSON file
def load_non_english_questions(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
# Function to validate the input language (moved here from main.py)
def validate_language(user_query):
    try:
        language = detect(user_query)
        if language != 'en':
            return False, "Please provide your query in English."
        else:
            return True, None
    except LangDetectException:
        return False, "Could not detect the language. Please try again."

# Function to run evaluation on the questions
def run_language_evaluation(chatbot_system, questions, output_file, retriever, is_non_english):
    timestamp = datetime.now().isoformat()

    print(f"Processing {'non-English' if is_non_english else 'English'} questions:")
    for question in tqdm(questions):
        if is_non_english:
            query = question["questions"]  # Non-English dataset
            not_english_actual = True  # Mark as non-English
        else:
            query = question  # Banking77 (English dataset)
            not_english_actual = False  # Mark as English

        response = chatbot_system.retrieve_and_generate_response(query)
        not_english_predicted = "Please provide your query in English." in response
        fact = "English placeholder fact" if is_non_english else retriever.retrieve_fact(query)

        bertscore_result = compute_bertscore(response, fact)
        cosine_similarity_result = compute_cosine_similarity_bert(response, fact)

        result = {
            "timestamp": timestamp,
            "dataset_name": "Custom Non-English Questions" if is_non_english else "Banking77",
            "query": query,
            "raw_llm_output": response,
            "BERTScore": bertscore_result,
            "Cosine Similarity": cosine_similarity_result,
            "not_english_actual": not_english_actual,  # Set correctly based on dataset
            "not_english_predicted": not_english_predicted
        }

        append_result_to_json(result, output_file)

# Function to evaluate groundedness of custom questions
def evaluate_groundedness(chatbot_system, questions_data, output_file, retriever):
    timestamp = datetime.now().isoformat()

    print("Processing groundedness questions:")
    for category_data in tqdm(questions_data):
        category = category_data['category']
        for question in category_data['questions']:
            query = question
            response = chatbot_system.retrieve_and_generate_response(query)
            fact = retriever.retrieve_fact(query)

            bertscore_result = compute_bertscore(response, fact)
            cosine_similarity_result = compute_cosine_similarity_bert(response, fact)

            result = {
                "timestamp": timestamp,
                "dataset_name": "Custom Questions",
                "query": query,
                "raw_llm_output": response,
                "BERTScore": bertscore_result,
                "Cosine Similarity": cosine_similarity_result,
            }

            append_result_to_json(result, output_file)

# Function to evaluate topic filtering
def evaluate_guardrails_on_datasets(chatbot_system, in_scope_dataset, out_of_scope_dataset, output_file, retriever):
    timestamp = datetime.now().isoformat()

    in_scope_queries = list(set(in_scope_dataset['train']['text']))[:500]
    out_of_scope_queries = list(set(out_of_scope_dataset['train']['question']))[:500]

    print("Processing in-scope queries:")
    for query in tqdm(in_scope_queries):
        response = chatbot_system.retrieve_and_generate_response(query)
        fact = retriever.retrieve_fact(query)
        guardrail_message = "I'm sorry, I can only provide information related to personal loans, home loans, credit cards, or other banking products."
        guardrails_out_of_scope_flag = guardrail_message in response
        
        bertscore_result = compute_bertscore(response, fact)
        cosine_similarity_result = compute_cosine_similarity_bert(response, fact)

        result = {
            "timestamp": timestamp,
            "dataset_name": "Banking77",
            "query": query,
            "raw_llm_output": response,
            "BERTScore": bertscore_result,
            "Cosine Similarity": cosine_similarity_result,
        }

        append_result_to_json(result, output_file)

    print("Processing out-of-scope queries:")
    for query in tqdm(out_of_scope_queries):
        response = chatbot_system.retrieve_and_generate_response(query)
        fact = "Non-banking example fact."

        bertscore_result = compute_bertscore(response, fact)
        cosine_similarity_result = compute_cosine_similarity_bert(response, fact)

        result = {
            "timestamp": timestamp,
            "dataset_name": "WikiQA",
            "query": query,
            "raw_llm_output": response,
            "BERTScore": bertscore_result,
            "Cosine Similarity": cosine_similarity_result,
        }

        append_result_to_json(result, output_file)

def main():
    parser = argparse.ArgumentParser(description="Run evaluation of chatbot system.")
    parser.add_argument("-t", "--type", required=True, choices=["topic_filter", "groundedness", "language_check"], help="Specify evaluation type: 'topic_filter', 'groundedness', or 'language_check'.")
    
    args = parser.parse_args()

    # Initialize FAISS retriever and Guardrails
    retriever = FaissRetriever("data/products.json")
    guardrails = Guardrails("specs/guardrail_spec.xml")

    # Initialize chatbot system
    chatbot_system = OpenAIGPTWithGuardrails(retriever, guardrails)

    # Create unique filename with timestamp and chosen type
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"evaluation_results_{args.type}_{timestamp}.json"
    
    if args.type == "topic_filter":
        banking77 = load_or_download_dataset('banking77')
        wiki_qa = load_or_download_dataset('wiki_qa')
        evaluate_guardrails_on_datasets(chatbot_system, banking77, wiki_qa, output_file, retriever)
    elif args.type == "groundedness":
        with open("data/questions.json", 'r') as f:
            questions_data = json.load(f)
        evaluate_groundedness(chatbot_system, questions_data, output_file, retriever)
    elif args.type == "language_check":
        non_english_questions = load_non_english_questions('data/non_english_questions.json')
        banking77 = load_or_download_dataset('banking77')
        banking77_questions = list(set(banking77['train']['text']))[:100]

        # Evaluate non-English questions
        run_language_evaluation(chatbot_system, non_english_questions, output_file, retriever, is_non_english=True)

        # Evaluate Banking77 English questions
        run_language_evaluation(chatbot_system, banking77_questions, output_file, retriever, is_non_english=False)

if __name__ == "__main__":
    main()