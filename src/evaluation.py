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
    # Convert any non-serializable objects (like torch.Tensor or float32) to float
    result = json.loads(json.dumps(result, default=lambda o: float(o) if isinstance(o, (torch.Tensor, np.float32, np.float64)) else o))
    
    with open(output_file, 'a') as f:
        json.dump(result, f)
        f.write('\n')

# Load or download datasets and cache them locally
def load_or_download_dataset(name, cache_dir='data'):
    # Check if the dataset exists in the cache_dir
    dataset_path = os.path.join(cache_dir, name)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    if not os.path.exists(dataset_path):
        # If dataset is not cached, download it
        print(f"Downloading {name} dataset...")
        dataset = load_dataset(name, cache_dir=cache_dir)
    else:
        # If dataset is cached, load it from local storage
        print(f"Loading {name} dataset from cache...")
        dataset = load_dataset(name, cache_dir=cache_dir)
    return dataset

# Function to evaluate true positives, true negatives, false positives, and false negatives
def evaluate_guardrails_on_datasets(chatbot_system, in_scope_dataset, out_of_scope_dataset, output_file, retriever):
    tp, tn, fp, fn = 0, 0, 0, 0
    guardrail_message = "I'm sorry, I can only provide information related to personal loans, home loans, credit cards, or other banking products."
    
    # Timestamp when the evaluation started
    timestamp = datetime.now().isoformat()

    # Sample 500 unique in-scope queries
    in_scope_queries = list(set(in_scope_dataset['train']['text']))[:500]
    
    # Sample 500 unique out-of-scope queries
    out_of_scope_queries = list(set(out_of_scope_dataset['train']['question']))[:500]

    print("Processing in-scope queries:")
    for query in tqdm(in_scope_queries):
        response = chatbot_system.retrieve_and_generate_response(query)
        fact = retriever.retrieve_fact(query)  # Retrieve the actual fact from the knowledge base
        not_banking_flag = False
        guardrails_out_of_scope_flag = guardrail_message in response
        
        bertscore_result = compute_bertscore(response, fact)
        cosine_similarity_result = compute_cosine_similarity_bert(response, fact)

        result = {
            "timestamp": timestamp,
            "dataset_name": "Banking77",
            "query": query,
            "raw_llm_output": response,
            "validation_summaries": None,
            "validated_output": response,
            "reask": None,
            "validation_passed": not guardrails_out_of_scope_flag,
            "error": None,
            "BERTScore": bertscore_result,
            "Cosine Similarity": cosine_similarity_result,
            "not_banking_flag": not_banking_flag,
            "guardrails_out_of_scope_flag": guardrails_out_of_scope_flag
        }

        append_result_to_json(result, output_file)

    print("Processing out-of-scope queries:")
    for query in tqdm(out_of_scope_queries):
        response = chatbot_system.retrieve_and_generate_response(query)
        fact = "Non-banking example fact."  # Placeholder fact for out-of-scope questions
        not_banking_flag = True
        guardrails_out_of_scope_flag = guardrail_message in response
        
        bertscore_result = compute_bertscore(response, fact)
        cosine_similarity_result = compute_cosine_similarity_bert(response, fact)

        result = {
            "timestamp": timestamp,
            "dataset_name": "WikiQA",
            "query": query,
            "raw_llm_output": response,
            "validation_summaries": None,
            "validated_output": response,
            "reask": None,
            "validation_passed": guardrails_out_of_scope_flag,
            "error": None,
            "BERTScore": bertscore_result,
            "Cosine Similarity": cosine_similarity_result,
            "not_banking_flag": not_banking_flag,
            "guardrails_out_of_scope_flag": guardrails_out_of_scope_flag
        }

        append_result_to_json(result, output_file)

# Main function for evaluation
def main():
    # Load or download the datasets
    banking77 = load_or_download_dataset('banking77')
    wiki_qa = load_or_download_dataset('wiki_qa')

    # Initialize FAISS retriever and Guardrails
    retriever = FaissRetriever("data/products.json")
    guardrails = Guardrails("specs/guardrail_spec.xml")

    # Initialize chatbot system
    chatbot_system = OpenAIGPTWithGuardrails(retriever, guardrails)

    # Create unique filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"evaluation_results_{timestamp}.json"
    
    # Evaluate the datasets and write results to JSON
    evaluate_guardrails_on_datasets(chatbot_system, banking77, wiki_qa, output_file, retriever)

if __name__ == "__main__":
    main()
