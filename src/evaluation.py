import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bert_score import score as bertscore

# Load the SentenceTransformer model for embeddings (e.g., 'paraphrase-MiniLM-L6-v2')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# BERTScore: Measure semantic similarity using pre-trained transformer model
def compute_bertscore(response, fact):
    # Compute BERTScore (Precision, Recall, F1)
    P, R, F1 = bertscore([response], [fact], lang="en", verbose=True)
    return {"Precision": P.mean().item(), "Recall": R.mean().item(), "F1": F1.mean().item()}

# Cosine Similarity using BERT embeddings
def compute_cosine_similarity_bert(response, fact):
    # Generate embeddings for both the response and the fact
    response_embedding = model.encode([response], convert_to_tensor=True)
    fact_embedding = model.encode([fact], convert_to_tensor=True)

    # Compute cosine similarity between the two embeddings
    cosine_sim = cosine_similarity(response_embedding.cpu().numpy(), fact_embedding.cpu().numpy())
    
    # Return the cosine similarity score
    return cosine_sim[0][0]

# Knowledge Grounding Ratio (KGR): Proportion of grounded facts in the response
def compute_kgr(response_facts, grounded_facts):
    return len(grounded_facts) / len(response_facts) if response_facts else 0

# Example usage
if __name__ == "__main__":
    response = "The Bank offers flexible home loans with competitive interest rates."
    fact = "The Bank offers flexible home loans with competitive rates and repayment options."

    # Compute BERTScore
    bertscore_result = compute_bertscore(response, fact)
    print(f"BERTScore: {bertscore_result}")

    # Compute Cosine Similarity using BERT embeddings
    cosine_similarity_result = compute_cosine_similarity_bert(response, fact)
    print(f"Cosine Similarity (BERT): {cosine_similarity_result}")

    # Compute Knowledge Grounding Ratio (KGR)
    response_facts = ["The Bank offers flexible home loans", "interest rates starting at 3.49%"]
    knowledge_base_facts = ["The Bank offers flexible home loans", "maximum loan amount up to $1,000,000"]
    kgr_result = compute_kgr(response_facts, knowledge_base_facts)
    print(f"KGR: {kgr_result}")
