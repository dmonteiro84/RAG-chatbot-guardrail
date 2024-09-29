import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from bert_score import score as bertscore

# BERTScore: Measure semantic similarity using pre-trained transformer model
def compute_bertscore(response, fact):
    # Compute BERTScore (Precision, Recall, F1)
    P, R, F1 = bertscore([response], [fact], lang="en", verbose=True)
    return {"Precision": P.mean().item(), "Recall": R.mean().item(), "F1": F1.mean().item()}

# Precision and Recall: Measure factual correctness and completeness
def compute_precision_recall(response_facts, knowledge_base_facts):
    correct_facts = [fact for fact in response_facts if fact in knowledge_base_facts]
    precision = len(correct_facts) / len(response_facts) if response_facts else 0
    recall = len(correct_facts) / len(knowledge_base_facts) if knowledge_base_facts else 0
    return {"Precision": precision, "Recall": recall}

# Cosine Similarity: Measure semantic similarity using TF-IDF vectors
def compute_cosine_similarity(response, fact):
    vectorizer = TfidfVectorizer().fit_transform([response, fact])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

# Knowledge Grounding Ratio (KGR): Proportion of grounded facts in the response
def compute_kgr(response_facts, grounded_facts):
    return len(grounded_facts) / len(response_facts) if response_facts else 0