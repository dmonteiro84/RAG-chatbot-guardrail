import json
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_json(json_file):
    """
    Load the results from the specified JSON file.
    """
    with open(json_file, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def plot_histogram(values, title, xlabel):
    """
    Plot a histogram for given values with title and xlabel.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=10, color='blue', alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.show()

def analyze_results(data, evaluation_type):
    """
    Analyze the results from the evaluation:
    - For 'topic_filter', compute and display Confusion Matrix with True/False labels and percentages.
    - For 'groundedness', calculate the mean BERTScore and Cosine Similarity and plot histograms.
    - Also print Accuracy, Precision, Recall, and F1 Score for 'topic_filter'.
    """

    # Initialize variables for groundedness metrics
    bert_scores = {'Precision': [], 'Recall': [], 'F1': []}
    cosine_similarities = []

    if evaluation_type == "topic_filter":
        y_true = []
        y_pred = []

        for entry in data:
            # Ground truth and predicted outcome
            y_true.append(entry["not_banking_flag"])
            y_pred.append(entry["guardrails_out_of_scope_flag"])

        # Convert to NumPy arrays for easier processing
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Normalize the confusion matrix to display percentages
        conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum() * 100

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Display the confusion matrix with percentages
        plt.figure(figsize=(10, 8))  # Increased figure size for better visibility
        sns.heatmap(conf_matrix_percent, annot=True, fmt=".2f", cmap="Blues", 
                    xticklabels=["True", "False"], yticklabels=["True", "False"],
                    annot_kws={"size": 14})  # Increased annotation font size

        # Increase font sizes for title and labels
        plt.title("Confusion Matrix (Percentages)", fontsize=18)
        plt.xlabel("Predicted", fontsize=16)
        plt.ylabel("Actual", fontsize=16)

        # Increase the font size for the tick labels on both axes
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.show()

        # Print the performance metrics
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

    elif evaluation_type == "groundedness":
        for entry in data:
            bert_score = entry["BERTScore"]
            cosine_similarity = entry["Cosine Similarity"]

            # Append BERTScore values
            bert_scores['Precision'].append(bert_score['Precision'])
            bert_scores['Recall'].append(bert_score['Recall'])
            bert_scores['F1'].append(bert_score['F1'])

            # Append cosine similarity value
            cosine_similarities.append(cosine_similarity)

        # Calculate and print mean BERTScore and Cosine Similarity
        mean_bert_score = {metric: np.mean(bert_scores[metric]) for metric in bert_scores}
        mean_cosine_similarity = np.mean(cosine_similarities)

        print("Mean BERTScore:")
        print(f"Precision: {mean_bert_score['Precision']:.2f}")
        print(f"Recall: {mean_bert_score['Recall']:.2f}")
        print(f"F1 Score: {mean_bert_score['F1']:.2f}")
        
        print(f"Mean Cosine Similarity: {mean_cosine_similarity:.2f}")

        # Plot histograms for BERTScore F1 and Cosine Similarity
        plot_histogram(bert_scores['F1'], 'BERTScore F1 Distribution', 'BERTScore F1')
        plot_histogram(cosine_similarities, 'Cosine Similarity Distribution', 'Cosine Similarity')

def main():
    parser = argparse.ArgumentParser(description="Analyze the evaluation results JSON file")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the JSON file containing evaluation results")
    parser.add_argument("--type", type=str, required=True, choices=["topic_filter", "groundedness"], help="Type of evaluation: 'topic_filter' or 'groundedness'")
    
    args = parser.parse_args()

    # Load data from the JSON file
    data = load_json(args.json_file)
    
    # Analyze the results based on the evaluation type
    analyze_results(data, args.type)

if __name__ == "__main__":
    main()
