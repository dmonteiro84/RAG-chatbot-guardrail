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

def analyze_results(data):
    """
    Analyze the results from the evaluation, including:
    - Confusion Matrix with True/False labels and percentages.
    - Accuracy, Precision, Recall, and F1 Score.
    """
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

def main():
    parser = argparse.ArgumentParser(description="Analyze the evaluation results JSON file")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the JSON file containing evaluation results")
    
    args = parser.parse_args()
    json_file = args.json_file

    # Load data from the JSON file
    data = load_json(json_file)
    
    # Analyze the results
    analyze_results(data)

if __name__ == "__main__":
    main()
