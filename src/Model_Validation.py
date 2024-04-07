from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class ModelValidation:
    @staticmethod
    def plot_evaluation_metrics(nsga2, classification_rules):
        accuracies = [nsga2.rule_evaluator.accuracy(rule) for rule in classification_rules]
        precisions = [nsga2.rule_evaluator.precision(rule) for rule in classification_rules]
        recalls = [nsga2.rule_evaluator.recall(rule) for rule in classification_rules]
        f1_scores = [nsga2.rule_evaluator.f1_score(rule) for rule in classification_rules]

        # Plotting code for accuracies
        plt.hist(accuracies, bins=10, edgecolor='black')
        plt.title('Distribution of Accuracy Scores')
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.show()

        # Plotting code for precisions
        plt.hist(precisions, bins=10, edgecolor='black')
        plt.title('Distribution of Precision Scores')
        plt.xlabel('Precision')
        plt.ylabel('Frequency')
        plt.show()

        # Plotting code for recalls
        plt.hist(recalls, bins=10, edgecolor='black')
        plt.title('Distribution of Recall Scores')
        plt.xlabel('Recall')
        plt.ylabel('Frequency')
        plt.show()

        # Plotting code for F1 scores
        plt.hist(f1_scores, bins=10, edgecolor='black')
        plt.title('Distribution of F1 Scores')
        plt.xlabel('F1 Score')
        plt.ylabel('Frequency')
        plt.show()