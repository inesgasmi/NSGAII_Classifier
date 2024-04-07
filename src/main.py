from rule_generator import RuleGenerator
from nsga2 import NSGA2
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():

    # Path to the dataset
    dataset_path = "C:/Users/tim_o/PycharmProjects/NSGAII_Classifier/Data/german_credit_data.csv"

    dataset = pd.read_csv(dataset_path)

    X = dataset.drop(columns=['Risk'])
    y = dataset['Risk']

    # Path to save the generated rules
    output_path = "C:/Users/tim_o/PycharmProjects/NSGAII_Classifier/output/classification_rules.txt"

    # Split the dataset into X_train, X_val, y_train, y_val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize RuleGenerator with dataset and output paths
    rule_generator = RuleGenerator(dataset_path, output_path)

    # Generate classification rules using NSGA-II algorithm
    nsga2 = NSGA2(100, 50, dataset_path)  # Adjust parameters as needed
    nsga2.load_validation_data(X_val,y_val)
    classification_rules = nsga2.nsga2_algorithm()

    # Save the generated rules
    rule_generator.save_rules(output_path, classification_rules)

    # Plot evaluation metrics
    accuracies = [nsga2.rule_evaluator.evaluate_rule(rule)[0] for rule in classification_rules]
    plt.hist(accuracies, bins=10, edgecolor='black')
    plt.title('Distribution of Accuracy Scores')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    main()
