from sklearn.metrics import accuracy_score
import pandas as pd

class RuleEvaluator:
    def __init__(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path, index_col=0)
        # print("Dataset columns: ", self.dataset.columns)
    def get_num_features(self):
        """
        Get the number of features in the dataset.

        Returns:
            int: Number of features.
        """
        return len(self.dataset.columns) - 2

    def evaluate_rule(self, rule):
        """
        Evaluate a rule based on accuracy and complexity.

        Parameters:
            rule (list): Rule to be evaluated.

        Returns:
            tuple: Accuracy and complexity of the rule.
        """
        predicted_labels = self.apply_rule(rule)
        true_labels = self.dataset.iloc[:, -1]  # Assuming the last column is the label column
        predicted_labels = ['bad' if label == 0 else 'good' for label in predicted_labels]
        accuracy = accuracy_score(true_labels, predicted_labels)
        return accuracy, None  # Returning None for complexity as it's not relevant here

    def apply_rule(self, rule):
        """
        Apply a rule to the dataset and return predicted labels.

        Parameters:
            rule (list): Rule to be applied.

        Returns:
            list: Predicted labels.
        """
        #print("Applying rule:", rule)
        predicted_labels = []
        for _, row in self.dataset.iterrows():
            if self.satisfy_rule(row, rule):
                predicted_labels.append(1)  # Assigning a label value (e.g., 1) for demonstration
            else:
                predicted_labels.append(0)  # Assigning a label value (e.g., 0) for demonstration
        return predicted_labels

    def satisfy_rule(self, instance, rule):
        """
        Check if an instance satisfies a rule.

        Parameters:
            instance (pd.Series): Row of the dataset.
            rule (list): Rule to be checked.

        Returns:
            bool: True if the instance satisfies the rule, False otherwise.
        """
        for feature, operator, value in rule:
            #print("Cheking feature: ", feature)
            if operator == '==':
                if instance[feature] != value:
                    return False
            # Add conditions for other operators if needed
        return True