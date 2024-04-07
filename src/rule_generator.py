import numpy as np
from nsga2 import NSGA2
from utils import load_dataset, save_rules


class RuleGenerator:
    def __init__(self, dataset_path, output_path, population_size=100, num_generations=50):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.population_size = population_size
        self.num_generations = num_generations

    def generate_classification_rules(self):
        # Load dataset
        dataset = self.load_dataset(self.dataset_path)

        # Preprocess dataset if we needed
        # dataset = self.preprocess_dataset(dataset)

        # Feature selection if we needed
        # dataset = self.feature_selection(dataset)


        # Initialize NSGA-II algorithm
        nsga2 = NSGA2(self.population_size, self.num_generations, 11)

        # Generate classification rules using NSGA-II
        rules = nsga2.nsga2_algorithm()

        # Save rules to output file
        self.save_rules(self.output_path, rules)

    def load_dataset(self, dataset_path):
        # Load dataset from file
        import pandas as pd
        dataset = pd.read_csv(dataset_path)
        return dataset

    def preprocess_dataset(self, dataset):
        # Perform data preprocessing tasks
        return dataset

    def feature_selection(self, dataset):
        # Perform feature selection tasks
        return dataset

    def save_rules(self, output_path, rules):
        # Save rules to a file
        with open(output_path, 'w') as file:
            for rule in rules:
                file.write(str(rule) + '\n')

