def load_dataset(dataset_path):
    # Load dataset from file
    # Example usage of pandas to load dataset
    import pandas as pd
    dataset = pd.read_csv(dataset_path)
    return dataset
def save_rules(output_path, rules):
    # Save rules to a file
    with open(output_path, 'w') as file:
        for rule in rules:
            # Convert the rule to a string before writing to the file
            file.write(str(rule) + '\n')
