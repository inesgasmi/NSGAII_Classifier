import numpy as np
from rule_evaluator import RuleEvaluator
import pandas as pd
from Model_Validation import ModelValidation

class NSGA2:

    def __init__(self, population_size, num_generations, dataset):
        self.population_size = population_size
        self.num_generations = num_generations
        self.rule_evaluator = RuleEvaluator(dataset)
        self.validation_data_loaded = False

    def load_validation_data(self, X_val, y_val):
            self.X_val = X_val
            self.y_val = y_val
            self.validation_data_loaded = True

    def initialize_population(self, population_size, num_features):
        # Initialize population randomly
        population = []
        feature_names = self.rule_evaluator.dataset.columns[:-1]  # Exclude the last column ('Risk')
        for _ in range(population_size):
            individual = [(feature, '==', np.random.rand()) for feature in feature_names]
            population.append(individual)
        return population

    def nsga2_algorithm(self):
        # Initialize population randomly
        population = self.initialize_population(self.population_size, self.rule_evaluator.get_num_features())

        for generation in range(self.num_generations):
            # Evaluate the population
            self.evaluate_population(population)


            print("Generated Rules:", population)
            # Select parents for crossover
            parents = self.select_parents(population)

            # Perform crossover and mutation
            offspring = self.crossover(parents)
            offspring = self.mutate(offspring)

            # Combine parents and offspring
            combined_population = population + offspring

            # Non-dominated sorting
            fronts = self.non_dominated_sorting(combined_population)

            # Select next generation based on fronts
            population = []
            front_index = 0
            while front_index < len(fronts) and len(population) + len(fronts[front_index]) <= self.population_size:
                population += fronts[front_index]
                front_index += 1

            # Select remaining individuals using crowding distance
            crowded_selection = self.select_by_crowding(fronts[front_index], self.population_size - len(population))
            population += crowded_selection

        # Return final population
        return population

    def evaluate_population(self, population):
        """
        Evaluate the fitness of each individual in the population.

        Parameters:
            population (list): List of individuals (rules).

        Returns:
            list: List of tuples containing the fitness values of each individual.
        """
        fitness_values = []
        for rule in population:
            accuracy, complexity = self.rule_evaluator.evaluate_rule(rule)
            fitness_values.append((accuracy, complexity))
        return fitness_values

    def select_parents(self, population):
        # Convert population list to 1-dimensional array
        population_array = np.array(population)

        # Flatten the population_array
        population_array_flat = population_array.flatten()

        # Select parents for crossover
        parents = []
        for _ in range(len(population) // 2):
            parent1 = np.random.choice(population_array_flat)
            parent2 = np.random.choice(population_array_flat)
            parents.append((parent1, parent2))
        return parents

    def crossover(self, parents):
        # Perform crossover
        offspring = []
        for parent1, parent2 in parents:
            if isinstance(parent1, np.ndarray) and isinstance(parent2, np.ndarray):
                if len(parent1) == len(parent2) == self.rule_evaluator.get_num_features():
                    crossover_point = np.random.randint(1, self.rule_evaluator.get_num_features() - 1)
                    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
                    offspring.append(child1)
                    offspring.append(child2)
                else:
                    print("Error: Parents have incorrect dimensions.")
            else:
                print("It is done")
        return offspring

    def mutate(self, offspring):
        # Perform mutation
        mutation_rate = 0.1
        for individual in offspring:
            for i in range(len(individual)):
                if np.random.rand() < mutation_rate:
                    individual[i] = np.random.rand()
        return offspring

    def dominates(self, individual1, individual2):
        # Check if individual1 dominates individual2
        # Return True if individual1 dominates individual2, False otherwise
        # Here, we assume minimization, so individual1 dominates individual2 if all objectives of individual1 are less than or equal to the corresponding objectives of individual2
        # Modify this according to your optimization problem
        return all(ind1 <= ind2 for ind1, ind2 in zip(individual1, individual2))

    def non_dominated_sorting(self, population):
        # Perform non-dominated sorting
        fronts = [[] for _ in range(len(population))]
        for i, individual1 in enumerate(population):
            for individual2 in population:
                if self.dominates(individual1, individual2):
                    fronts[i].append(individual2)
        return fronts
    def select_by_crowding(self, front, remaining_size):
        """
        Select individuals from a front based on crowding distance.

        Parameters:
            front (list): List of individuals belonging to a single front.
            remaining_size (int): Number of individuals needed to be selected.

        Returns:
            list: Selected individuals based on crowding distance.
        """
        # Calculate crowding distance for individuals in the front
        crowding_distances = self.crowding_distance(front)

        # Sort the front based on crowding distance
        sorted_front = [individual for _, individual in sorted(zip(crowding_distances, front), reverse=True)]

        # Select the top individuals based on crowding distance
        selected = sorted_front[:remaining_size]

        return selected

    def crowding_distance(self, front):
        """
        Calculate crowding distance for individuals in a front.

        Parameters:
            front (list): List of individuals belonging to a single front.

        Returns:
            numpy.array: Array of crowding distances for each individual in the front.
        """
        distances = np.zeros(len(front))
        num_objectives = len(front[0])

        for obj_index in range(num_objectives):
            # Extract numerical values from each rule in the front
            rule_values = []
            for rule in front:
                # Extract the numerical value for the current objective index
                rule_values.append(rule[obj_index][2])  # Index 2 corresponds to the numerical value in the tuple

            # Sort the rule_values along with their corresponding rules
            sorted_front_with_values = sorted(zip(front, rule_values), key=lambda x: x[1])

            # Boundary points get infinite distance
            distances[0] = distances[-1] = np.inf

            # Calculate the crowding distance for each rule in the sorted front
            for i in range(1, len(front) - 1):
                # Extract the numerical values of neighboring rules
                value_next = sorted_front_with_values[i + 1][1]
                value_prev = sorted_front_with_values[i - 1][1]

                # Calculate the difference between the current and next rules in the sorted front
                diff = float(value_next) - float(value_prev)

                # Update the distance for the current rule
                distances[i] += diff

        return distances


