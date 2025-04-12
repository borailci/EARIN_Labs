import random

import numpy as np
from functions import styblinski_tang_2d, init_ranges


def set_seed(seed: int) -> None:
    # Set fixed random seed to make the results reproducible
    random.seed(seed)
    np.random.seed(seed)


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        mutation_strength: float,
        crossover_rate: float,
        num_generations: int,
        fitness_function=styblinski_tang_2d,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.fitness_function = fitness_function
        self.ranges = init_ranges[fitness_function]

    def initialize_population(self):
        """Initialize random population within the specified ranges"""
        x_range, y_range = self.ranges
        population = []

        for _ in range(self.population_size):
            x = random.uniform(x_range[0], x_range[1])
            y = random.uniform(y_range[0], y_range[1])
            population.append((x, y))

        return population

    def evaluate_population(self, population):
        """Evaluate the fitness of each individual in the population"""
        fitness_values = []

        for x, y in population:
            fitness = self.fitness_function(x, y)
            fitness_values.append(fitness)

        return fitness_values

    def selection(self, population, fitness_values):
        """Select parents using tournament selection"""
        parents = []
        population_with_fitness = list(zip(population, fitness_values))

        # Tournament selection - we're finding minimum as we're minimizing the function
        for _ in range(self.population_size):
            # Select random candidates for tournament
            tournament_size = max(2, self.population_size // 5)  # 20% of population
            tournament = random.sample(population_with_fitness, tournament_size)

            # Find the best individual in the tournament (minimum fitness value)
            winner = min(tournament, key=lambda x: x[1])
            parents.append(winner[0])

        return parents

    def crossover(self, parents):
        """Perform crossover on randomly selected pairs of parents"""
        offspring = []

        # Determine how many individuals will undergo crossover
        num_to_crossover = int(self.population_size * self.crossover_rate)
        # Ensure it's an even number
        if num_to_crossover % 2 == 1:
            num_to_crossover -= 1

        # Copy individuals that won't undergo crossover
        offspring.extend(parents[num_to_crossover:])

        # Perform crossover on selected individuals
        parents_for_crossover = parents[:num_to_crossover]
        random.shuffle(parents_for_crossover)

        for i in range(0, len(parents_for_crossover), 2):
            if i + 1 < len(parents_for_crossover):
                parent1 = parents_for_crossover[i]
                parent2 = parents_for_crossover[i + 1]

                # Random interpolation crossover
                alpha = random.random()  # Random number between 0 and 1

                # Perform crossover for x and y coordinates
                x_offspring = alpha * parent1[0] + (1 - alpha) * parent2[0]
                y_offspring = alpha * parent1[1] + (1 - alpha) * parent2[1]

                offspring.append((x_offspring, y_offspring))

                # Second child with reversed alpha
                x_offspring2 = (1 - alpha) * parent1[0] + alpha * parent2[0]
                y_offspring2 = (1 - alpha) * parent1[1] + alpha * parent2[1]

                offspring.append((x_offspring2, y_offspring2))

        return offspring

    def mutate(self, individuals):
        """Apply Gaussian mutation to randomly selected individuals"""
        mutated_population = []

        for individual in individuals:
            x, y = individual

            # Apply mutation with probability mutation_rate
            if random.random() < self.mutation_rate:
                # Apply Gaussian mutation
                x_mutation = random.gauss(0, self.mutation_strength)
                y_mutation = random.gauss(0, self.mutation_strength)

                x += x_mutation
                y += y_mutation

                # Ensure values stay within bounds
                x_range, y_range = self.ranges
                x = max(min(x, x_range[1]), x_range[0])
                y = max(min(y, y_range[1]), y_range[0])

            mutated_population.append((x, y))

        return mutated_population

    def evolve(self, seed: int):
        """Run the genetic algorithm and return results for each generation"""
        set_seed(seed)

        population = self.initialize_population()

        best_solutions = []
        best_fitness_values = []
        average_fitness_values = []

        for generation in range(self.num_generations):
            # Evaluate the current population
            fitness_values = self.evaluate_population(population)

            # Find the best solution in this generation
            best_idx = np.argmin(fitness_values)
            best_solution = population[best_idx]
            best_fitness = fitness_values[best_idx]
            average_fitness = np.mean(fitness_values)

            # Save results for this generation
            best_solutions.append(best_solution)
            best_fitness_values.append(best_fitness)
            average_fitness_values.append(average_fitness)

            # Print progress every 10 generations
            if generation % 10 == 0:
                print(
                    f"Generation {generation}: Best fitness = {best_fitness:.6f}, Avg fitness = {average_fitness:.6f}"
                )

            # Selection, crossover and mutation
            parents_for_reproduction = self.selection(population, fitness_values)
            offspring = self.crossover(parents_for_reproduction)
            population = self.mutate(offspring)

        # Evaluate final population
        final_fitness_values = self.evaluate_population(population)
        final_best_idx = np.argmin(final_fitness_values)
        final_best_solution = population[final_best_idx]
        final_best_fitness = final_fitness_values[final_best_idx]
        final_average_fitness = np.mean(final_fitness_values)

        print(
            f"Final best solution: {final_best_solution}, fitness: {final_best_fitness:.6f}, avg fitness: {final_average_fitness:.6f}"
        )

        return best_solutions, best_fitness_values, average_fitness_values
