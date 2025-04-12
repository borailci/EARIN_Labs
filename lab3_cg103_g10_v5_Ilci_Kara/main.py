import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm
from functions import styblinski_tang_2d


def run_experiment1():
    """
    Experiment 1: Finding genetic algorithm parameters
    Run the algorithm with different parameters to find a set that obtains good fitness values
    """
    print("\n=== Experiment 1: Finding good parameters ===")

    # Parameters to test
    param_sets = [
        # (population_size, mutation_rate, mutation_strength, crossover_rate, generations)
        (50, 0.1, 0.5, 0.7, 100),
        (100, 0.1, 0.5, 0.7, 100),
        (100, 0.2, 0.5, 0.7, 100),
        (100, 0.1, 1.0, 0.7, 100),
        (100, 0.1, 0.5, 0.9, 100),
        (100, 0.05, 0.3, 0.8, 100),
    ]

    results = []

    for params in param_sets:
        pop_size, mut_rate, mut_strength, cross_rate, gens = params

        print(
            f"\nTesting parameters: Population={pop_size}, Mutation Rate={mut_rate}, "
            f"Mutation Strength={mut_strength}, Crossover Rate={cross_rate}, Generations={gens}"
        )

        ga = GeneticAlgorithm(
            population_size=pop_size,
            mutation_rate=mut_rate,
            mutation_strength=mut_strength,
            crossover_rate=cross_rate,
            num_generations=gens,
        )

        _, best_fitness_values, avg_fitness_values = ga.evolve(seed=42)

        # Store the final best fitness
        final_best_fitness = best_fitness_values[-1]
        results.append((params, final_best_fitness))

        # Plot fitness vs generation for this parameter set
        plt.figure(figsize=(10, 6))

        # Plot both best and average fitness
        plt.plot(
            range(gens),
            best_fitness_values,
            label="Best Fitness",
            color="blue",
            linewidth=2,
        )
        plt.plot(
            range(gens),
            avg_fitness_values,
            label="Avg Fitness",
            color="red",
            linewidth=1,
            alpha=0.7,
        )

        plt.title(
            f"Fitness vs Generation\nPopulation={pop_size}, MutRate={mut_rate}, "
            f"MutStrength={mut_strength}, CrossRate={cross_rate}"
        )
        plt.xlabel("Generation")
        plt.ylabel("Fitness Value")
        plt.grid(True)
        plt.legend()

        # Add annotations for best fitness achieved
        plt.annotate(
            f"Best: {best_fitness_values[-1]:.2f}",
            xy=(gens - 5, best_fitness_values[-1]),
            xytext=(gens - 30, best_fitness_values[-1] - 10),
            arrowprops=dict(arrowstyle="->"),
        )

        plt.savefig(
            f"experiment1_pop{pop_size}_mut{mut_rate}_str{mut_strength}_cross{cross_rate}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Sort results by fitness (lower is better)
    results.sort(key=lambda x: x[1])

    # Print results in a table format
    print("\nParameters Tested and Results:")
    print("-" * 90)
    print(
        "{:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
            "Population",
            "Mutation Rate",
            "Mut. Strength",
            "Crossover Rate",
            "Generations",
            "Best Fitness",
        )
    )
    print("-" * 90)

    for params, fitness in results:
        pop_size, mut_rate, mut_strength, cross_rate, gens = params
        print(
            "{:<15} {:<15.3f} {:<15.3f} {:<15.3f} {:<15} {:<15.6f}".format(
                pop_size, mut_rate, mut_strength, cross_rate, gens, fitness
            )
        )

    # Return the best parameters
    best_params = results[0][0]
    return best_params


def run_experiment2(best_params):
    """
    Experiment 2: Randomness in genetic algorithm
    Run the algorithm with the best parameters using 5 different random seeds
    Then rerun with decreasing population sizes
    """
    print("\n=== Experiment 2: Randomness in genetic algorithm ===")

    pop_size, mut_rate, mut_strength, cross_rate, gens = best_params
    seeds = [42, 123, 456, 789, 101112]

    # Run with best parameters and different seeds
    print("\nRunning with best parameters and different seeds:")
    all_fitness_values = []
    best_overall_solution = None
    best_overall_fitness = float("inf")

    for seed in seeds:
        print(f"\nSeed: {seed}")
        ga = GeneticAlgorithm(
            population_size=pop_size,
            mutation_rate=mut_rate,
            mutation_strength=mut_strength,
            crossover_rate=cross_rate,
            num_generations=gens,
        )

        best_solutions, best_fitness_values, avg_fitness_values = ga.evolve(seed=seed)

        final_best_fitness = best_fitness_values[-1]
        all_fitness_values.append(final_best_fitness)

        # Check if this is the best solution overall
        if final_best_fitness < best_overall_fitness:
            best_overall_fitness = final_best_fitness
            best_overall_solution = best_solutions[-1]

    # Calculate statistics
    avg_fitness = np.mean(all_fitness_values)
    std_fitness = np.std(all_fitness_values)

    print(f"\nBest overall solution: {best_overall_solution}")
    print(f"Best overall fitness: {best_overall_fitness:.6f}")
    print(f"Average fitness: {avg_fitness:.6f}")
    print(f"Standard deviation: {std_fitness:.6f}")

    # Run with different population sizes
    pop_size_factors = [1.0, 0.5, 0.25, 0.1]
    pop_size_results = []

    for factor in pop_size_factors:
        new_pop_size = int(pop_size * factor)
        if new_pop_size < 5:
            new_pop_size = 5

        print(
            f"\nTesting with population size: {new_pop_size} ({factor*100:.0f}% of original)"
        )

        all_seed_fitness = []
        all_best_fitness_curves = []
        all_avg_fitness_curves = []

        for seed in seeds:
            ga = GeneticAlgorithm(
                population_size=new_pop_size,
                mutation_rate=mut_rate,
                mutation_strength=mut_strength,
                crossover_rate=cross_rate,
                num_generations=gens,
            )

            _, best_fitness_values, avg_fitness_values = ga.evolve(seed=seed)

            all_best_fitness_curves.append(best_fitness_values)
            all_avg_fitness_curves.append(avg_fitness_values)

            final_best_fitness = best_fitness_values[-1]
            all_seed_fitness.append(final_best_fitness)

        # Calculate statistics for this population size
        avg_fitness = np.mean(all_seed_fitness)
        std_fitness = np.std(all_seed_fitness)
        best_fitness = min(all_seed_fitness)

        # Calculate average fitness curves across seeds
        avg_best_curve = np.mean(all_best_fitness_curves, axis=0)
        avg_avg_curve = np.mean(all_avg_fitness_curves, axis=0)

        # Plot the average curves for this population size
        if factor in [0.5]:  # Only plot the 50% curve as an example
            plt.figure(figsize=(10, 6))
            plt.plot(
                range(gens),
                avg_best_curve,
                label="Best Fitness",
                color="blue",
                linewidth=2,
            )
            plt.plot(
                range(gens),
                avg_avg_curve,
                label="Avg Fitness",
                color="red",
                linewidth=1,
                alpha=0.7,
            )
            plt.title(
                f"Fitness vs Generation with Population Size {new_pop_size} ({factor*100:.0f}% of original)"
            )
            plt.xlabel("Generation")
            plt.ylabel("Fitness Value")
            plt.grid(True)
            plt.legend()
            plt.annotate(
                f"Best: {best_fitness:.2f}",
                xy=(gens - 5, avg_best_curve[-1]),
                xytext=(gens - 30, avg_best_curve[-1] - 10),
                arrowprops=dict(arrowstyle="->"),
            )
            plt.savefig(
                f"experiment1_pop{new_pop_size}_mut{mut_rate}_str{mut_strength}_cross{cross_rate}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        pop_size_results.append(
            (new_pop_size, factor, best_fitness, avg_fitness, std_fitness)
        )

    # Print results in a table format
    print("\nPopulation Size Experiment Results:")
    print("-" * 80)
    print(
        "{:<15} {:<15} {:<15} {:<15} {:<15}".format(
            "Pop. Size",
            "% of Original",
            "Best Fitness",
            "Avg Fitness",
            "Std. Deviation",
        )
    )
    print("-" * 80)

    for pop, factor, best, avg, std in pop_size_results:
        print(
            "{:<15} {:<15.0f}% {:<15.6f} {:<15.6f} {:<15.6f}".format(
                pop, factor * 100, best, avg, std
            )
        )


def run_experiment3(best_params):
    """
    Experiment 3: Crossover impact
    Run the algorithm with different crossover rates
    """
    print("\n=== Experiment 3: Crossover impact ===")

    pop_size, mut_rate, mut_strength, _, gens = best_params
    crossover_rates = [0.0, 0.25, 0.5, 0.75, 1.0]
    seeds = [42, 123, 456]

    crossover_results = []

    for cr in crossover_rates:
        print(f"\nTesting crossover rate: {cr}")

        all_best_fitness = []
        all_avg_fitness = []

        for seed in seeds:
            ga = GeneticAlgorithm(
                population_size=pop_size,
                mutation_rate=mut_rate,
                mutation_strength=mut_strength,
                crossover_rate=cr,
                num_generations=gens,
            )

            _, best_fitness_values, avg_fitness_values = ga.evolve(seed=seed)

            all_best_fitness.append(best_fitness_values)
            all_avg_fitness.append(avg_fitness_values)

        # Average results across seeds
        avg_best_fitness = np.mean(all_best_fitness, axis=0)
        avg_avg_fitness = np.mean(all_avg_fitness, axis=0)

        crossover_results.append((cr, avg_best_fitness, avg_avg_fitness))

    # Plot results
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    for cr, best_fitness, _ in crossover_results:
        plt.plot(range(gens), best_fitness, label=f"CR={cr}")
    plt.title("Impact of Crossover Rate on Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.grid(True)

    # Add text annotation with final best values
    for i, (cr, best_fitness, _) in enumerate(crossover_results):
        plt.annotate(
            f"CR={cr}: {best_fitness[-1]:.2f}",
            xy=(gens - 1, best_fitness[-1]),
            xytext=(gens - 25, best_fitness[-1] + i * 2 - 4),
            arrowprops=dict(arrowstyle="->"),
            fontsize=8,
        )

    plt.subplot(2, 1, 2)
    for cr, _, avg_fitness in crossover_results:
        plt.plot(range(gens), avg_fitness, label=f"CR={cr}")
    plt.title("Impact of Crossover Rate on Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("experiment3_crossover_impact.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_experiment4(best_params):
    """
    Experiment 4: Mutation and convergence
    Run the algorithm with increasing mutation rate and strength
    """
    print("\n=== Experiment 4: Mutation and convergence ===")

    pop_size, _, _, cross_rate, gens = best_params
    mutation_configs = [
        (0.01, 0.1),  # (rate, strength)
        (0.05, 0.3),
        (0.1, 0.5),
        (0.2, 0.7),
        (0.3, 1.0),
    ]
    seeds = [42, 123, 456]

    mutation_results = []

    for mut_rate, mut_strength in mutation_configs:
        print(f"\nTesting mutation rate: {mut_rate}, strength: {mut_strength}")

        all_best_fitness = []
        all_avg_fitness = []

        for seed in seeds:
            ga = GeneticAlgorithm(
                population_size=pop_size,
                mutation_rate=mut_rate,
                mutation_strength=mut_strength,
                crossover_rate=cross_rate,
                num_generations=gens,
            )

            _, best_fitness_values, avg_fitness_values = ga.evolve(seed=seed)

            all_best_fitness.append(best_fitness_values)
            all_avg_fitness.append(avg_fitness_values)

        # Average results across seeds
        avg_best_fitness = np.mean(all_best_fitness, axis=0)
        avg_avg_fitness = np.mean(all_avg_fitness, axis=0)

        mutation_results.append(
            (mut_rate, mut_strength, avg_best_fitness, avg_avg_fitness)
        )

    # Plot results
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    for mut_rate, mut_strength, best_fitness, _ in mutation_results:
        plt.plot(range(gens), best_fitness, label=f"MR={mut_rate}, MS={mut_strength}")
    plt.title("Impact of Mutation Parameters on Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.grid(True)

    # Add text annotation with final best values
    for i, (mut_rate, mut_strength, best_fitness, _) in enumerate(mutation_results):
        plt.annotate(
            f"MR={mut_rate}, MS={mut_strength}: {best_fitness[-1]:.2f}",
            xy=(gens - 1, best_fitness[-1]),
            xytext=(gens - 40, best_fitness[-1] + i * 2 - 4),
            arrowprops=dict(arrowstyle="->"),
            fontsize=8,
        )

    plt.subplot(2, 1, 2)
    for mut_rate, mut_strength, _, avg_fitness in mutation_results:
        plt.plot(range(gens), avg_fitness, label=f"MR={mut_rate}, MS={mut_strength}")
    plt.title("Impact of Mutation Parameters on Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("experiment4_mutation_impact.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Global minimum of Styblinski-Tang function is at approximately (-2.903534, -2.903534)
    # with a function value of approximately -78.332
    print(
        f"Known global minimum of Styblinski-Tang function: approximately (-2.903534, -2.903534)"
    )
    print(f"with function value: approximately -78.332")

    # Run experiments
    best_params = run_experiment1()
    run_experiment2(best_params)
    run_experiment3(best_params)
    run_experiment4(best_params)
