# main.py

from genetic_algorithm import run_ga_optimization

# Main function calling the other modules
# The genetic algorithm will produce results after every generation
# THe best chromosomes
def main():
    print("Genetic algorithm in progress")
    best_parameters = run_ga_optimization()
    print("Optimization complete.")
    print("Best Parameters:")
    print(best_parameters)
    print("Best parameters saved to 'best_parameters.json'.")

if __name__ == "__main__":
    main()
