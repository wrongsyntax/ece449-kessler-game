# main.py

from genetic_algorithm import run_ga_optimization

def main():
    print("Starting Genetic Algorithm Optimization...")
    best_parameters = run_ga_optimization()
    print("Optimization complete.")
    print("Best Parameters:")
    print(best_parameters)
    print("Best parameters saved to 'best_parameters.json'.")

if __name__ == "__main__":
    main()
