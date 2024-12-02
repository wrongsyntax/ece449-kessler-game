# fix_parameters.py

import json


def fix_parameters(input_file, output_file):
    """
    Load parameters from input_file, sort each membership function's parameters,
    and save the corrected parameters to output_file.
    """
    with open(input_file, 'r') as f:
        params = json.load(f)

    for param_type in ['thrust_params', 'turn_rate_params']:
        for mf in params[param_type]:
            original = params[param_type][mf]
            sorted_mf = sorted(original)
            params[param_type][mf] = sorted_mf
            print(f"Sorted {param_type}[{mf}]: {sorted_mf}")

    with open(output_file, 'w') as f:
        json.dump(params, f, indent=4)

    print(f"\nFixed parameters saved to '{output_file}'.")


if __name__ == "__main__":
    input_file = 'best_parameters.json'
    output_file = 'fixed_best_parameters.json'
    fix_parameters(input_file, output_file)
