#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
#%%
n_sim = 16384

def cff(a, r1, r2, qp, qs_in, qs_out, ev, ev_star, erec, erec_eol, ed):
    return (1 - r1) * ev + r1 * (a * erec + (1 - a) * ev * qs_in / qp) + (1 - a) * r2 * (
                erec_eol - ev_star * qs_out / qp) + (1 - r2) * ed

case_studies = [
    {
        "name": "concrete industrial floor",
        "case_study_key": "industrial_floor",
        "problem": {
            'num_vars': 4,
            'names': ['a', 'r2', 'qs_in', 'qs_out'],
            'bounds': [
                [0.45, 0.55, 0.5, 0.05],
                [0., 0.7, 0.9999],
                [0.73, 1, 0.85, 0.1],
                [0.48, 1, 0.76, 0.1]
            ],
            'dists': ['truncnorm', 'triang', 'truncnorm', 'truncnorm']
        },
        "constants": {
            "r1": 0,
            "qp": 1
        }
    },
    {
        "name": "carbon concrete industrial floor",
        "case_study_key": "composite_floor",
        "problem": {
            'num_vars': 4,
            'names': ['a', 'r2', 'qs_in', 'qs_out'],
            'bounds': [
                [0.5, 0.7],
                [0, 0.97, 0.9999],
                [0.73, 0.9, 0.85, 0.05],
                [0.23, 0.81, 0.67, 0.1]
            ],
            'dists': ['unif', 'triang', 'truncnorm', 'truncnorm']
        },
        "constants": {
            "r1": 0.6,
            "qp": 1
        }
    },
]

impact_data = pd.read_excel("data/CFF Paper Data V10_Clean_SO.xlsx", sheet_name="python_impact")
impact_categories = ["GWP", "ADP_elements", "ADP_fossil", "AP", "EP", "HTP", "ODP", "POCP"]

# Initialize a list to collect all results
results = []

#%%
for case_study in case_studies:
    case_study_key = case_study["case_study_key"]
    if case_study_key not in impact_data.columns:
        raise ValueError(f"Column '{case_study_key}' not found in data.")

    problem = case_study["problem"]
    param_values = sobol.sample(problem, n_sim)
    print('param_values has a shape:', np.shape(param_values))
    # Plot histograms for each parameter
    num_vars = problem["num_vars"]
    fig, axes = plt.subplots(1, num_vars, figsize=(15, 4))

    for i in range(num_vars):
        ax = axes[i] if num_vars > 1 else axes  # Handle single variable case
        ax.hist(param_values[:, i], bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(problem["names"][i])
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    plt.suptitle(f"Parameter Distributions - {case_study_key}")
    plt.tight_layout()
    plt.savefig(f'plots/sobol/{case_study_key}_Parameter distributions.png')

    impact_values = impact_data[case_study_key]
    constants = case_study.get("constants", {})

    # Initialize a dictionary to store results for this case study
    case_results = {"Parameter": problem['names']}

    for impact_category in impact_categories:
        erec = impact_values.loc[impact_data['Impact Value'] == f'erec_{impact_category}'].values[0]
        erec_eol = impact_values.loc[impact_data['Impact Value'] == f'erec_eol_{impact_category}'].values[0]
        ed = impact_values.loc[impact_data['Impact Value'] == f'ed_{impact_category}'].values[0]
        ev = impact_values.loc[impact_data['Impact Value'] == f'ev_{impact_category}'].values[0]
        ev_star = impact_values.loc[impact_data['Impact Value'] == f'ev_star_{impact_category}'].values[0]

        Y = np.array([
            cff(
                constants.get("a",
                              param_values[i, problem['names'].index("a")] if "a" in problem['names'] else None),
                constants.get("r1",
                              param_values[i, problem['names'].index("r1")] if "r1" in problem['names'] else None),
                constants.get("r2",
                              param_values[i, problem['names'].index("r2")] if "r2" in problem['names'] else None),
                constants.get("qp",
                              param_values[i, problem['names'].index("qp")] if "qp" in problem['names'] else None),
                constants.get("qs_in",
                              param_values[i, problem['names'].index("qs_in")] if "qs_in" in problem['names'] else None),
                constants.get("qs_out",
                              param_values[i, problem['names'].index("qs_out")] if "qs_out" in problem['names'] else None),
                ev, ev_star, erec, erec_eol, ed
            )
            for i in range(len(param_values))
        ])

        if None in Y:
            raise ValueError("One or more required parameters are missing values.")

        print("Vector Y has a shape:", np.shape(Y))

        Si = sobol_analyze.analyze(problem, Y)
        #  print(sum(Si['S1']))  # Test for checking that first order indices sum = 1

        # Collect results
        for i, param in enumerate(problem['names']):
            results.append({
                'Case Study': case_study_key,
                'Impact Category': impact_category,
                'Parameter': param,
                'First-order Index': Si['S1'][i],
                'Total-order Index': Si['ST'][i]
            })

        x = np.arange(len(problem['names']))
        width = 0.4

        plt.figure(figsize=(10, 6))
        plt.bar(x - width / 2, Si['S1'], width, yerr=Si['S1_conf'], capsize=5, label='First-order')
        plt.bar(x + width / 2, Si['ST'], width, yerr=Si['ST_conf'], capsize=5, alpha=0.5, label='Total-order')

        plt.xticks(ticks=x, labels=problem['names'])
        plt.title(f'Sobol Sensitivity Analysis - {case_study_key} ({impact_category})')
        plt.ylabel('Sensitivity Index')
        plt.legend()
        plt.grid(True)

        plt.savefig(f'plots/sobol/{case_study_key}/{case_study_key}_Sobol_{impact_category}.png')

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Write results to an Excel file
with pd.ExcelWriter('results/Sobol_floor.xlsx') as writer:
    results_df.to_excel(writer, index=False, sheet_name='Sobol Indices')