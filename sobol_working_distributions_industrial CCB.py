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
        "name": "industrial cross car beam",
        "case_study_key": ["industrial_CCB_al", "industrial_CCB_st"],
        "problem": {
            'num_vars': 10,
            'names': ['a_al', 'r2_al', 'qp_al', 'qs_in_al', 'qs_out_al',
                      'a_st', 'r2_st', 'qp_st', 'qs_in_st', 'qs_out_st'],
            'bounds': [
                [0.15, 0.25],
                [0.76, 0.98, 0.92, 0.03],
                [0.95, 1],
                [0.8, 1, 0.9999],
                [0.42, 1, 0.78, 0.15],
                [0.15, 0.25],
                [0.81, 0.98, 0.9, 0.07],
                [0.95, 1],
                [0.5, 1., 0.8, 0.13],
                [0.51, 1., 0.82, 0.12]
            ],
            'dists': ['unif', 'truncnorm', 'unif','triang', 'truncnorm', 'unif','truncnorm', 'unif','truncnorm', 'truncnorm']
        },
        "constants": {
            #"a_al": 0.2,
            "r1_al": 0,
            #"qp_al": 1,
            #"a_st": 0.2,
            "r1_st": 0,
            #"qp_st": 1
        }
    },
]

impact_data = pd.read_excel("data/CFF Paper Data V10_Clean_SO.xlsx", sheet_name="python_impact")
impact_categories = ["GWP", "ADP_elements", "ADP_fossil", "AP", "EP", "HTP", "ODP", "POCP"]

# Initialize a list to collect all results
results = []

#%%
for case_study in case_studies:
    case_study_key_list = case_study["case_study_key"]

    for case_study_key in case_study_key_list:
        if case_study_key not in impact_data.columns:
            raise ValueError(f"Column '{case_study_key}' not found in data.")

    problem = case_study["problem"]
    param_values = sobol.sample(problem, n_sim)
    print('param_values is shape', np.shape(param_values))

    # Plot histograms for each parameter
    num_vars = problem["num_vars"]
    fig, axes = plt.subplots(1, num_vars, figsize=(15, 4))

    for i in range(num_vars):
        ax = axes[i] if num_vars > 1 else axes  # Handle single variable case
        ax.hist(param_values[:, i], bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(problem["names"][i])
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    plt.suptitle(f"Parameter Distributions - industrial CCB")
    plt.tight_layout()
    #plt.savefig(f'plots/sobol/industrial_CCB/industrial_CCB_Parameter distributions.png')
    plt.savefig(f'plots/sobol/industrial_CCB/varyall_industrial_CCB_Parameter distributions.png')

    impact_values_al = impact_data[case_study_key_list[0]]
    impact_values_st = impact_data[case_study_key_list[1]]

    constants = case_study.get("constants", {})

    # Initialize a dictionary to store results for this case study
    case_results = {"Parameter": problem['names']}

    for impact_category in impact_categories:
        erec_al = impact_values_al.loc[impact_data['Impact Value'] == f'erec_{impact_category}'].values[0]
        erec_eol_al = impact_values_al.loc[impact_data['Impact Value'] == f'erec_eol_{impact_category}'].values[0]
        ed_al = impact_values_al.loc[impact_data['Impact Value'] == f'ed_{impact_category}'].values[0]
        ev_al = impact_values_al.loc[impact_data['Impact Value'] == f'ev_{impact_category}'].values[0]
        ev_star_al = impact_values_al.loc[impact_data['Impact Value'] == f'ev_star_{impact_category}'].values[0]

        erec_st = impact_values_st.loc[impact_data['Impact Value'] == f'erec_{impact_category}'].values[0]
        erec_eol_st = impact_values_st.loc[impact_data['Impact Value'] == f'erec_eol_{impact_category}'].values[0]
        ed_st = impact_values_st.loc[impact_data['Impact Value'] == f'ed_{impact_category}'].values[0]
        ev_st = impact_values_st.loc[impact_data['Impact Value'] == f'ev_{impact_category}'].values[0]
        ev_star_st = impact_values_st.loc[impact_data['Impact Value'] == f'ev_star_{impact_category}'].values[0]

        Y = np.array([
            0.95*cff(
                constants.get("a_al", param_values[i, problem['names'].index("a_al")] if "a_al" in problem['names'] else None),
                constants.get("r1_al",
                              param_values[i, problem['names'].index("r1_al")] if "r1_al" in problem['names'] else None),
                constants.get("r2_al",
                              param_values[i, problem['names'].index("r2_al")] if "r2_al" in problem['names'] else None),
                constants.get("qp_al",
                              param_values[i, problem['names'].index("qp_al")] if "qp_al" in problem['names'] else None),
                constants.get("qs_in_al", param_values[i, problem['names'].index("qs_in_al")] if "qs_in_al" in problem[
                    'names'] else None),
                constants.get("qs_out_al", param_values[i, problem['names'].index("qs_out_al")] if "qs_out_al" in problem[
                    'names'] else None),
                ev_al, ev_star_al, erec_al, erec_eol_al, ed_al
            )
            +0.05*cff(
                constants.get("a_st", param_values[i, problem['names'].index("a_st")] if "a_st" in problem['names'] else None),
                constants.get("r1_st",
                              param_values[i, problem['names'].index("r1_st")] if "r1_st" in problem['names'] else None),
                constants.get("r2_st",
                              param_values[i, problem['names'].index("r2_st")] if "r2_st" in problem['names'] else None),
                constants.get("qp_st",
                              param_values[i, problem['names'].index("qp_st")] if "qp_st" in problem['names'] else None),
                constants.get("qs_in_st", param_values[i, problem['names'].index("qs_in_st")] if "qs_in_st" in problem[
                    'names'] else None),
                constants.get("qs_out_st", param_values[i, problem['names'].index("qs_out_st")] if "qs_out_st" in problem[
                    'names'] else None),
                ev_st, ev_star_st, erec_st, erec_eol_st, ed_st
            )
            for i in range(len(param_values))
        ])

        if None in Y:
            raise ValueError("One or more required parameters are missing values.")
        print('Y is shape: ', np.shape(Y))

        Si = sobol_analyze.analyze(problem, Y)
        #  print(sum(Si['S1']))  # Test for checking that first order indices sum = 1

        # Collect results
        for i, param in enumerate(problem['names']):
            results.append({
                'Case Study': case_study['name'],
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
        plt.title(f'Sobol Sensitivity Analysis - industrial CCB ({impact_category})')
        plt.ylabel('Sensitivity Index')
        plt.legend()
        plt.grid(True)

        #plt.savefig(f'plots/sobol/industrial_CCB/industrial_CCB_Sobol_{impact_category}.png')
        plt.savefig(f'plots/sobol/industrial_CCB/varyall_industrial_CCB_Sobol_{impact_category}.png')

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Write results to an Excel file
with pd.ExcelWriter('results/varyall_Sobol_industrial CCB.xlsx') as writer:
    results_df.to_excel(writer, index=False, sheet_name='Sobol Indices')