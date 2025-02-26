import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol

n_sim = 1024

def cff(a, r1, r2, qp, qs_in, qs_out, ev, ev_star, erec, erec_eol, ed):
    return (1 - r1) * ev + r1 * (a * erec + (1 - a) * ev * qs_in / qp) + (1 - a) * r2 * (erec_eol - ev_star * qs_out / qp) + (1 - r2) * ed

case_studies = [
    {
        "name": "concrete industrial floor", 
        "case_study_key": "industrial_floor", 
        "problem": {
            'num_vars': 2,
            'names': ['r2', 'qs_out'],
            'bounds': [
                [0, 0.7],
                [0.5, 1]
            ]
        },
        "constants": {
            "a": 0.5,
            "r1": 0,
            "qp": 1,
            "qs_in": 0.52
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
                [0, 0.8],
                [0.3, 0.6],
                [0.5, 1]
            ]
        },
        "constants": {
            "r1": 0,
            "qp": 1
        }
    },
]

impact_data = pd.read_excel("data/CFF Paper Data V10_Clean_SO.xlsx", sheet_name="python_impact")
impact_categories = ["GWP", "ADP_elements", "ADP_fossil", "AP", "EP", "HTP", "ODP", "POCP"]

for case_study in case_studies:
    case_study_key = case_study["case_study_key"]
    if case_study_key not in impact_data.columns:
        raise ValueError(f"Column '{case_study_key}' not found in data.")
    
    problem = case_study["problem"]
    param_values = saltelli.sample(problem, n_sim)
    impact_values = impact_data[case_study_key]
    constants = case_study.get("constants", {})
    
    for impact_category in impact_categories:
        erec = impact_values.loc[impact_data['Impact Value'] == f'erec_{impact_category}'].values[0]
        erec_eol = impact_values.loc[impact_data['Impact Value'] == f'erec_eol_{impact_category}'].values[0]
        ed = impact_values.loc[impact_data['Impact Value'] == f'ed_{impact_category}'].values[0]
        ev = impact_values.loc[impact_data['Impact Value'] == f'ev_{impact_category}'].values[0]
        ev_star = impact_values.loc[impact_data['Impact Value'] == f'ev_star_{impact_category}'].values[0]
        
        Y = np.array([
            cff(
                constants.get("a", param_values[i, problem['names'].index("a")] if "a" in problem['names'] else None),
                constants.get("r1", param_values[i, problem['names'].index("r1")] if "r1" in problem['names'] else None),
                constants.get("r2", param_values[i, problem['names'].index("r2")] if "r2" in problem['names'] else None),
                constants.get("qp", param_values[i, problem['names'].index("qp")] if "qp" in problem['names'] else None),
                constants.get("qs_in", param_values[i, problem['names'].index("qs_in")] if "qs_in" in problem['names'] else None),
                constants.get("qs_out", param_values[i, problem['names'].index("qs_out")] if "qs_out" in problem['names'] else None),
                ev, ev_star, erec, erec_eol, ed
            ) for i in range(len(param_values))
        ])
        
        if None in Y:
            raise ValueError("One or more required parameters are missing values.")
        
        Si = sobol.analyze(problem, Y)

        x = np.arange(len(problem['names']))
        width = 0.4

        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, Si['S1'], width, yerr=Si['S1_conf'], capsize=5, label='First-order')
        plt.bar(x + width/2, Si['ST'], width, yerr=Si['ST_conf'], capsize=5, alpha=0.5, label='Total-order')

        plt.xticks(ticks=x, labels=problem['names'])
        plt.title(f'Sobol Sensitivity Analysis - {case_study_key} ({impact_category})')
        plt.ylabel('Sensitivity Index')
        plt.legend()
        plt.grid(True)

        save_dir = 'plots'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{case_study_key}_Sobol_{impact_category}.png')
        plt.savefig(save_path)
        