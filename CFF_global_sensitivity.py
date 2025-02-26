
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

n_sim = 10000

# Define CFF function and its partial derivatives

def cff(a, r1, r2, qp, qs_in, qs_out, ev, ev_star, erec, erec_eol, ed):
    return (1-r1)*ev + r1*(a*erec+(1-a)*ev*qs_in/qp) + (1-a)*r2*(erec_eol-ev_star*qs_out/qp)+(1 - r2)*ed

# Define case study parameters
case_studies = [
    {
        "case_study_key": "industrial_floor",
    },
    {
        "case_study_key": "composite_floor", 
    },
]

# Load data from Excel
impact_data = pd.read_excel("data/CFF Paper Data V10_Clean_SO.xlsx", sheet_name="python_impact")


def truncated_normal(mean, std, lower, upper, size):
    a, b = (lower - mean) / std, (upper - mean) / std  # Convert bounds to standard normal space
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

def from_excel_to_distributions(case_study_key):
    distribution_data = pd.read_excel("data/CFF Paper Data V10_Clean_SO.xlsx", sheet_name="python_distr")

    # loading distribution data for the selected case study
    mask = (distribution_data["case_study"] == case_study_key)
    case_study_distributions = distribution_data[mask].to_numpy()
    param_distributions = case_study_distributions[:, 2:]
    samples = []
    for p in range(len(param_distributions)):
        if param_distributions[p][0] == "truncnorm":
            sample = truncated_normal(param_distributions[p][1],
                                      param_distributions[p][2],
                                      param_distributions[p][3],
                                      param_distributions[p][4],
                                      n_sim)
        elif param_distributions[p][0] == "uniform":
            sample = np.random.uniform(param_distributions[p][1],
                                       param_distributions[p][2],
                                       n_sim)
        elif param_distributions[p][0] == "triang":
            sample = np.random.triangular(param_distributions[p][1],
                                          param_distributions[p][2],
                                          param_distributions[p][3],
                                          n_sim)
        elif param_distributions[p][0] == "norm":
            sample = np.random.normal(param_distributions[p][1],
                                      param_distributions[p][2],
                                      n_sim)
        elif param_distributions[p][0] == "fixed":
            sample = np.full(n_sim, param_distributions[p][1])
        else:
            raise ValueError(f"Unknown distribution type '{param_distributions[p][0]}'.")
        samples.append(sample)
    return samples


# Define impact categories
impact_categories = ["GWP", "ADP_elements", "ADP_fossil", "AP", "EP", "HTP", "ODP", "POCP"]

# Precompute row indices for all impact categories
impact_index_map = {}
for impact_category in impact_categories:
    for prefix in ["erec", "erec_eol", "ed", "ev", "ev_star"]:
        key = f"{prefix}_{impact_category}"
        if impact_data["Impact Value"].eq(key).any():
            impact_index_map[key] = impact_data.index[impact_data["Impact Value"].eq(key)][0]
        else:
            raise ValueError(f"Key '{key}' not found in 'Impact Value' column of impact data.")

# Collect results
derivatives_results = []

# Iterate over case studies and extract values
for case_study in case_studies:    
    case_study_key = case_study["case_study_key"]
    if case_study_key not in impact_data.columns:
        raise ValueError(f"Column '{case_study_key}' not found in data. Skipping case study '{case_study['name']}'.")

    impact_values = impact_data[case_study_key]
    distr_a, distr_r1, distr_r2, distr_qp, distr_qs_in, distr_qs_out = from_excel_to_distributions(case_study_key)

    for impact_category in impact_categories:
        erec = impact_values.iloc[impact_index_map[f"erec_{impact_category}"]]
        erec_eol = impact_values.iloc[impact_index_map[f"erec_eol_{impact_category}"]]
        ed = impact_values.iloc[impact_index_map[f"ed_{impact_category}"]]
        ev = impact_values.iloc[impact_index_map[f"ev_{impact_category}"]]
        ev_star = impact_values.iloc[impact_index_map[f"ev_star_{impact_category}"]]

        cff_samples = []
        for i in range(len(distr_a)):
            a = distr_a[i]
            r1 = distr_r1[i]
            r2 = distr_r2[i]
            qp = distr_qp[i]
            qs_in = distr_qs_in[i]
            qs_out = distr_qs_out[i]

            cff_case_study = cff(a, r1, r2, qp, qs_in, qs_out, ev, ev_star, erec, erec_eol, ed)
            cff_samples.append(cff_case_study)

        plt.figure(figsize=(10, 6))
        plt.hist(cff_samples, bins=50, edgecolor='black', alpha=0.7)
        plt.title(f'Global sensitivity analysis results for {" ".join(case_study_key.split("_"))} ({n_sim} samples)')
        plt.xlabel(f'CFF values ({impact_category})')
        plt.ylabel('Frequency')
        plt.grid(True)
        # Add a textbox with median and standard deviation
        stats_text = (f"Mean: {np.mean(cff_samples):.2f}\nMedian: {np.median(cff_samples):.2f}"
                      f"\nStd Dev: {np.std(cff_samples):.2f}\nCV: {np.std(cff_samples) / np.mean(cff_samples):.2f}")
        plt.text(
            0.05, 0.9,  # Position (normalized coordinates: top-right corner)
            stats_text,
            transform=plt.gca().transAxes,  # Use axes coordinates
            fontsize=12,
            verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
        )

        save_dir = f'plots'
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f'{case_study_key}_GSA_CFF_{impact_category}.png')
        plt.savefig(save_path)