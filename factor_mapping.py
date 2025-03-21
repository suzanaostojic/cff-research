# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from scipy.stats import truncnorm

# %%
# Step 0: Load case study data
n_sim = 100000

case_studies = ["industrial_CCB_al", "industrial_CCB_st", "composite_CCB_al", "composite_CCB_st", "composite_CCB_cp",
                "industrial_floor", "composite_floor"]

gwp100_data = pd.read_excel(f"data/CFF Paper Data V10_Clean_SO.xlsx", sheet_name="python_impact", index_col=0)

distribution_data = pd.read_excel(f"data/CFF Paper Data V10_Clean_SO.xlsx", sheet_name="python_distr")
distribution_data = distribution_data.fillna(0)


# %%
# Step 1: Define the distributions for parameters
# Assume parameters follow normal distributions with given means and standard deviations

def truncated_normal(mean, std, lower, upper, size):
    a, b = (lower - mean) / std, (upper - mean) / std  # Convert bounds to standard normal space
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)


def setup_case_study_impacts(case_study):
    # loading gwp100 values for the selected case study
    erec = gwp100_data[case_study]["erec_GWP"]
    erec_eol = gwp100_data[case_study]["erec_eol_GWP"]
    ed = gwp100_data[case_study]["ed_GWP"]
    ev = gwp100_data[case_study]["ev_GWP"]
    ev_star = gwp100_data[case_study]["ev_star_GWP"]

    return [erec, erec_eol, ed, ev, ev_star]


def setup_case_study_distributions(case_study):
    # loading distribution data for the selected case study
    mask = (distribution_data["case_study"] == case_study)
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
            samples.append(sample)
        elif param_distributions[p][0] == "uniform":
            sample = np.random.uniform(param_distributions[p][1],
                                       param_distributions[p][2],
                                       n_sim)
            samples.append(sample)
        elif param_distributions[p][0] == "triang":
            sample = np.random.triangular(param_distributions[p][1],
                                          param_distributions[p][2],
                                          param_distributions[p][3],
                                          n_sim)
            samples.append(sample)
        elif param_distributions[p][0] == "norm":
            sample = np.random.normal(param_distributions[p][1],
                                      param_distributions[p][2],
                                      n_sim)
            samples.append(sample)
        elif param_distributions[p][0] == "fixed":
            sample = np.full(n_sim, param_distributions[p][1])
            samples.append(sample)
        else:
            print("Invalid distribution type")

    return samples


# %%
# Step 2: Define the CFF function

def cff(a, r1, r2, qs_in, qs_out, e_rec=0, e_rec_eol=7.63, ed=10.2, ev=156, ev_star=156):
    # using the concrete industrial floor with GWP100
    # now qs_in = qsin/qp, qs_out = qs_out/qp
    return (1 - r1) * ev + r1 * (a * e_rec + (1 - a) * ev * qs_in) + (1 - a) * r2 * (e_rec_eol - ev_star * qs_out) + (
                1 - r2) * ed


# %%
# Define gsa functions

def plotting_parameter_distributions(case_study, show=False):
    distributions = setup_case_study_distributions(case_study)

    params = {
        "a": distributions[0],
        "r1": distributions[1],
        "r2": distributions[2],
        "qp": distributions[3],
        "qs_in": distributions[4],
        "qs_out": distributions[5]
    }

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # 2 rows, 3 columns

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Plot histograms for each parameter
    for i, (param_name, samples) in enumerate(params.items()):
        axes[i].hist(samples, bins=50, edgecolor='black', alpha=0.7)
        axes[i].set_title(f'Distribution of {param_name} ({n_sim} samples)')
        axes[i].set_xlabel(param_name)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True)

    plt.tight_layout()

    if show:
        plt.show()


def gsa_cff(case_study, show_distribution=True, show_statistics=True, show_cff_histogram=True):
    impacts = setup_case_study_impacts(cs)

    distributions = setup_case_study_distributions(case_study)
    a_samples = distributions[0]
    r1_samples = distributions[1]
    r2_samples = distributions[2]
    qp_samples = distributions[3]
    qs_in_samples = distributions[4]
    qs_out_samples = distributions[5]

    if show_distribution:
        plotting_parameter_distributions(case_study, show=True)
    else:
        plotting_parameter_distributions(case_study, show=False)

    cff_samples = cff(a_samples, r1_samples, r2_samples,
                      qs_in_samples, qs_out_samples,
                      e_rec=impacts[0],
                      e_rec_eol=impacts[1],
                      ed=impacts[2],
                      ev=impacts[3],
                      ev_star=impacts[4])

    # Calculate and display statistics
    if show_statistics:
        print(f"Max of CFF: {np.max(cff_samples)}")
        print(f"Min of CFF: {np.min(cff_samples)}")
        print(f"Mean of CFF: {np.mean(cff_samples)}")
        print(f"Standard Deviation of CFF: {np.std(cff_samples)}")
        print(f"Median of CFF: {np.median(cff_samples)}")
        print(f"95th Percentile of CFF: {np.percentile(cff_samples, 95)}")
        print(f"Coefficient of Variation of CFF: {np.std(cff_samples) / np.mean(cff_samples)}")
        print("\n")

    # Plot histogram of cff_samples
    if show_cff_histogram:
        plt.figure(figsize=(10, 6))
        plt.hist(cff_samples, bins=50, edgecolor='black', alpha=0.7)
        plt.title(f'Global sensitivity analysis results for {case_study} ({n_sim} samples)')
        plt.xlabel('CFF values')
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
        plt.show()

    return cff_samples, [a_samples, r1_samples, r2_samples, qp_samples, qs_in_samples, qs_out_samples]


# %%
# Generating GSA results for all case studies

cff_all = []
param_values_all = []

for cs in case_studies:
    print(f"Case Study: {cs}")
    cff_cs, param_values_cs = gsa_cff(cs, show_distribution=True, show_statistics=True, show_cff_histogram=True)
    cff_all.append(cff_cs)
    param_values_all.append(param_values_cs)

# %%

cff_case_studies = {
    "industrial_CCB_al": cff_all[0],
    "industrial_CCB_st": cff_all[1],
    "composite_CCB_al": cff_all[2],
    "composite_CCB_st": cff_all[3],
    "composite_CCB_cp": cff_all[4],
    "industrial_floor": cff_all[5],
    "composite_floor": cff_all[6],
}

# Create subplots (2 rows, 4 columns) - 8 spaces, 1 will be empty
fig, axes = plt.subplots(4, 2, figsize=(18, 10))

# Flatten axes for easy iteration
axes = axes.flatten()

# Plot histograms
for i, (cs_name, samples) in enumerate(cff_case_studies.items()):
    j = i

    if i > 4:
        j = i + 1
    axes[j].hist(samples, bins=50, edgecolor='black', alpha=0.7)

    # Add a textbox with median and standard deviation
    stats_text = (f"Mean: {np.mean(samples):.2f}\nMedian: {np.median(samples):.2f}"
                  f"\nStd Dev: {np.std(samples):.2f}\nCV: {np.std(samples) / np.mean(samples):.2f}")
    axes[j].text(
        0.95, 0.95,
        stats_text,
        transform=axes[j].transAxes,  # Use axes coordinates
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
    )

    axes[j].set_title(f'CFF GSA results for {cs_name} ({n_sim} samples)')
    axes[j].set_xlabel(cs_name)
    axes[j].set_ylabel('Frequency')
    axes[j].grid(True)

# Hide the extra subplot (8th one)
axes[-3].set_visible(False)

# Adjust layout
plt.tight_layout()
plt.savefig('plots/GSA/GSA_CFF_comparison.png')
plt.show()

# %%

floor1 = cff_case_studies["industrial_floor"]
floor2 = cff_case_studies["composite_floor"]


CCB1 = (0.95 * cff_case_studies["industrial_CCB_al"]
        + 0.05 * cff_case_studies["industrial_CCB_st"])
CCB2 = (0.54 * cff_case_studies["composite_CCB_al"]
        + 0.045 * cff_case_studies["composite_CCB_st"]
        + 0.415 * cff_case_studies["composite_CCB_cp"])
# Now use the code from chatgpt with param_values from above to run the KS test and factor mapping

#%%
factor_mapping_inputs = {
    'industrial_floor':[np.transpose(param_values_all[5]), floor1],
    'composite_floor':[np.transpose(param_values_all[6]), floor2],
    'industrial_CCB':[np.transpose(np.concatenate((param_values_all[0], param_values_all[1]), axis=0)), CCB1],
    'composite_CCB':[np.transpose(np.concatenate((param_values_all[2], param_values_all[3], param_values_all[4]), axis=0)), CCB2],
}

for i, (cs_name, problem) in enumerate(factor_mapping_inputs.items()):
    # Assuming you already have:
    # - param_values: (n, d) matrix of sampled input parameters
    # - outputs: (n,) array of cff output
    param_values = problem[0]
    print('param_values: ', np.shape(param_values))
    outputs = problem[1]
    print('outputs: ', np.shape(outputs))

    if i==0: #industrial floor
        df = pd.DataFrame(param_values, columns=["a", "r1", "r2", "qp", "qs_in", "qs_out"])
        df['cff'] = outputs
    elif i==1: #composite floor
        df= pd.DataFrame(param_values, columns=["a", "r1", "r2", "qp", "qs_in", "qs_out"])
        df['cff'] = outputs
    elif i==2: #industrial CCB
        df = pd.DataFrame(param_values, columns=["a_al", "r1_al", "r2_al", "qp_al", "qs_in_al", "qs_out_al",
                                                 "a_st", "r1_st", "r2_st", "qp_st", "qs_in_st", "qs_out_st"])
        df['cff'] = outputs
    else: #composite CCB
        df = pd.DataFrame(param_values,
                          columns=["a_al", "r1_al", "r2_al", "qp_al", "qs_in_al", "qs_out_al",
                                   "a_st", "r1_st", "r2_st", "qp_st", "qs_in_st", "qs_out_st",
                                   "a_cp", "r1_cp", "r2_cp", "qp_cp", "qs_in_cp", "qs_out_cp"])
        df['cff'] = outputs

    # Define percentile-based regions
    df['cff_category'] = pd.qcut(df['cff'], q=[0, 0.25, 0.75, 1.0], labels=['Low', 'Medium', 'High'])

    # Extract Low & High impact groups
    low_group = df[df['cff_category'] == 'Low']
    high_group = df[df['cff_category'] == 'High']

    # Perform KS test for each parameter
    ks_results = []  # Store results for all comparisons

    if i == 0 or i == 1:  # Industrial/Composite Floor
        params = ["a", "r1", "r2", "qp", "qs_in", "qs_out"]

    elif i == 2:  # Industrial CCB
        params = ["a_al", "r1_al", "r2_al", "qp_al", "qs_in_al", "qs_out_al",
                  "a_st", "r1_st", "r2_st", "qp_st", "qs_in_st", "qs_out_st"]

    else:  # Composite CCB
        params = ["a_al", "r1_al", "r2_al", "qp_al", "qs_in_al", "qs_out_al",
                  "a_st", "r1_st", "r2_st", "qp_st", "qs_in_st", "qs_out_st",
                  "a_cp", "r1_cp", "r2_cp", "qp_cp", "qs_in_cp", "qs_out_cp"]

    # Extract impact groups
    low_group = df[df['cff_category'] == 'Low']
    medium_group = df[df['cff_category'] == 'Medium']
    high_group = df[df['cff_category'] == 'High']

    # Perform KS test for each parameter and region pair
    for param in params:
        ks_stat_low_medium, _ = ks_2samp(low_group[param], medium_group[param])
        ks_stat_low_high, _ = ks_2samp(low_group[param], high_group[param])
        ks_stat_medium_high, _ = ks_2samp(medium_group[param], high_group[param])

        ks_results.append({
            "Parameter": param,
            "Low-Medium": ks_stat_low_medium,
            "Low-High": ks_stat_low_high,
            "Medium-High": ks_stat_medium_high
        })

    # Convert results to DataFrame
    ks_df = pd.DataFrame(ks_results).set_index("Parameter")

    # Plot KS heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(ks_df, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title(f"KS Factor Mapping Heatmap - {cs_name}")
    plt.xlabel("KS Test (Category Comparison)")
    plt.ylabel("Input Variables")
    plt.savefig(f'plots/factor mapping/{cs_name}_KS test.png')
    plt.show()
