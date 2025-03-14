#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

#%%
# Step 0: Load case study data
n_sim = 100000

case_studies = ["industrial_CCB_al", "industrial_CCB_st", "composite_CCB_al", "composite_CCB_st", "composite_CCB_cp",
                "industrial_floor", "composite_floor"]

gwp100_data = pd.read_excel(f"data/CFF Paper Data V10_Clean_SO.xlsx", sheet_name="python_impact", index_col=0)

distribution_data = pd.read_excel(f"data/CFF Paper Data V10_Clean_SO.xlsx", sheet_name="python_distr")
distribution_data = distribution_data.fillna(0)


#%%
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


#%%
# Step 2: Define the CFF function

def cff(a, r1, r2, qs_in, qs_out, e_rec=0, e_rec_eol=7.63, ed=10.2, ev=156, ev_star=156):
    # using the concrete industrial floor with GWP100
    # now qs_in = qsin/qp, qs_out = qs_out/qp
    return (1-r1)*ev + r1*(a*e_rec+(1-a)*ev*qs_in) + (1-a)*r2*(e_rec_eol-ev_star*qs_out) + (1 - r2)*ed


#%%
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

    save_dir = f'plots/GSA/{case_study}'
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 'GSA_parameter_distributions.png')
    plt.savefig(save_path)

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
        plt.savefig(f'plots/GSA/{case_study}/GSA_CFF.png')
        plt.show()

    return cff_samples


#%%
# Generating GSA results for all case studies

cff_all = []

for cs in case_studies:
    print(f"Case Study: {cs}")
    cff_cs = gsa_cff(cs, show_distribution=True, show_statistics=True, show_cff_histogram=True)
    cff_all.append(cff_cs)

#%%

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
    j=i

    if i>4:
        j = i+1
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

#%%

floor1 = cff_case_studies["industrial_floor"]
floor2 = cff_case_studies["composite_floor"]
# Plot histograms
plt.figure(figsize=(10, 6))
plt.hist(floor1, bins=50, color='blue', alpha=0.5, label='industrial floor', density=True)
plt.hist(floor2, bins=50, color='red', alpha=0.5, label='composite floor', density=True)

# Labels and legend
plt.xlabel("GWP100 (kgCO2,eq)")
plt.ylabel("Density")
plt.legend()
plt.title("Climate change impact results for the floor case studies")
plt.savefig('plots/GSA/floor_comparison.png')
plt.show()

CCB1 = (0.95*cff_case_studies["industrial_CCB_al"]
        + 0.05*cff_case_studies["industrial_CCB_st"])
CCB2 = (0.54*cff_case_studies["composite_CCB_al"]
        + 0.045*cff_case_studies["composite_CCB_st"]
        + 0.415*cff_case_studies["composite_CCB_cp"])
# Plot histograms
plt.figure(figsize=(10, 6))
plt.hist(CCB1, bins=50, color='blue', alpha=0.5, label='industrial CCB', density=True)
plt.hist(CCB2, bins=50, color='red', alpha=0.5, label='composite CCB', density=True)

# Labels and legend
plt.xlabel("GWP100 (kgCO2,eq)")
plt.ylabel("Density")
plt.legend()
plt.title("Climate change impact results for the cross car beam case studies")
plt.savefig('plots/GSA/CCB_comparison.png')
plt.show()