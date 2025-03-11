import numpy as np
import pandas as pd
from scipy.stats import triang, truncnorm
from SALib.sample import saltelli
from SALib.analyze import sobol

#%%
# Load the parameter table (if stored as a CSV)
# param_df = pd.read_excel(f"data/CFF Paper Data V10_Clean_SO.xlsx", sheet_name="python_distr")
# param_df = param_df.iloc[30:36, :]
# print(param_df)


def setup_case_study_distributions(case_study):
    distribution_data = pd.read_excel(f"data/CFF Paper Data V10_Clean_SO.xlsx", sheet_name="python_distr")

    # loading distribution data for the selected case study
    mask = (distribution_data["case_study"] == case_study)
    return distribution_data[mask]


#Here add the impact category calculations
def setup_case_study_impacts(case_study):

    gwp100_data = pd.read_excel(f"data/CFF Paper Data V10_Clean_SO.xlsx", sheet_name="python_impact", index_col=0)

    # loading gwp100 values for the selected case study
    erec = gwp100_data[case_study]["erec"]
    erec_eol = gwp100_data[case_study]["erec_eol"]
    ed = gwp100_data[case_study]["ed"]
    ev = gwp100_data[case_study]["ev"]
    ev_star = gwp100_data[case_study]["ev_star"]

    return [erec, erec_eol, ed, ev, ev_star]


# Function to sample from the defined distributions
def sample_distribution(row, n):
    """Samples N values based on the distribution type."""
    dist_type = row['distribution']
    p1, p2, p3, p4 = row['p1'], row['p2'], row.get('p3', None), row.get('p4', None)

    if dist_type == 'fixed':
        return np.full(n, p1)  # Constant value

    elif dist_type == 'triang':
        return triang.rvs((p2 - p1) / (p3 - p1), loc=p1, scale=(p3 - p1), size=n)

    elif dist_type == 'truncnorm':
        a, b = (p3 - p1) / p2, (p4 - p1) / p2  # Transform bounds for truncnorm
        return truncnorm.rvs(a, b, loc=p1, scale=p2, size=n)

    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


# Define the model function
def cff(a, r1, r2, qs_in, qs_out, e_rec=0, e_rec_eol=7.63, ed=10.2, ev=156, ev_star=156):
    return (
            (1 - r1) * ev
            + r1 * (a * e_rec + (1 - a) * ev * qs_in)
            + (1 - a) * r2 * (e_rec_eol - ev_star * qs_out)
            + (1 - r2) * ed
    )


def allocation(am1, am2, am3,
               a1, r11, r21, qs_in1, qs_out1,
               a2, r12, r22, qs_in2, qs_out2,
               a3=0, r13=0, r23=0, qs_in3=0, qs_out3=0):
    return (
            am1*cff(a1, r11, r21, qs_in1, qs_out1)
            + am2*cff(a2, r12, r22, qs_in2, qs_out2)
            + am3*cff(a3, r13, r23, qs_in3, qs_out3)
    )


def gsa_sobol(case_study, n=1000):
    # Define the number of samples for Sobol analysis
    # N is the base sample size for Saltelli


    param_df = setup_case_study_distributions(case_study)

    problem = {
        'num_vars': len(param_df),
        'names': param_df['CFF_param'].tolist(),
        'bounds': []  # Will be filled based on distributions
    }

    # Generate samples for each parameter
    samples = []
    for _, row in param_df.iterrows():
        problem['bounds'].append([0, 1])  # SALib requires bounds, but we sample separately
        samples.append(sample_distribution(row, n * (2 * len(problem['names']) + 2)))

    # Convert samples into a NumPy array
    param_values = np.column_stack(samples)
    # Evaluate the function for each sampled input
    y = np.array([cff(*params) for params in param_values])

    # Compute Sobol sensitivity indices
    sobol_indices = sobol.analyze(problem, y)

    # Print results
    print(f"{case_study}, First-order Sobol indices:")
    for name, s1 in zip(problem['names'], sobol_indices['S1']):
        print(f"{name}: {s1:.4f}")

    print(f"\n{case_study}, Total-order Sobol indices:")
    for name, st in zip(problem['names'], sobol_indices['ST']):
        print(f"{name}: {st:.4f}")

    # Store results in a NumPy array
    results_array = np.column_stack((sobol_indices['S1'], sobol_indices['ST']))

    return results_array  # First column: S1, Second column: ST


#%%
# Run the GSA for each case study
case_studies = ["industrial_CCB_al", "industrial_CCB_st", "composite_CCB_al", "composite_CCB_st", "composite_CCB_cp",
                "industrial_floor", "composite_floor"]

# Define the output file name
output_file = f'plots/sobol/sobol_analysis_results.xlsx'

# Open an ExcelWriter to save multiple sheets in one file
with pd.ExcelWriter(output_file) as writer:
    for cs in case_studies:
        print(f"Running Sobol analysis for: {cs}")

        # Run sensitivity analysis and get results as a NumPy array
        results = gsa_sobol(cs, n=100000)  # Ensure gsa_sobol() returns a NumPy array

        # Retrieve parameter names for the current case study
        param_names = setup_case_study_distributions(cs)['CFF_param'].tolist()

        # Convert results into a Pandas DataFrame
        df_sobol = pd.DataFrame(results, columns=["First Order (S1)", "Total Order (ST)"])
        df_sobol.insert(0, "Parameter", param_names)  # Add parameter names

        # Save the DataFrame in a new sheet in the Excel file
        df_sobol.to_excel(writer, sheet_name=cs, index=False)

        print(f"Completed: {cs}\n")

print(f"All Sobol indices saved in {output_file}!")

