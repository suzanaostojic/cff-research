
import numpy as np
import pandas as pd

# Define CFF function and its partial derivatives

def cff(a, r1, r2, qp, qs_in, qs_out, ev, ev_star, erec, erec_eol, ed):
    return (1-r1)*ev + r1*(a*erec+(1-a)*ev*qs_in/qp) + (1-a)*r2*(erec_eol-ev_star*qs_out/qp)+(1 - r2)*ed

def d_da(r1, r2, qp, qs_in, qs_out, ev, ev_star, erec, erec_eol):
    return r1*(erec - ev*qs_in/qp) - r2*(erec_eol - ev_star*qs_out/qp)

def d_dr1(a, qp, qs_in, ev, erec):
    return a*erec - ev + ev*qs_in*(1 - a)/qp

def d_dr2(a, qp, qs_out, ev_star, erec_eol, ed):
    return -ed + (1 - a)*(erec_eol - ev_star*qs_out/qp)

def d_qp(a, r1, r2, qp, qs_in, qs_out, ev, ev_star):
    return -ev*qs_in*r1*(1 - a)/qp**2 + ev_star*qs_out*r2*(1 - a)/qp**2

def d_dqs_in(a, r1, qp, ev):
    return ev*r1*(1 - a)/qp

def d_dqs_out(a, r2, qp, ev_star):
    return -ev_star*r2*(1 - a)/qp

# Define case study parameters
case_studies = [
    {"name": "concrete industrial floor", "excel_column_name": "industrial_floor", "a": 0.5, "r1": 0, "r2": np.mean([0, 0.7]), "qp": 1, "qs_in": 0.52, "qs_out": np.mean([0.5, 1])},
    {"name": "carbon concrete industrial floor", "excel_column_name": "composite_floor", "a": np.mean([0.5, 0.7]), "r1": 0.6, "r2": np.mean([0, 0.8]), "qp": 1, "qs_in": np.mean([0.3, 0.6]), "qs_out": np.mean([0.5, 1])},
]

# Load data from Excel
impact_data = pd.read_excel("data/CFF Paper Data V10_Clean_SO.xlsx", sheet_name="python_impact")

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
    excel_column_name = case_study["excel_column_name"]
    if excel_column_name not in impact_data.columns:
        print(f"Warning: Column '{excel_column_name}' not found in data. Skipping case study '{case_study['name']}'.")
        continue

    impact_values = impact_data[excel_column_name]

    for impact_category in impact_categories:
        erec = impact_values.iloc[impact_index_map[f"erec_{impact_category}"]]
        erec_eol = impact_values.iloc[impact_index_map[f"erec_eol_{impact_category}"]]
        ed = impact_values.iloc[impact_index_map[f"ed_{impact_category}"]]
        ev = impact_values.iloc[impact_index_map[f"ev_{impact_category}"]]
        ev_star = impact_values.iloc[impact_index_map[f"ev_star_{impact_category}"]]

        cff_case_study = cff(case_study["a"], case_study["r1"], case_study["r2"], case_study["qp"], case_study["qs_in"], case_study["qs_out"], ev, ev_star, erec, erec_eol, ed)
        S_a = d_da(case_study["r1"], case_study["r2"], case_study["qp"], case_study["qs_in"], case_study["qs_out"], ev, ev_star, erec, erec_eol)
        S_r1 = d_dr1(case_study["a"], case_study["qp"], case_study["qs_in"], ev, erec)
        S_r2 = d_dr2(case_study["a"], case_study["qp"], case_study["qs_out"], ev_star, erec_eol, ed)
        S_qp = d_qp(case_study["a"], case_study["r1"], case_study["r2"], case_study["qp"], case_study["qs_in"], case_study["qs_out"], ev, ev_star)
        S_qs_in = d_dqs_in(case_study["a"], case_study["r1"], case_study["qp"], ev)
        S_qs_out = d_dqs_out(case_study["a"], case_study["r2"], case_study["qp"], ev_star)

        result = {
            "Case Study": case_study["name"],
            "Impact Category": impact_category,
            "S_a": S_a,
            "S_r1": S_r1,
            "S_r2": S_r2,
            "S_qp": S_qp,
            "S_qs_in": S_qs_in,
            "S_qs_out": S_qs_out,
            "RS_a": S_a * case_study["a"] / cff_case_study,
            "RS_r1": S_r1 * case_study["r1"] / cff_case_study,
            "RS_r2": S_r2 * case_study["r2"] / cff_case_study,
            "RS_qp": S_qp * case_study["qp"] / cff_case_study,
            "RS_qs_in": S_qs_in * case_study["qs_in"] / cff_case_study,
            "RS_qs_out": S_qs_out * case_study["qs_out"] / cff_case_study,
        }
        derivatives_results.append(result)

# Save to CSV
derivatives_df = pd.DataFrame(derivatives_results)
derivatives_df.to_csv("derivatives_results.csv", index=False)

print("Derivative results saved to 'derivatives_results.csv'")
