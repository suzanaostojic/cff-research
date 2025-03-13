
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
    {
        "name": "concrete industrial floor", 
        "excel_column_name": "industrial_floor", 
        "a_min": 0.5, 
        "a_max": 0.5,
        "a_mean": 0.5,
        "r1_min": 0,
        "r1_max": 0, 
        "r1_mean": 0,
        "r2_min": 0,
        "r2_max": 0.7,
        "r2_mean": 0.35,
        "qp_min": 1,
        "qp_max": 1,
        "qp_mean": 1,
        "qs_in_min": 0.73,
        "qs_in_max": 1,
        "qs_in_mean": 0.86,
        "qs_out_min": 0.48,
        "qs_out_max": 1,
        "qs_out_mean": 0.74
    },
    {
        "name": "concrete composite floor", 
        "excel_column_name": "composite_floor", 
        "a_min": 0.5,
        "a_max": 0.7,
        "a_mean": 0.6,
        "r1_min": 0.6,
        "r1_max": 0.6,
        "r1_mean": 0.6,
        "r2_min": 0,
        "r2_max": 0.97,
        "r2_mean": 0.485,
        "qp_min": 1,
        "qp_max": 1,
        "qp_mean": 1,
        "qs_in_min": 0.73,
        "qs_in_max": 1,
        "qs_in_mean": 0.86,
        "qs_out_min": 0.23,
        "qs_out_max": 0.81,
        "qs_out_mean": 0.52
    },
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

        cff_case_study_min = cff(case_study["a_min"], case_study["r1_min"], case_study["r2_min"], case_study["qp_min"], case_study["qs_in_min"], case_study["qs_out_min"], ev, ev_star, erec, erec_eol, ed)
        cff_case_study_max = cff(case_study["a_max"], case_study["r1_max"], case_study["r2_max"], case_study["qp_max"], case_study["qs_in_max"], case_study["qs_out_max"], ev, ev_star, erec, erec_eol, ed)
        cff_case_study_mean = cff(case_study["a_mean"], case_study["r1_mean"], case_study["r2_mean"], case_study["qp_mean"], case_study["qs_in_mean"], case_study["qs_out_mean"], ev, ev_star, erec, erec_eol, ed)

        S_a_min = d_da(case_study["r1_min"], case_study["r2_min"], case_study["qp_min"], case_study["qs_in_min"], case_study["qs_out_min"], ev, ev_star, erec, erec_eol)
        S_a_max = d_da(case_study["r1_max"], case_study["r2_max"], case_study["qp_max"], case_study["qs_in_max"], case_study["qs_out_max"], ev, ev_star, erec, erec_eol)
        S_a_mean = d_da(case_study["r1_mean"], case_study["r2_mean"], case_study["qp_mean"], case_study["qs_in_mean"], case_study["qs_out_mean"], ev, ev_star, erec, erec_eol)
        S_r1_min = d_dr1(case_study["a_min"], case_study["qp_min"], case_study["qs_in_min"], ev, erec)
        S_r1_max = d_dr1(case_study["a_max"], case_study["qp_max"], case_study["qs_in_max"], ev, erec)
        S_r1_mean = d_dr1(case_study["a_mean"], case_study["qp_mean"], case_study["qs_in_mean"], ev, erec)
        S_r2_min = d_dr2(case_study["a_min"], case_study["qp_min"], case_study["qs_out_min"], ev_star, erec_eol, ed)
        S_r2_max = d_dr2(case_study["a_max"], case_study["qp_max"], case_study["qs_out_max"], ev_star, erec_eol, ed)
        S_r2_mean = d_dr2(case_study["a_mean"], case_study["qp_mean"], case_study["qs_out_mean"], ev_star, erec_eol, ed)
        S_qp_min = d_qp(case_study["a_min"], case_study["r1_min"], case_study["r2_min"], case_study["qp_min"], case_study["qs_in_min"], case_study["qs_out_min"], ev, ev_star)
        S_qp_max = d_qp(case_study["a_max"], case_study["r1_max"], case_study["r2_max"], case_study["qp_max"], case_study["qs_in_max"], case_study["qs_out_max"], ev, ev_star)
        S_qp_mean = d_qp(case_study["a_mean"], case_study["r1_mean"], case_study["r2_mean"], case_study["qp_mean"], case_study["qs_in_mean"], case_study["qs_out_mean"], ev, ev_star)
        S_qs_in_min = d_dqs_in(case_study["a_min"], case_study["r1_min"], case_study["qp_min"], ev)
        S_qs_in_max = d_dqs_in(case_study["a_max"], case_study["r1_max"], case_study["qp_max"], ev)
        S_qs_in_mean = d_dqs_in(case_study["a_mean"], case_study["r1_mean"], case_study["qp_mean"], ev)
        S_qs_out_min = d_dqs_out(case_study["a_min"], case_study["r2_min"], case_study["qp_min"], ev_star)
        S_qs_out_max = d_dqs_out(case_study["a_max"], case_study["r2_max"], case_study["qp_max"], ev_star)
        S_qs_out_mean = d_dqs_out(case_study["a_mean"], case_study["r2_mean"], case_study["qp_mean"], ev_star)
        
        result = {
            "Case Study": case_study["name"],
            "Impact Category": impact_category,
            "S_a_min": S_a_min,
            "S_a_max": S_a_max,
            "S_a_mean": S_a_mean,
            "S_r1_min": S_r1_min,
            "S_r1_max": S_r1_max,
            "S_r1_mean": S_r1_mean,
            "S_r2_min": S_r2_min,
            "S_r2_max": S_r2_max,
            "S_r2_mean": S_r2_mean,
            "S_qp_min": S_qp_min,
            "S_qp_max": S_qp_max,
            "S_qp_mean": S_qp_mean,
            "S_qs_in_min": S_qs_in_min,
            "S_qs_in_max": S_qs_in_max,
            "S_qs_in_mean": S_qs_in_mean,
            "S_qs_out_min": S_qs_out_min,
            "S_qs_out_max": S_qs_out_max,
            "S_qs_out_mean": S_qs_out_mean,
            "RS_a_min": S_a_min * case_study["a_min"] / cff_case_study_min,
            "RS_a_max": S_a_max * case_study["a_max"] / cff_case_study_max,
            "RS_a_mean": S_a_mean * case_study["a_mean"] / cff_case_study_mean,
            "RS_r1_min": S_r1_min * case_study["r1_min"] / cff_case_study_min,
            "RS_r1_max": S_r1_max * case_study["r1_max"] / cff_case_study_max,
            "RS_r1_mean": S_r1_mean * case_study["r1_mean"] / cff_case_study_mean,
            "RS_r2_min": S_r2_min * case_study["r2_min"] / cff_case_study_min,
            "RS_r2_max": S_r2_max * case_study["r2_max"] / cff_case_study_max,
            "RS_r2_mean": S_r2_mean * case_study["r2_mean"] / cff_case_study_mean,
            "RS_qp_min": S_qp_min * case_study["qp_min"] / cff_case_study_min,
            "RS_qp_max": S_qp_max * case_study["qp_max"] / cff_case_study_max,
            "RS_qp_mean": S_qp_mean * case_study["qp_mean"] / cff_case_study_mean,
            "RS_qs_in_min": S_qs_in_min * case_study["qs_in_min"] / cff_case_study_min,
            "RS_qs_in_max": S_qs_in_max * case_study["qs_in_max"] / cff_case_study_max,
            "RS_qs_in_mean": S_qs_in_mean * case_study["qs_in_mean"] / cff_case_study_mean,
            "RS_qs_out_min": S_qs_out_min * case_study["qs_out_min"] / cff_case_study_min,
            "RS_qs_out_max": S_qs_out_max * case_study["qs_out_max"] / cff_case_study_max,
            "RS_qs_out_mean": S_qs_out_mean * case_study["qs_out_mean"] / cff_case_study_mean,
        }
        derivatives_results.append(result)

# Save to CSV
derivatives_df = pd.DataFrame(derivatives_results)
derivatives_df.to_csv("derivatives_results.csv", index=False)

print("Derivative results saved to 'derivatives_results.csv'")
