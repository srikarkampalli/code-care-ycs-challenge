"""
This module will analyze the correlation between the risk score and other variables in the CodeCare dataset.
"""

import pandas as pd

# Load the cleaned CodeCare dataset
codecare_df = pd.read_csv("../../data/codecare_data.csv")

# List of columns to analyze for correlation with the risk score
X_values = [
    "total_beds_7_day_avg",
    "a_adult_hospbeds_7d_avg",
    "a_adult_hospinpbeds_7d_avg",
    "inpatient_beds_used_7_day_avg",
    "a_adult_hospinpbed_occ_7d_avg",
    "adults_hospconfsusp_7d_avg",
    "adults_hospconf_7d_avg",
    "pediatric_hospconfsusp_7d_avg",
    "pediatric_hospconf_7d_avg",
    "inpatient_beds_7_day_avg",
    "total_icu_beds_7_day_avg",
    "t_staff_adult_icubeds_7d_avg",
    "icu_beds_used_7_day_avg",
    "staff_adult_icubed_occ_7d_avg",
    "staff_icuadultconfsusp_7d_avg",
    "staff_icuadultconf_7_avg",
    "patients_hospconf_flu_7d_avg",
    "icu_patients_conf_flu_7d_avg",
    "hospconf_flucovid_7d_avg",
    "total_beds_7_day_sum",
    "a_adult_hospbeds_7d_sum",
    "a_adult_hospinpbeds_7d_sum",
    "inpatient_beds_used_7_day_sum",
    "a_adult_hospinpbed_occ_7d_sum",
    "adults_hospconfsusp_7d_sum",
    "adults_hospconf_7d_sum",
    "pediatric_hospconfsusp_7d_sum",
    "pediatric_hospconf_7d_sum",
    "inpatient_beds_7_day_sum",
    "total_icu_beds_7_day_sum",
    "t_staff_adult_icubeds_7d_sum",
    "icu_beds_used_7_day_sum",
    "staff_adult_icubed_occ_7d_sum",
    "staff_icuadultconfsusp_7d_sum",
    "staff_icuadultconf_7d_sum",
    "patients_hospconf_flu_7d_sum",
    "icu_patients_conf_flu_7d_sum",
    "hospconf_flucovid_7d_sum",
    "total_beds_7_day_coverage",
    "a_adult_hospbeds_7d_cov",
    "a_adult_hospinpbeds_7d_cov",
    "inpatient_beds_used_7d_cov",
    "a_adult_hospinpbed_occ_7d_cov",
    "adults_hospconfsusp_7d_cov",
    "adults_hospconf_7d_cov",
    "pediatric_hospconfsusp_7d_cov",
    "pediatric_hospconf_7d_cov",
    "inpatient_beds_7_day_coverage",
    "total_icu_beds_7_day_coverage",
    "t_staff_adult_icubeds_7d_cov",
    "icu_beds_used_7_day_coverage",
    "staff_adult_icubed_occ_7d_cov",
    "staff_icuadultconfsusp_7d_cov",
    "staff_icuadultconf_7_cov",
    "patients_hospconf_flu_7d_cov",
    "icu_patients_conf_flu_7d_cov",
    "hospconf_flucovid_7d_cov",
    "prevadmit_adult_conf_7d_sum",
    "prevadmit_1819_conf_7d_sum",
    "prevadmit_2029_conf_7d_sum",
    "prevadmit_3039_conf_7d_sum",
    "prevadmit_4049_conf_7d_sum",
    "prevadmit_5059_conf_7d_sum",
    "prevadmit_6069_conf_7d_sum",
    "prevadmit_7079_conf_7d_sum",
    "prevadmit_80p_conf_7d_sum",
    "prevadmit_conf_unk_7d_sum",
    "prevadmit_pedi_conf_7d_sum",
    "prevday_covidED_visits_7d_sum",
    "prevadmit_adult_susp_7d_sum",
    "prevadmit_1819_susp_7d_sum",
    "prevadmit_2029_susp_7d_sum",
    "prevadmit_3039_susp_7d_sum",
    "prevadmit_4049_susp_7d_sum",
    "prevadmit_5059_susp_7d_sum",
    "prevadmit_6069_susp_7d_sum",
    "prevadmit_7079_susp_7d_sum",
    "prevadmit_80p_susp_7d_sum",
    "prevadmit_susp_unk_7d_sum",
    "prevadmit_pedi_susp_7d_sum",
    "prevday_totED_visits_7d_sum",
    "prevadmit_flu_conf_7d_sum",
]

# Get correlation matrix
correlation_matrix = codecare_df[X_values + ["risk_score"]].corr()

# Extract correlations with risk score
risk_score_correlation = correlation_matrix["risk_score"].drop("risk_score")

suitable_vars = []

# Print the correlations and rank
print("Correlation of variables with Risk Score:")
for variable, correlation in risk_score_correlation.sort_values(
    ascending=False
).items():
    print(f"{variable}: {correlation:.4f}")
    if correlation >= 0.25 or correlation <= -0.25:
        # If there is a sign of sufficient correlation (above weak) based on r, then append
        suitable_vars.append(variable)

print("\nVariables with strong correlation (>= 0.25 or <= -0.25) with Risk Score:")
print(suitable_vars)
