"""
This module will help to do the first step in data cleaning for the CodeCare dataset.
It will load the dataset, rename columns for clarity, drop unnecessary columns,
ensure no negative values in specific columns, and create a risk score and risk class/label
based on hospital bed occupancy, ICU occupancy, infectious pressure, and emergency department pressure.
"""

# Import in pandas for data manipulation
import pandas as pd
from scipy.stats import boxcox

# Load the CodeCare dataset from a CSV file
codecare_df = pd.read_csv("../../data/codecare_data.csv")

# Rename columns for better clarity
codecare_df.rename(
    # Rename specific columns to more descriptive names
    columns={
        "X": "latitude",
        "Y": "longitude",
        "collection_week": "date",
    },
    # Perform the renaming operation in place
    inplace=True,
)

# Drop unnecessary columns from the dataset
codecare_df = codecare_df.drop(
    columns=[
        "OBJECTID",
        "hospital_pk",
        "ccn",
        "fips_code",
        "is_metro_micro",
        "last_updated",
    ],
)

# Ensure that specific columns have no negative values by clipping them at zero
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

# Clip negative values to zero for the specified columns
codecare_df[X_values] = codecare_df[X_values].clip(lower=1)

# Now, create the y-value (risk)

# Calculate intermediate metrics for risk score calculation
codecare_df["bed_occupancy"] = (
    codecare_df["inpatient_beds_used_7_day_avg"] / codecare_df["total_beds_7_day_avg"]
)

# Calculate ICU occupancy
codecare_df["icu_occupancy"] = (
    codecare_df["icu_beds_used_7_day_avg"] / codecare_df["total_icu_beds_7_day_avg"]
)

# Calculate infectious pressure
codecare_df["infectious_pressure"] = (
    codecare_df["adults_hospconf_7d_avg"] + codecare_df["hospconf_flucovid_7d_avg"]
) / codecare_df["total_beds_7_day_avg"]

# Calculate emergency department pressure
codecare_df["ed_pressure"] = (
    codecare_df["prevday_totED_visits_7d_sum"] / codecare_df["total_beds_7_day_avg"]
)

# Combine the metrics into a single risk score
codecare_df["risk_score"] = (
    0.5 * codecare_df["bed_occupancy"]
    + 0.3 * codecare_df["icu_occupancy"]
    + 0.1 * codecare_df["infectious_pressure"]
    + 0.1 * codecare_df["ed_pressure"]
)

# Normalize the risk score to a 0-1 scale
codecare_df["risk_score"] = (
    codecare_df["risk_score"] - codecare_df["risk_score"].min()
) / (codecare_df["risk_score"].max() - codecare_df["risk_score"].min()) + 0.01

codecare_df["risk_score"], lambda_value = boxcox(codecare_df["risk_score"])

codecare_df["risk_score"] = codecare_df["risk_score"] + 125

codecare_df["risk_score"] = (
    codecare_df["risk_score"] - codecare_df["risk_score"].min()
) / (codecare_df["risk_score"].max() - codecare_df["risk_score"].min())

# Create risk classes/labels based on the risk score
codecare_df["risk_class"] = pd.cut(
    codecare_df["risk_score"],
    bins=[0, 0.25, 0.50, 0.75, 1.0],
    labels=[1, 2, 3, 4],
    include_lowest=True,
)

# Output the summary statistics of the risk score and save the cleaned dataset to a new CSV file
codecare_df.to_csv("../data/codecare_data.csv", index=False)
