import os
import json
import pandas as pd

def split_json(input_json_file, GT_REPORTS, PREDICTED_REPORTS):
    """
    Converts a JSON file of predictions into ground truth and prediction CSV files.
    
    Parameters:
    - input_json_file: str, path to the input JSON file
    - GT_REPORTS: str, path to the output ground truth CSV file
    - PREDICTED_REPORTS: str, path to the output prediction CSV file
    """
    with open(input_json_file, 'r') as file:
        input_data_dict = json.load(file)

    gt_reports = []
    predicted_reports = []

    for study_id, input_data_idx in input_data_dict.items():
        model_prediction = input_data_idx['model_prediction']
        findings = input_data_idx['section_findings']
        impression = input_data_idx['section_impression']

        # Handle NaN values
        findings = findings if pd.notna(findings) else None
        impression = impression if pd.notna(impression) else None

        # This model predicts only the "Findings". So no concatenation is needed
        # groundtruth = f"Findings: {findings} Impression: {impression}"
        groundtruth = findings
        predicted_reports.append([study_id, model_prediction])
        gt_reports.append([study_id, groundtruth])

    # Convert to DataFrame
    predicted_reports_df = pd.DataFrame(predicted_reports, columns=['study_id', 'report'])
    gt_reports_df = pd.DataFrame(gt_reports, columns=['study_id', 'report'])

    # Save to CSV files
    predicted_reports_df.to_csv(PREDICTED_REPORTS, index=False)
    gt_reports_df.to_csv(GT_REPORTS, index=False)
    
    # Check for NaN values in 'report' column of gt_reports_df
    if gt_reports_df['report'].isnull().values.any():
        print("WARNING: There are NaN values in 'report' column of gt_reports_df.")

    # Check for NaN values in 'report' column of predicted_reports_df
    if predicted_reports_df['report'].isnull().values.any():
        print("WARNING: There are NaN values in 'report' column of predicted_reports_df.")

input_json_file = 'results/iu_xray/MoERad.json'
GT_REPORTS = 'results/iu_xray/gts.csv'
PREDICTED_REPORTS = 'results/iu_xray/preds.csv'

split_json(input_json_file, GT_REPORTS, PREDICTED_REPORTS)