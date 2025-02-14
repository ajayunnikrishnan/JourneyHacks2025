import pandas as pd
import numpy as np
import ace_tools as tools

# Define the weights for each feature
WEIGHTS = {
    'age': 0.3,
    'blood_pressure': 0.2,
    'heart_rate': 0.15,
    'oxygen_saturation': -0.25,  # Negative because lower saturation is worse
    'respiratory_rate': 0.2,
    'cholesterol_level': 0.1,
    'bmi': 0.15,  # Higher BMI may indicate health risks
    'glucose_level': 0.2,  # Higher glucose levels can be concerning
    'smoking_status': 0.3,  # Smoking contributes to health risks
    'exercise_frequency': -0.2  # More exercise is beneficial, so negative weight
}

def calculate_severity_score(patient, weights):
    """
    Calculate severity score based on the weighted sum of patient features.
    """
    score = sum(weights[feature] * patient[feature] for feature in weights if feature in patient)
    return score

def process_patient_data(csv_file=None):
    """
    Read patient data from a CSV file or generate sample data, compute severity scores, and sort patients.
    """
    # Load CSV into pandas DataFrame
    df = pd.read_csv(csv_file)

    # Ensure all weight keys exist in the dataset
    required_features = set(WEIGHTS.keys())
    missing_features = required_features - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing required features in dataset: {missing_features}")

    # Compute severity scores
    df['Severity_Score'] = df.apply(lambda row: calculate_severity_score(row, WEIGHTS), axis=1)

    # Sort patients by severity score in descending order
    df_sorted = df.sort_values(by='Severity_Score', ascending=False)

    # Display the sorted data
    tools.display_dataframe_to_user(name="Sorted Patient Data", dataframe=df_sorted)

    return df_sorted

# Example usage
if __name__ == "__main__":
    csv_filename = "../data/patient_data.csv"  # Update with the actual file path if needed
    sorted_patients = process_patient_data(csv_filename)
