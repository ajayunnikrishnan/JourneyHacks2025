import pandas as pd

# Define the weights for each feature
WEIGHTS = {
    'age': 0.10,                        
    'chest_pain': 0.30,               
    'difficulty_breathing': 0.25,     
    'unconsciousness': 0.35,            
    'stroke_symptoms': 0.35,            
    'severe_injury': 0.20,             
    'uncontrolled_bleeding': 0.45,      
    'severe_allergic_reaction': 0.35,   
    'high_fever': 0.15,               
    'severe_abdominal_pain': 0.20,       
    'persistent_vomiting': 0.10        
}
def calculate_severity_score(patient, weights):
    """
    Calculate severity score based on the weighted sum of patient features.
    """
    score = sum(weights[feature] * patient[feature] for feature in weights if feature in patient)
    return score

def process_patient_data(csv_file):
    """
    Read patient data from a CSV file, compute severity scores, and sort patients.
    """
    df = pd.read_csv(csv_file)

    # Ensure all required weight keys exist in the dataset
    required_features = set(WEIGHTS.keys())
    missing_features = required_features - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing required features in dataset: {missing_features}")

    # Compute severity scores
    df['Severity_Score'] = df.apply(lambda row: calculate_severity_score(row, WEIGHTS), axis=1)

    # Sort patients by severity score in descending order
    df_sorted = df.sort_values(by='Severity_Score', ascending=False)

    # Print sorted data in terminal
    print(df_sorted.to_string(index=False))

    return df_sorted

# Example usage
if __name__ == "__main__":
    csv_filename = "../data/patient_data.csv"  # Update with the actual file path if needed
    sorted_patients = process_patient_data(csv_filename)
