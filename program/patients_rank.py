import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor


# --------------------------
# Step 1: Train an AI Model with Expanded Training Data
# --------------------------
# Expanded training data: condition descriptions mapped to urgency scores (scale 1-10)
train_conditions = [
    "severe chest pain",              # 9
    "high fever",                     # 7
    "mild headache",                  # 2
    "broken arm",                     # 5
    "severe injury",                  # 8
    "light cold",                     # 1
    "difficulty breathing",           # 8
    "dizziness",                      # 4
    "unconscious",                    # 10
    "stroke symptoms",                # 10
    "heart attack symptoms",          # 10
    "severe allergic reaction",       # 9
    "shortness of breath",            # 8
    "severe abdominal pain",          # 8
    "loss of consciousness",          # 10
    "severe burns",                   # 9
    "extreme fatigue",                # 3
    "intense migraine",               # 4
    "severe back pain",               # 5
    "worsening chest tightness",      # 9
    "sudden vision loss",             # 8
    "severe joint pain",              # 5
    "acute anxiety attack",           # 6
    "persistent vomiting",            # 7
    "uncontrolled bleeding",          # 10
    "high blood pressure crisis",     # 9
    "severe dehydration",             # 7
    "moderate respiratory distress",  # 7
    "sudden weakness"                 # 6
]


train_urgency = [
    9, 7, 2, 5, 8, 1, 8, 4,
    10, 10, 10, 9, 8, 8, 10, 9,
    3, 4, 5, 9, 8, 5, 6, 7,
    10, 9, 7, 7, 6
]


# Create a pipeline that vectorizes text and trains a regressor
model_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("regressor", RandomForestRegressor(n_estimators=50, random_state=42))
])
model_pipeline.fit(train_conditions, train_urgency)


# Function to predict urgency using the AI model
def assign_urgency_with_ai(condition):
    predicted_urgency = model_pipeline.predict([condition])[0]
    return int(round(predicted_urgency))


# --------------------------
# Step 2: Read Patient Data from a Text File Using pandas
# --------------------------
# Ensure you have a file named 'patients.txt' (or 'patients.csv') with the following content:
# patient_id,name,age,condition_description,urgency_rank
# 1,Alice,30,"chest pain and difficulty breathing",0
# 2,Bob,45,"high fever and severe headache",0
# 3,Charlie,25,"severe injury in accident",0
# 4,Dana,50,"mild headache and slight dizziness",0


file_path = 'patients.txt'  # Update with your file path if necessary
patients_df = pd.read_csv(file_path)


# --------------------------
# Step 3: Update Urgency Rank Using the AI Model
# --------------------------
patients_df['urgency_rank'] = patients_df['condition_description'].apply(assign_urgency_with_ai)


# Convert DataFrame to list of lists for custom sorting.
# Each record is represented as: [patient_id, name, age, condition_description, urgency_rank]
patients_list = patients_df.values.tolist()


# --------------------------
# Step 4: Implement Merge Sort to Sort Patients by Urgency Rank (Descending)
# --------------------------
def merge_sort(records):
    """
    Sorts a list of patient records based on urgency_rank (index 4) in descending order.
    """
    if len(records) <= 1:
        return records
    mid = len(records) // 2
    left = merge_sort(records[:mid])
    right = merge_sort(records[mid:])
    return merge(left, right)


def merge(left, right):
    sorted_list = []
    i = j = 0
    # Merge the two sorted lists
    while i < len(left) and j < len(right):
        # Compare urgency_rank (index 4) for descending order.
        if left[i][4] >= right[j][4]:
            sorted_list.append(left[i])
            i += 1
        else:
            sorted_list.append(right[j])
            j += 1
    # Append any remaining elements
    sorted_list.extend(left[i:])
    sorted_list.extend(right[j:])
    return sorted_list


sorted_patients_list = merge_sort(patients_list)


# --------------------------
# Step 5: Display the Sorted Patient Data
# --------------------------
print("Sorted Patients by AI-Predicted Urgency (Highest to Lowest):")
for patient in sorted_patients_list:
    print(patient)



