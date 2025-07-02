# ---------- train_model.py ----------
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset (make sure the CSV is in the same folder)
df = pd.read_csv('mental_health_data.csv')

# Input features (Q1 to Q10)
X = df[[f"Q{i+1}" for i in range(10)]]

# Output labels
y_anxiety = df['anxiety']
y_depression = df['depression']
y_stress = df['stress']

# Train models
model_anxiety = RandomForestClassifier()
model_anxiety.fit(X, y_anxiety)

model_depression = RandomForestClassifier()
model_depression.fit(X, y_depression)

model_stress = RandomForestClassifier()
model_stress.fit(X, y_stress)

# Save models to files
with open('model_anxiety.pkl', 'wb') as f:
    pickle.dump(model_anxiety, f)

with open('model_depression.pkl', 'wb') as f:
    pickle.dump(model_depression, f)

with open('model_stress.pkl', 'wb') as f:
    pickle.dump(model_stress, f)

print("\u2705 Models trained and saved!")