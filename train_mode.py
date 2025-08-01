import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle
import os

# Load dataset
df = pd.read_csv("updated_dataset.csv")

# Feature columns from frontend questions
feature_cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']
X = df[feature_cols]

# Target columns
y_stress = df['Stress Level']
y_anxiety = df['Anxiety Level']
y_depression = df['Depression Level']

# Train-test split
X_train, X_test, y_stress_train, y_stress_test = train_test_split(X, y_stress, test_size=0.2, random_state=42)
_, _, y_anxiety_train, y_anxiety_test = train_test_split(X, y_anxiety, test_size=0.2, random_state=42)
_, _, y_depression_train, y_depression_test = train_test_split(X, y_depression, test_size=0.2, random_state=42)

# Train models

model_stress = RandomForestClassifier(class_weight='balanced').fit(X, y_stress)
model_anxiety = RandomForestClassifier(class_weight='balanced').fit(X, y_anxiety)
model_depression = RandomForestClassifier(class_weight='balanced').fit(X, y_depression)

# Anxiety Feature Importance
features = X.columns
importances = model_anxiety.feature_importances_

plt.figure(figsize=(10, 5))
plt.barh(features, importances)
plt.title("Feature Importance for Anxiety Prediction")
plt.xlabel("Importance Score")
plt.show()


# Predict
pred_stress = model_stress.predict(X_test)
pred_anxiety = model_anxiety.predict(X_test)
pred_depression = model_depression.predict(X_test)

# Accuracy scores
acc_stress = accuracy_score(y_stress_test, pred_stress)
acc_anxiety = accuracy_score(y_anxiety_test, pred_anxiety)
acc_depression = accuracy_score(y_depression_test, pred_depression)

print("✅ Stress Model Accuracy:", acc_stress)
print("✅ Anxiety Model Accuracy:", acc_anxiety)
print("✅ Depression Model Accuracy:", acc_depression)

# Save models in 'model' folder
os.makedirs("model", exist_ok=True)

with open('model/stress_model.pkl', 'wb') as f:
    pickle.dump(model_stress, f)

with open('model/anxiety_model.pkl', 'wb') as f:
    pickle.dump(model_anxiety, f)

with open('model/depression_model.pkl', 'wb') as f:
    pickle.dump(model_depression, f)
