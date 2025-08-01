import pandas as pd
import numpy as np

# Load your original dataset
df = pd.read_csv("mental_dataset.csv")

# Add Q1 to Q10 with random Likert-scale values (0 to 3)
for i in range(1, 11):
    df[f"Q{i}"] = np.random.randint(0, 4, size=len(df))

# Save updated dataset
df.to_csv("updated_dataset.csv", index=False)

print("✅ Q1 to Q10 columns added with dummy values!")
