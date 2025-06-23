import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# File path
file_path = "position_recommendations.csv"

# Step 1: Check if the file exists
if not os.path.exists(file_path):
    print("❌ File 'position_recommendations.csv' not found. Please run the prediction script first.")
    exit()

# Step 2: Load the file
df = pd.read_csv(file_path)

# Step 3: Show pie chart for assessment distribution
print("📊 Assessment distribution:")
assessment_counts = df["Assessment"].value_counts()
print(assessment_counts)

plt.figure(figsize=(6, 6))
assessment_counts.plot.pie(autopct='%1.1f%%', startangle=140)
plt.title("Assessment Summary of Player Position Fit")
plt.ylabel("")  # Remove y-label
plt.tight_layout()
plt.show()

# Step 4: Confusion matrix for Top1 prediction vs actual
# Special handling if real position contains multiple values
def canonicalize(pos_str):
    return pos_str.split(",")[0].strip() if "," in pos_str else pos_str.strip()

df["Pos_main"] = df["Pos"].apply(canonicalize)

labels = sorted(df["Pos_main"].unique())
cm = confusion_matrix(df["Pos_main"], df["Top1_Pred"], labels=labels)

plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix: Real vs Predicted Top1 Position")
plt.tight_layout()
plt.show()
