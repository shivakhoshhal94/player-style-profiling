import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Ensure figures folder exists
os.makedirs("figures", exist_ok=True)

# File path
file_path = "position_recommendations.csv"
if not os.path.exists(file_path):
    print("❌ File 'position_recommendations.csv' not found. Please run the prediction script first.")
    exit()

# Load data
df = pd.read_csv(file_path)

# -------------------------------
# 📊 Pie Chart: Assessment Summary
# -------------------------------
assessment_counts = df["Assessment"].value_counts()
print("📊 Assessment Summary:")
print(assessment_counts)

plt.figure(figsize=(6, 6))
assessment_counts.plot.pie(autopct='%1.1f%%', startangle=140)
plt.title("Assessment Summary of Player Position Fit")
plt.ylabel("")
plt.tight_layout()
plt.savefig("figures/assessment_pie_chart.png")
plt.show()

# ---------------------------------------
# 🔢 Confusion Matrix: Top1 vs Real Pos
# ---------------------------------------
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
plt.savefig("figures/confusion_matrix_top1.png")
plt.show()

# -----------------------------------------------
# 📶 Bar Chart: Frequency of Predicted Positions
# -----------------------------------------------
top_preds = pd.concat([df["Top1_Pred"], df["Top2_Pred"], df["Top3_Pred"]])
top_counts = top_preds.value_counts().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_counts.index, y=top_counts.values, palette="mako")
plt.title("Top-3 Predicted Positions Frequency")
plt.xlabel("Position")
plt.ylabel("Total Mentions in Top-3")
plt.tight_layout()
plt.savefig("figures/top3_predictions_barplot.png")
plt.show()

print("✅ All visualizations saved in 'figures/' folder.")
