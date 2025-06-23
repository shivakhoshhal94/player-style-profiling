import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("data/archive_3/2023-2024/combined_players_stats.csv")

# Feature columns for prediction
features = [
    'Age', 'Playing Time 90s',
    'Performance Gls', 'Performance Ast',
    'Expected xG', 'Expected xAG',
    'Progression PrgC', 'Progression PrgP',
    'Per 90 Minutes Gls', 'Per 90 Minutes Ast',
    'Per 90 Minutes xG', 'Per 90 Minutes xAG'
]

# Filter out rows with missing data in features and labels
df = df.dropna(subset=features + ['Pos', 'Player'])

# Encode target labels (positions)
le = LabelEncoder()
df['Pos_encoded'] = le.fit_transform(df['Pos'])

# Split data to prevent leakage
X = df[features]
y = df['Pos_encoded']
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df[['Player', 'Pos']], test_size=0.3, random_state=42, stratify=y
)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict probabilities and get top 3
probs = clf.predict_proba(X_test)
top3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1]

# Decode each top prediction separately
top1 = le.inverse_transform(top3_indices[:, 0])
top2 = le.inverse_transform(top3_indices[:, 1])
top3 = le.inverse_transform(top3_indices[:, 2])

# Assemble results
results = df_test.copy()
results.reset_index(drop=True, inplace=True)
results['Top1_Pred'] = top1
results['Top2_Pred'] = top2
results['Top3_Pred'] = top3

# Define assessment rule
def assess(row):
    if row['Top1_Pred'] == row['Pos']:
        return "Best Positioned"
    elif row['Pos'] in [row['Top2_Pred'], row['Top3_Pred']]:
        return "Good Fit"
    else:
        return "Consider Review"

results['Assessment'] = results.apply(assess, axis=1)

# Save results to CSV
results.to_csv("position_recommendations.csv", index=False)

# Summary
summary = results['Assessment'].value_counts()
print("\n📊 Position Fit Summary:")
for label, count in summary.items():
    print(f"{label}: {count} players")

# Print classification report
print("\n🔍 Classification Report (Top1 only):")
y_pred_top1 = le.transform(results['Top1_Pred'])
y_true = le.transform(results['Pos'])
print(classification_report(y_true, y_pred_top1, target_names=le.classes_))
