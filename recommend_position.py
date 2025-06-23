import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load and clean data
df = pd.read_csv("data/archive_3/2023-2024/combined_players_stats.csv")

# Clean player names
df['Player'] = df['Player'].str.strip()
df['Pos'] = df['Pos'].astype(str).str.strip()

# Features
features = [
    'Age', 'Playing Time 90s',
    'Performance Gls', 'Performance Ast',
    'Expected xG', 'Expected xAG',
    'Progression PrgC', 'Progression PrgP',
    'Per 90 Minutes Gls', 'Per 90 Minutes Ast',
    'Per 90 Minutes xG', 'Per 90 Minutes xAG'
]

# Drop rows with missing values
df = df.dropna(subset=features + ['Pos', 'Player'])

# Prepare single-label encoding for model training
df['Main_Pos'] = df['Pos'].apply(lambda x: x.split(',')[0].strip())
le = LabelEncoder()
df['Pos_encoded'] = le.fit_transform(df['Main_Pos'])

# Train-test split (no leakage)
X = df[features]
y = df['Pos_encoded']
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df[['Player', 'Pos']], test_size=0.3, random_state=42, stratify=y
)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
probs = clf.predict_proba(X_test)
top3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
top1 = le.inverse_transform(top3_indices[:, 0])
top2 = le.inverse_transform(top3_indices[:, 1])
top3 = le.inverse_transform(top3_indices[:, 2])

# Combine results
results = df_test.copy()
results.reset_index(drop=True, inplace=True)
results['Top1_Pred'] = top1
results['Top2_Pred'] = top2
results['Top3_Pred'] = top3

# ✅ Updated assessment logic to handle multiple true positions
def assess(row):
    true_positions = [p.strip() for p in row['Pos'].split(',')]
    if row['Top1_Pred'] in true_positions:
        return "Best Positioned"
    elif row['Top2_Pred'] in true_positions or row['Top3_Pred'] in true_positions:
        return "Good Fit"
    else:
        return "Consider Review"

results['Assessment'] = results.apply(assess, axis=1)

# Save results
results.to_csv("position_recommendations.csv", index=False)

# Metrics summary
summary = results['Assessment'].value_counts()
print("\n📊 Position Fit Summary:")
for label, count in summary.items():
    print(f"{label}: {count} players")

# Evaluation on Top1 accuracy
print("\n🔍 Classification Report (Top1 only):")
y_pred_top1 = le.transform(results['Top1_Pred'])
y_true = le.transform([x.split(',')[0].strip() for x in results['Pos']])
print(classification_report(y_true, y_pred_top1, target_names=le.classes_))
