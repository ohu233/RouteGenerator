import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load artificial trajectory features
df = pd.read_csv('data/realWorldMixedFeatures.csv')

# Split by trajectory ID to avoid data leakage (same trajectory in both train and test)
traj_ids = df['ID'].unique()
train_ids, test_ids = train_test_split(traj_ids, test_size=0.3, random_state=42)

df_train = df[df['ID'].isin(train_ids)]
df_test = df[df['ID'].isin(test_ids)]

X_train = df_train[['speed','TG','GG','GSD','TS']]
Y_train = df_train['mode']

# Train model
clf = RandomForestClassifier(max_depth=9, n_estimators=200, max_features=10, random_state=42)
clf.fit(X_train, Y_train)

# Test model
X_test = df_test[['speed','TG','GG','GSD','TS']]
Y_test = df_test['mode']
Y_pred = clf.predict(X_test)


def majority_vote(predictions, ids):
    vote_results = {}
    for idx, pred in zip(ids, predictions):
        if idx not in vote_results:
            vote_results[idx] = []
        vote_results[idx].append(pred)

    final_predictions = []
    for idx in ids:
        final_predictions.append(Counter(vote_results[idx]).most_common(1)[0][0])

    return final_predictions

Y_pred_voted = majority_vote(Y_pred, df_test['ID'])

# Point-level metrics
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, average='weighted')
recall = recall_score(Y_test, Y_pred, average='weighted')
f1 = f1_score(Y_test, Y_pred, average='weighted')

print('=== Point-level ===')
print(f'Accuracy:  {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall:    {recall:.4f}')
print(f'F1:        {f1:.4f}')

# Trajectory-level metrics (after majority vote)
accuracy_voted = accuracy_score(Y_test, Y_pred_voted)
precision_voted = precision_score(Y_test, Y_pred_voted, average='weighted')
recall_voted = recall_score(Y_test, Y_pred_voted, average='weighted')
f1_voted = f1_score(Y_test, Y_pred_voted, average='weighted')

print('\n=== Trajectory-level (majority vote) ===')
print(f'Accuracy:  {accuracy_voted:.4f}')
print(f'Precision: {precision_voted:.4f}')
print(f'Recall:    {recall_voted:.4f}')
print(f'F1:        {f1_voted:.4f}')