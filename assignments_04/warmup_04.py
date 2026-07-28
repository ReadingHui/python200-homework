import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    f1_score,
    classification_report,
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# --- ROC and AUC ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Q1
# Logistic Regression Model
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_probs_lr = lr.predict_proba(X_test)
lr_score = roc_auc_score(y_test, y_probs_lr[:, 1])
print(f'AUC score for Logistic Regression Model: {lr_score:.4f}')

# KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_probs_knn = knn.predict_proba(X_test_scaled)
knn_score = roc_auc_score(y_test, y_probs_knn[:, 1])
print(f'AUC score for kNN Model: {knn_score:.4f}')

# kNN has a higher AUC (0.9394 vs 0.7060). That tells me kNN model is better at separating the two classes, independently of any threshold choice.

# Q2
fig, ax = plt.subplots(figsize=(6, 5))
fpr, tpr, _ = roc_curve(y_test, y_probs_lr[:, 1])
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, name="Logistic Regression")
fpr, tpr, _ = roc_curve(y_test, y_probs_knn[:, 1])
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, name="k-Nearest Neighbor")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
ax.set_title("ROC Curve — Classifiers")
ax.legend()
plt.tight_layout()
plt.savefig("assignments_04/outputs/roc_comparison.png")
plt.close()

# At the point on each curve where TPR = 0.80, kNN has a much lower FPR. 
# That means kNN is better at separating the two classes, if I need to catch 80% of the positives,
# kNN would produce fewer false alarms.

# Q3
fpr, tpr, thresholds = roc_curve(y_test, y_probs_lr[:, 1])
best_score = 0
best_fpr, best_tpr, best_threshold = fpr[0], tpr[0], thresholds[0]
for fpr, tpr, threshold in zip(fpr, tpr, thresholds):
    y_pred = (y_probs_lr[:, 1] > threshold).astype(int)
    score = f1_score(y_test, y_pred)
    if score > best_score:
        best_score = score
        best_fpr = fpr
        best_tpr = tpr
        best_threshold = threshold
print(f'{'Optimal threshold: ':<20}{best_threshold:>5.4f}')
print(f'{'Optimal TPR: ':<20}{best_tpr:>5.4f}')
print(f'{'Optimal FPR: ':<20}{best_fpr:>5.4f}')
print(f'{'Optimal f1: ':<20}{best_score:>5.4f}')

# This optimal threshold is much lower than the default 0.5.
# In real application, we may choose a threshold lower than 0.5 if 0.5 threshold
# generated too many False-Negative cases to a point that affects the accuracy a lot.

# --- GridSearchCV ---
#Q1
param_grid = {
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

pipeline_lr = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]
)

grid_search_lr = GridSearchCV(
    estimator=pipeline_lr,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search_lr.fit(X_train, y_train)

print('====== Logistic Regression ======')
print(f"Best C:                     {grid_search_lr.best_params_['clf__C']}")
print(f"Best CV AUC:                {grid_search_lr.best_score_:.3f}")
print(f"Test AUC of best model:     {roc_auc_score(y_test, grid_search_lr.best_estimator_.predict_proba(X_test)[:, 1]):.4f}")
print(f"Test AUC of the base model: {lr_score:.4f}")

# The grid search picked C=100.0 as the best C, which is the highest we can get, same as what I would expect.
# As a higher C means a lower regularization penalty, which means the model fit the training data better,
# unless it overfits, the CV score is going to be better.  
# Comparing to the original test AUC score, it has decreased by 0.0003.

# Q2
param_grid = {
    "clf__max_depth":  [2, 3, 5, 8, None]
}

pipeline_dt = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier(random_state=42))
    ]
)

grid_search_dt = GridSearchCV(
    estimator=pipeline_dt,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search_dt.fit(X_train, y_train)

print('====== Decision Tree ======')
print(f"Best max_depth:             {grid_search_dt.best_params_['clf__max_depth']}")
print(f"Best CV AUC:                {grid_search_dt.best_score_:.3f}")
print(f"Test AUC of best model:     {roc_auc_score(y_test, grid_search_dt.best_estimator_.predict_proba(X_test)[:, 1]):.4f}")

# Comparing the Logistic Regression (0.7060) and Decision Tree (0.9357), Decision Tree is way higher in the test AUC. That implies the Decision Tree is much
# better in differentiating the true and false label, hence I would bring that into further development. However, AUC is not the only thing I consider,
# More importantly, the CV AUC and test AUC difference in Decision Tree is much smaller than that of the Logistic Regression, which means the decision tree is
# not overfitted while the Logistic Regression is. This shows the Decision Tree is better as well. In practice, I will also consider the application and see whether
# precision and recall are better deciding factor as well. However, this is just a toy dataset with no actual meaning, so I will just decide by AUC and train-test difference.

# Q3

max_depths = [2, 3, 5, 8, None]
mean_scores = grid_search_dt.cv_results_['mean_test_score'].round(4)
std_test_scores = grid_search_dt.cv_results_['std_test_score'].round(4)
cv_results = pd.DataFrame({
    'Mean Score': mean_scores,
    'Std Score': std_test_scores
}, index=max_depths)
print(cv_results.sort_values(by='Mean Score', ascending=False))

# For max_depth=3.0 or 5.0, the Mean Score are similar (0.9165 vs 0.9024), but standard deviations are quite different (0.0213 bs 0.0191).
# If I have to choose, I will choose the one with lower standard deviation (0.0191), which means it is more stable across different folds,
# and hence should be more stable in production environment.

# --- joblib ---
# Q1
best_lr_pipe = grid_search_lr.best_estimator_
joblib.dump(best_lr_pipe, 'assignments_04/models/warmup_mode.pkl')

loaded_clf = joblib.load('assignments_04/models/warmup_mode.pkl')

original_preds = best_lr_pipe.predict(X_test)
loaded_preds = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")

# Q2
new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])
# --- Simulated prediction script ---
def prediction(samples):
    loaded_clf = joblib.load('assignments_04/models/warmup_mode.pkl')
    pred = loaded_clf.predict(samples)
    proba = loaded_clf.predict_proba(samples)[:, 1].round(4)
    return pd.DataFrame({
        'Predicted Class': pred,
        'Probability': proba
    })
print('Predicted class and probability:')
print(prediction(new_samples))

# As the logistic regression formula is given by P(z) = 1 / (1 + e^(-z)), with all entries to be zero, the z reduces to the intercept b.
# That makes the prob = 1 / (1 + e^(-b)). Now since the dataset is generated using make_dataset, the distribution of classes should be 
# balanced, which makes b close to zero. Hence, the probability will be very close to 0.5, the only differentiating point is whether b
# is positive or negative, which is not possible to predict unless we check the coeffient of the model directly.
print(f'Constant in the Logistic Regression: {loaded_clf["clf"].coef_[0][-1]:.4f}')
# Since the constant term is positive, the probability will be >0.5, hence it should be class 1, which matches.
