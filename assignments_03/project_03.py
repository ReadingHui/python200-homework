import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import seaborn as sns
from io import BytesIO

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Task 1: Load and Explore
COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
print('First 5 rows:')
print(df.head())
print()

print('Shape:')
print(df.shape)
print()

print('DataFrame Info:')
print(df.info())
print()

print('Number of different spam_labels:')
print(df['spam_label'].value_counts())
print()
# There are 4601 emails in the dataset.
# The class are slightly imbalanced in a 60-40 split.
# With imbalanced dataset, accuracy high doesn't directly translate into a good prediction,
# as blind guess could be >80% accurate if the classes are 80-20 split.

key_features = ['word_freq_free', 'char_freq_!', 'capital_run_length_total']
for feature in key_features:
    sns.boxplot(df, x=feature, hue='spam_label', showfliers=False) # Not showing outliers as it compresses the plots
    plt.title(f'Boxplot of {feature} distribution by spam_label')
    plt.xlabel(f'{feature}')
    plt.ylabel('Spam Label')
    plt.savefig(f'assignments_03/outputs/{feature}_spam.png')
    plt.close()
# For spam email, the frequency of `free` is much higher than that of non spam emails (almost 0)
# The frequency of `!` and Capital run length are also significantly higher for spam emails

print('Data description (Full):')
print(df.describe())
print()

print('Data description (Spam):')
print(df[df['spam_label'] == 1].describe())
print()

print('Data description (Ham):')
print(df[df['spam_label'] == 0].describe())
print()

# The heavy skew toward zero tells me that the word and char frequency are chosen carefully,
# those are the words that appears frequently in spam emails but not ham emails.
# The numeric scale for word_freq are normally under 20, while the capital run length can go up to thousands.
# This is cause by for a normally passage, word count should not exceed thousand, and among the words,
# it is almost impossible for the same word to be used repeated extensively, otherwise the passage will not be readable.
# However, for the capital count, it is counting characters instead of words, and full capitalized email are still readable.
# This may affect the models which are sensitive to numeric scale, like regressions and deep learning models, scaling may have to be done.

# Task 2: Prepare Your Data
# Train-Test Split
X = df.drop('spam_label', axis=1)
y = df['spam_label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.8,
    test_size=0.2,
    random_state=42,
    stratify=y
) # An 80-20 split with stratify to ensure both spam and ham emails are represented.

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

# Scaling by StandardScaler()
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
    )
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
    )

# PCA
pca = PCA()
pca.fit(X_train_scaled)
cev = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cev)
plt.axhline(0.9, color='black', linestyle=':')
n = np.argmax(cev > 0.9) + 1
plt.axvline(n + 1, color='black', linestyle=':')
plt.title('Cumulative explained variance against Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative explained variance')
plt.savefig('assignments_03/outputs/cev_pca.png')
plt.close()

print(f'Number of components where it first reaches 90%: {n}')
print()
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca  = pca.transform(X_test_scaled)[:, :n]

# Task 3: A Classifier Comparison
# Print the accuracy and the full classification report.
def model_report(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Classification report:')
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print()

# KNN
# Unscaled data
knn_unscaled = KNeighborsClassifier(n_neighbors=5) 
knn_unscaled.fit(X_train, y_train)
print('=== Unscaled model ===')
model_report(knn_unscaled, X_test, y_test)

# Scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
print('=== Scaled model ===')
model_report(knn_scaled, X_test_scaled, y_test)

# Scaled data with PCA
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
print('=== Scaled model with PCA ===')
model_report(knn_pca, X_test_pca, y_test)

# The result for scaled data models are basically the same, with or without PCA.

# Decision Tree
depths = [3, 5, 10, None]
train_accuracies = []
test_accuracies = []
for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    accuracy = dt.score(X_train, y_train)
    train_accuracies.append(accuracy)
    accuracy = dt.score(X_test, y_test)
    test_accuracies.append(accuracy)
accuracy_df = pd.DataFrame(
    {
        'Train': train_accuracies,
        'Test': test_accuracies
    },
    index=depths
)
print('Accuracy scores of different depths decision tree:')
print(accuracy_df)
print()

# The accuracy score for training set increases much faster than test set as the depth increases,
# the gap widens and shows sign of overfitting. The model performs very well on training set but poor on test set,
# which means it is memorizing the answers instead of recognizing the patterns.
# I would pick max_depth=3 because the gap between train and test is the smallest, hence not much overfitting.

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
print('=== Decision Tree with max_depth=3 ===')
model_report(dt, X_test, y_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print('=== Random Forest with n_estimators=100 ===')
model_report(rf, X_test, y_test)

# Top 10 most important features
dt_top_10 = pd.Series(X_train.columns[np.argsort(dt.feature_importances_)[-10:][::-1]])
print('Top 10 most important features for Decision Tree:')
print(dt_top_10)

rf_top_10 = pd.Series(X_train.columns[np.argsort(rf.feature_importances_)[-10:][::-1]])
print('Top 10 most important features for Random Forest:')
print(rf_top_10)
# The two models agreed on 6 out of the 10 most important features, which mostly aligned with what I expect, like the
# excessive use of capital letters and exclamation marks, as well as focusing on money.

# Bar chart for RF
plt.bar(X_train.columns, rf.feature_importances_)
plt.title('Feature Importance of Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.savefig('assignments_03/outputs/feature_importances.png')
plt.close()

# Logistic Regression
lr_scaled = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
lr_scaled.fit(X_train_scaled, y_train)
print('=== Logistic Regression on Scaled Data ===')
model_report(lr_scaled, X_test_scaled, y_test)

lr_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
lr_pca.fit(X_train_pca, y_train)
print('=== Logistic Regression on PCA Data ===')
model_report(lr_pca, X_test_pca, y_test)

# The accuracy, precision, recall and f1-score are all slightly higher for plain Logistic Regression on scaled data instead of the PCA
# which is expected. However, the difference is not too far off, so for performance sake, choosing PCA is a good option. 

# In summary, the best performing model is the Random Forest with n_estimators=100. It has the highest metric across accuracy, precision, recall and f1-score.
# The accuracy, precision, recall and f1-score are all slightly higher for scaled data instead of the PCA
# which is expected. However, the difference is not too far off, so for performance sake, choosing PCA is a good option. 

# Pure accuracy is not the best metric to optimize in a spam classification task.
# To me, I would like to minimize the false positives, as email is a daily conversation tool,
# as well as business essentials. Any false positive can be devastating to a user or business.
# Spam detection should just be a first line of defence, not completely filtering all possible emails.

y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm)
cmd.plot(cmap='coolwarm', text_kw={'color': 'black'})
plt.savefig('assignments_03/outputs/best_model_confusion_matrix.png')
plt.close()

# With respect to the confusion matrix, the most error the model made was the false negative, which is what I would like to be.

# Task 4: Cross-Validation
models = {
    'knn': KNeighborsClassifier(n_neighbors=5),
    'dt': DecisionTreeClassifier(max_depth=3, random_state=42),
    'rf': RandomForestClassifier(n_estimators=100, random_state=42),
    'lr': LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
}
for name, model in models.items():
    if name in ['knn', 'lr']:
        cv = cross_val_score(model, X_train_scaled, y_train, cv=5)
    else:
        cv = cross_val_score(model, X_train, y_train, cv=5)
    print(f'Mean of {name}: {cv.mean():.4f}; Std of {name}: {cv.std():.4f}')

# Most accurate model is RandomForest with 0.9541 mean accuracy at highest, 
# while KNN is the most stable, with std lowest at 0.0094. The ranking is the same with just basic train-test split.

# Task 5: Building a Prediction Pipeline

# Pipeline 1: Tree-based classifier
tree_pipeline = Pipeline(
    [
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ]
)
tree_pipeline.fit(X_train, y_train)
y_pred = tree_pipeline.predict(X_test)
print('Classification report of the best tree model:')
print(classification_report(y_test, y_pred))
print()

# Pipeline 2: Non-tree model
non_tree_pipeline = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
    ]
)
non_tree_pipeline.fit(X_train, y_train)
y_pred = non_tree_pipeline.predict(X_test)
print('Classification report of the best non-tree model:')
print(classification_report(y_test, y_pred))
print()

# The pipeline of non-tree has the additional scaler to the tree pipeline.
# That is because tree-based models are not sensitive to numeric range, 
# so it is not necessary to scale the input or perform PCA on it.
# Packaging a model this way provides a more clear structure on how the pipeline is built,
# as well as easier to maintain when we want to change a part of the flow.
# This increases stability when handing off to someone else or deploying it,
# as the structure and models are separated, we can change part of it easily.
