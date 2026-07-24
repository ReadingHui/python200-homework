import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing ---
# Q1
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.8,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of y_test: {y_test.shape}')
print()

# Q2
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'Mean of each column: {X_train_scaled.mean(axis=0)}')
print()
# Fitting the scaler only on X_train prevents data leakage, as the distribution of the testing set should not be known by the model.

# --- KNN ---
# Q1
knn_q1 = KNeighborsClassifier(n_neighbors=5)
knn_q1.fit(X_train, y_train)
y_pred = knn_q1.predict(X_test)
print(f'Accuracy score of KNN: {knn_q1.score(X_test, y_test)}')
print(f'Classification report:')
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print()

# Q2
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
print(f'Accuracy score of KNN: {knn.score(X_test_scaled, y_test)}')
print(f'Classification report:')
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print()
# Scaling actually hurts the performance for this particular dataset, it could be due to Iris being a toy dataset.
# Unscaled Euclidean distance gives more weight to features with larger numeric ranges, scaling brings overlapping features into equal consideration.
# For this particular split, the larger value features happened to be more clean cut, which dominated the classification.
# So scaling actually misclassify a data or two.

# Q3
knn = KNeighborsClassifier(n_neighbors=5)
cv = cross_val_score(knn, X_train, y_train, cv=5)
print(f'Score of each fold: {cv}')
print(f'Mean of scores: {cv.mean():.4f}')
print(f'Standard Deviation of scores: {cv.std():.4f}')
print()
# This result is more trustworthy than a single train/test split, as it is not biased to only a single split, which could just be lucky in the classification.

# Q4
for k in range(1, 16, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    cv = cross_val_score(knn, X_train, y_train, cv=5)
    print(f'5-fold mean CV score for k={k} KNN model: {cv.mean():.4f}')
print()
# We can either choose k=7 or k=11 judging by the CV score, but k=7 should be chosen because of the simpler architecture

# --- Classifier Evaluation ---
# Q1
y_pred = knn_q1.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
cmd.plot(cmap='coolwarm', text_kw={'color': 'black'})
plt.savefig('assignments_03/outputs/knn_confusion_matrix.png')
plt.close()
# As the classification accuracy is 1.0, there is no species that got confused.

# --- The sklearn API: Decision Trees ---
# Q1
dtc = DecisionTreeClassifier(max_depth=3, random_state=42)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print(f'Accuracy score of Decision Tree: {dtc.score(X_test, y_test):.4f}')
print(f'Classification report:')
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print()
# The accuracy of the decision tree is slightly lower than that of the KNN.
# Scaling data should not affect the accuracy of the decision tree as it does not depend on Euclidean distance.

# --- Logistic Regression and Regularization ---
for C in [0.01, 1.0, 100]:
    log_reg = OneVsRestClassifier(
        LogisticRegression(
            C=C,
            max_iter=1000,
            solver="liblinear",
        )
    )
    log_reg.fit(X_train_scaled, y_train)
    coef_sum = np.abs(np.vstack([est.coef_ for est in log_reg.estimators_])).sum()
    print(f"For C={C}, total coefficient magnitude = {coef_sum:.4f}")
print()
# As the value of C increases, the coefficient of the model increases.
# This shows regularization forces the coefficient to be small, which increases the stability of the model and reduces overfitting.

# --- PCA ---
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

# Q1
print(f'Shape of X_digits: {X_digits.shape}')
print(f'Shape of images: {images.shape}')

indices = [
    np.where(y_digits == i)[0][0]
    for i in range(10)
]
example_images = images[indices]

_, axes = plt.subplots(1, 10)
for i in range(10):
    axes[i].imshow(example_images[i], cmap='gray_r')
    axes[i].set_title(f'Digit {i}')
    axes[i].axis('off')
plt.tight_layout()
plt.savefig('assignments_03/outputs/sample_digits.png')
plt.close()

# Q2
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)

scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)  # c = color array
plt.colorbar(scatter, label='Digit')
plt.savefig('assignments_03/outputs/pca_2d_projection.png')
plt.close()
# Same digit images tend to cluster together in this 2D space.

# Q3
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(0.8, color='black', linestyle=':')
plt.title('Cumulative explained variance vs number of components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.savefig('assignments_03/outputs/pca_variance_explained.png')
plt.close()
# According to the plot, approximately 12 components are required to explain 80% of the variance.

# Q4
def reconstruct_digit(sample_idx, scores, pca, n_components):
    # Reconstruct one digit using the first n_components principal components.
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)


fig, axes = plt.subplots(5, 5)

# Original digits
for idx in range(5):
    axes[0, idx].imshow(images[idx], cmap='gray_r')
    axes[0][idx].axis('off')
    axes[0][idx].set_title(f'Digit {idx}')
axes[0][0].text(-0.25, 0.5, 'Original', transform=axes[0][0].transAxes,
                        va='center', ha='right')

# Row 1 to 4
n_components = [2, 5, 15, 40]
for row, n in enumerate(n_components):
    for idx in range(5):
        reconstructed_im = reconstruct_digit(idx, scores, pca, n)
        axes[row + 1][idx].imshow(reconstructed_im, cmap='gray_r')
        axes[row + 1][idx].axis('off')
        if idx == 0:
            axes[row + 1][idx].text(-0.25, 0.5, f'{n}', transform=axes[row + 1][idx].transAxes,
                        va='center', ha='right')
fig.supylabel('Number of Components')
fig.suptitle('Digit Index')
plt.tight_layout()
plt.savefig('assignments_03/outputs/pca_reconstructions.png')
plt.close()
# At n=15, the digits start to sharpen and become clearly recognizable, which is approximately where the variance curve levels off (n ~ 12)