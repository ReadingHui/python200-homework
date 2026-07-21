import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


import os
from sklearn.model_selection import train_test_split

# --- scikit-learn API ---
# Q1


years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)
print(f'Salary prediction for 4 years experience: {model.predict(np.array(4).reshape(-1,1))}')
print(f'Salary prediction for 8 years experience: {model.predict(np.array(8).reshape(-1,1))}')
print(f'Slope of model: {model.coef_[0]}')
print(f'Intercept of the model: {model.intercept_}')

# Q2
X = np.array([10, 20, 30, 40, 50])
print(f'Shape of X: {X.shape}')
X = X.reshape(-1, 1)
print(f'New shape of X: {X.shape}')

# Scikit-learn require X to be 2D for 2 main reasons, one is to avoid ambiguity.
# Is a 1D array n features with 1 entry or 1 feature with n entries?
# Also, it makes the API more robust, the code can always assume the correct shape of the dataset instead of guessing and changing.
# At the same time, X being 2D also aligns with how scikit-learn handles regression, it is always trying to do matrix multiplication,
# where the features are represented by X, the feature Matrix, to multiply with other vectors.

# Q3


X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_clusters)
print(f'Cluster centers: {kmeans.cluster_centers_}')
print(f'Cluster point counts [first, second, third]: {np.bincount(labels)}')
df = pd.DataFrame(X_clusters, columns=['x', 'y'])
df['labels'] = labels

sns.scatterplot(df, x='x', y='y', hue='labels', palette='Set1')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='black', marker='x', label='Cluster centers')
plt.xlabel('x')
plt.ylabel('y')
plt.title('K-Means Cluster Example with 120 random data')
plt.legend()
plt.savefig('assignments_02/outputs/kmeans_clusters.png')
plt.close()

# --- Linear Regression ---

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Q1
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel('Age')
plt.ylabel('Cost')
plt.savefig('assignments_02/outputs/cost_vs_age.png')
plt.close()

# The smoker and non-smoker group are clearly visible, which suggests 'smoker' increases the medical cost.

# Q2
def split_train(X, y, feature_cnt=1, smoker=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)
    X_train = X_train.reshape(-1, feature_cnt)
    X_test = X_test.reshape(-1, feature_cnt)

    print(f'Shape of X_train: {X_train.shape}')
    print(f'Shape of X_test: {X_test.shape}')
    print(f'Shape of y_train: {y_train.shape}')
    print(f'Shape of y_test: {y_test.shape}')

    model = LinearRegression()
    model.fit(X_train, y_train)
    if not smoker:
        print(f'Slope of model: {model.coef_[0]}')
    else:
        print("Age coefficient:    ", model.coef_[0])
        print("Smoker coefficient: ", model.coef_[1])
    print(f'Intercept of model: {model.intercept_}')
    y_pred = model.predict(X_test)
    print(f'RMSE of prediction: {np.sqrt(np.mean((y_pred - y_test) ** 2))}')
    print(f'R^2 on test set: {model.score(X_test, y_test)}')

    return y_pred, y_test
print('========== Q3 ============')
age_pred, y_test = split_train(age, cost)
print()

# The slope means the increase in medical costs per every year passed.

# Q4
X_full = np.column_stack([age, smoker])
print('========== Q4 ============')
full_pred, _ = split_train(X_full, cost, 2, True)
print()

# Adding the smaker flag helps a lot, halving the RMSE, bringing it back down to around the noise added in the original data generation.
# The `smoker` coefficient represent that if you smoke, your medical cost is going to be $15,000 higher than non-smoker on average.

# Q5
y_pred = np.hstack((age_pred, full_pred))
model_smoke = np.hstack((np.zeros(len(age_pred)), np.ones(len(full_pred))))
y_test = np.hstack((y_test, y_test))
plt.scatter(y_pred, y_test, c=model_smoke, cmap="coolwarm")
plt.title('Predicted vs Actual')
plt.xlabel('Predicted cost')
plt.ylabel('Actual cost')
plt.axline((0,0), slope=1, linestyle=':', color='black')
plt.savefig('assignments_02/outputs/predicted_vs_actual.png')
plt.close()

# The point above diagonal means the model underestimated, below the diagronal means the model overestimated.