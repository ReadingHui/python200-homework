# When reading the csv, the separater needs to be specified as a semicolon.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error as rmse


# Task 1: Load and Explore
# Data import
df = pd.read_csv('assignments_02/student_performance_math.csv', sep=';')
print(f'Shape of data: {df.shape}\n')
print('First 5 rows of the data:')
print(df.head())
print()
print('Datatype of the columns:')
print(df.dtypes)
print()

# Histogram of G3
plt.hist(df['G3'], 21)
plt.title('Distribution of Final Math Grades')
plt.xlabel('G3')
plt.xlabel('Frequency')
plt.savefig('assignments_02/outputs/g3_distribution.png')
plt.close()

# Task 2
# Filtering the G3=0 rows
print(f'df shape before: {df.shape}')
df_G3_0 = df[df['G3'] == 0]
df_filtered = df[df['G3'] > 0]
print(f'df shape after: {df_filtered.shape}')
# Keeping the rows heavily skewed the data to an imaginery 0 point, which the students didn't actually get.

# Encoding the categorical data
df_filtered[['schoolsup', 'internet', 'higher', 'activities']] = df_filtered[['schoolsup', 'internet', 'higher', 'activities']].replace({'yes': 1, 'no': 0})
df_filtered['sex'] = df_filtered['sex'].replace({'M': 0, 'F': 1})
print(df_filtered.head())

# Correlation between `absences` and `G3`
print(f'Original corr between `absence` and `G3`: {df['absences'].corr(df['G3'])}')
print(f'New corr between `absence` and `G3`: {df_filtered['absences'].corr(df['G3'])}')

# fig, ax = plt.subplots(1, 2)
# sns.scatterplot(df, x='absences', y='G3', alpha=0.3, ax=ax[0])
# ax[0].set_title('Absences vs G3 pre-filter')
# ax[0].set_xlabel('Absences')
# ax[0].set_ylabel('G3')
# sns.scatterplot(df_filtered, x='absences', y='G3', alpha=0.3, ax=ax[1])
# ax[1].set_title('Absences vs G3 post-filter')
# ax[1].set_xlabel('Absences')
# ax[1].set_ylabel('G3')
# plt.tight_layout()
# plt.show()

# As G3=0 doesn't mean the students actually getting 0 marks, it highly favors the 0 absences student to get 0 mark in G3, which skewed the data a lot.

# Task 3
print(df_filtered.corr()['G3'].sort_values())
# Strongest relationship feature is failures at -0.294 corr except G1 and G2. 
# It is kind of surprising the freetime and activities of the students doesn't affect grades that much, as I would assume they have relationship on how much time students spend on studying.

sns.heatmap(df_filtered.corr(), 
            vmin=-1, 
            vmax=1, 
            annot=True, 
            annot_kws={'size': 6}, 
            fmt='.2f',
            cmap="coolwarm", 
            center=0)
plt.title('Heatmap of correlations')
plt.savefig('assignments_02/outputs/corr_heatmap.png')
plt.close()

_, ax = plt.subplots(2, 2, figsize=(16, 12))
sns.boxplot(df_filtered, x='failures', y='G3', hue='failures', ax=ax[0][0])
ax[0][0].set_title('Boxplot of failures against G3')
ax[0][0].set_xlabel('Failures')
ax[0][0].set_ylabel('G3')

sns.boxplot(df_filtered, x='schoolsup', y='G3', hue='schoolsup', ax=ax[0][1])
ax[0][1].set_title('Boxplot of schoolsup against G3')
ax[0][1].set_xlabel('School Support')
ax[0][1].set_ylabel('G3')

sns.scatterplot(df_filtered, x='absences', y='G3', alpha=0.3, ax=ax[1][0])
ax[1][0].set_title('Scatterplot of absences against G3')
ax[1][0].set_xlabel('Absences')
ax[1][0].set_ylabel('G3')

sns.boxplot(df_filtered, x='Medu', y='G3', hue='Medu', ax=ax[1][1])
ax[1][1].set_title('Boxplot of Mother Education against G3')
ax[1][1].set_xlabel('Mother Education')
ax[1][1].set_ylabel('G3')
plt.tight_layout()
plt.savefig('assignments_02/outputs/top_4_correlated_features_plot.png')
plt.close()

pivot_table = df_filtered.pivot_table(
    index='Fedu', 
    columns='Medu', 
    values='G3', 
    aggfunc=['mean', 'count'])

_, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.heatmap(
    pivot_table['mean'],
    annot=True,
    fmt='.2f',
    cmap='YlGnBu',
    square=True,
    cbar_kws={'label': 'Mean G3'},
    ax=ax[0]
)
ax[0].set_xlabel('Mother Education')
ax[0].set_ylabel('Father Education')
ax[0].set_title('Average G3 by Mother and Father Education')
ax[0].invert_yaxis()

sns.heatmap(
    pivot_table['count'],
    annot=True,
    cmap='YlGnBu',
    square=True,
    cbar_kws={'label': 'Count'},
    ax=ax[1]
)
ax[1].set_xlabel('Mother Education')
ax[1].set_ylabel('Father Education')
ax[1].set_title('Student count by Mother and Father Education')
ax[1].invert_yaxis()
plt.tight_layout()
plt.savefig('assignments_02/outputs/parents_education_vs_G3.png')
plt.close()

df_edu_diff = df_filtered.copy()
df_edu_diff['edudiff'] = abs(df_edu_diff['Medu'] - df_edu_diff['Fedu'])
sns.boxplot(df_edu_diff, x='edudiff', y='G3', hue='edudiff')
plt.title('Parent Education Level Difference vs G3')
plt.xlabel('Education Difference')
plt.ylabel('G3')
plt.savefig('assignments_02/outputs/parents_education_diff_vs_G3.png')
plt.close()

# The most related features are `failures`, `schoolsup`, `absences` and `Medu`. Those could worth a closer look.
# In the heatmap, I can see the most related features, as well as some interelated features like Mother and Father education.

# Hence, I plotted the relation graph between the most related features and G3,
# as well as the how Mother and Father Education affects G3.

# In the failures against G3 plot, we see the a negative relation between them
# In the schoolsup against G3 plot, we see the students with school support have a lower G3 than those without,
# which could potential means students who needed school support are the ones who are struggling.
# In the absences against G3 plot, we see the also a negative relation between them
# In the Mother/Father education to G3 heatmap, we see that the higher the education for both parents, the higher the average G3 will be.
# In the Education difference to G3 plot, we see that if the difference is about 2 levels, the average G3 is slightly higher.

# Task 4: Baseline Model
X_train, X_test, y_train, y_test = train_test_split(
    df_filtered['failures'].values.reshape(-1, 1), 
    df_filtered['G3'].values,
    train_size=0.8,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f'Slope of the model: {round(model.coef_[0], 4)}')
print(f'RMSE of the model: {round(rmse(y_test, y_pred), 4)}')
print(f'R^2 of the model: {round(model.score(X_test, y_test), 4)}')
print()

# The slope represents on average how much grades from G3 is reduced with an extra failures in the past.
# The RMSE represents the error of the prediction is on average 3 grades.
# The R^2 is about the same as what I expected from the EDA, as the correlation coefficient is -0.3, the R^2 should be about 0.09.

# Task 5: Build the Full Model

feature_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'absences', 'freetime', 'goout', 'Walc', 'schoolsup', 'internet', 'higher', 'activities', 'sex']
X = df_filtered[feature_cols].values
y = df_filtered["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train, y_train)
y_pred = model_full.predict(X_test)
print(f'Train R^2 of the model: {round(model_full.score(X_train, y_train), 4)}')
print(f'Test R^2 of the model: {round(model_full.score(X_test, y_test), 4)}')
print(f'Test RMSE of the model: {round(rmse(y_test, y_pred), 4)}')

# Adding the features helped improve the RMSE a little bit, but not significant.


for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Two things that are more suprising:
# With school support, the G3 is actually lower, which could indicate students who are using school support are the ones who are weaker.
# There is no significant relationship between freetime, activities and traveltime with G3, I would expect students who have more freetime means they can follow the school materials better.
# However, this shows there are no such relationship.

# Train R^2 is a bit higher than test R^2, which indicates there is an overfitting.
# In deployment, I would probably keep failures, school support, internet and will to higher education only. The other features are not as related as their coefficient is lower than 0.5.

# Task 6: Evaluate and Summarize
plt.scatter(y_pred, y_test)
plt.axline((0,0), slope=1, linestyle=':', color='black')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.title('Predicted vs Actual (Full Model)')
plt.savefig('assignments_02/outputs/predicted_vs_actual.png')
plt.close()

# The model seem to struggle roughly uniform across the grades.
# The point above diagonal means the model underestimated, below the diagronal means the model overestimated.

# The size of the filtered dataset: (357, 18) and the test set: 72

# The RMSE and R² of your best model in plain language -- on a 0-20 scale, what does a typical prediction error actually mean?
# The RMSE is 2.855, the R^2 is 0.1539
# The RMSE represents the error of the prediction is on average 3 grades.

# Which two features have the largest positive and largest negative coefficients, and what those mean
# Largest positive: internet, largest negative: schoolsup
# This means having internet is the best boost to the predicted grade
# and having schoolsup implies a largest negative boost to the predicted grade.
# One result that surprised you:
# With school support, the G3 is actually lower, which could indicate students who are using school support are the ones who are weaker.

# Neglected Feature: The Power of G1
feature_cols.append('G1')

X = df_filtered[feature_cols].values
y = df_filtered["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train, y_train)
y_pred = model_full.predict(X_test)
print(f'Train R^2 of the model: {round(model_full.score(X_train, y_train), 4)}')
print(f'Test R^2 of the model: {round(model_full.score(X_test, y_test), 4)}')
print(f'Test RMSE of the model: {round(rmse(y_test, y_pred), 4)}')

# A higher R^2 means G1 and G3 are highly correlated, but not causal relation. This is a useful model to identify students
# who might struggle, as G1 and G3 are highly related. However, if the educator wants to intervene early, before G1 is available,
# they have to look into the other features, like the failures before and schoolsup they receive.