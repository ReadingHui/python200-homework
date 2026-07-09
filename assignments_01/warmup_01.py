def print_title(title: str, level=0):
    title = ' ' * level * 5 + title + ' ' * level * 5
    print('=' * len(title))
    print(title)
    print('=' * len(title))

# --- Pandas ---
import pandas as pd

print_title('Pandas', 2)

# Pandas Q1
print_title('Pandas Q1', 1)
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print_title('First Three Rows')
print(df.head(3)) # First three rows
print()

print_title('Shape')
print(df.shape)
print()

print_title('Data Types')
print(df.dtypes)
print()

# Pandas Q2
print_title('Pandas Q2', 1)
print_title('Students passed AND grade above 80')
print(df.loc[(df['passed']) & (df['grade'] > 80)])
print()

# Pandas Q3
print_title('Pandas Q3', 1)
df['grade_curved'] = df['grade'] + 5
print_title('New column "grade_curved"')
print(df)
print()

# Pandas Q4
print_title('Pandas Q4', 1)
df['name_upper'] = df['name'].str.upper()
print_title('Added "name_upper"')
print(df[['name', 'name_upper']])
print()

# Pandas Q5
print_title('Pandas Q5', 1)
print_title('Mean grouped by city')
print(df.groupby('city')['grade'].mean())
print()

# Pandas Q6
print_title('Pandas Q6', 1)
df['city'] = df['city'].replace('Austin', 'Houston')
print_title('Replace Austin by Houston')
print(df)
print()

# Pandas Q7
print_title('Pandas Q7', 1)
df = df.sort_values(by='grade', ascending=False)
print_title('Top 3 rows sorted by grade')
print(df)
print()

# --- NumPy ---
import numpy as np
print_title('NumPy', 2)

# NumPy Q1
print_title('NumPy Q1', 1)
arr = np.array([10, 20, 30, 40, 50])
print_title('Shape')
print(arr.shape)
print()

print_title('dtype')
print(arr.dtype)
print()

print_title('ndim')
print(arr.ndim)
print()

# NumPy Q2
print_title('NumPy Q2', 1)
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print_title('Shape')
print(arr.shape)
print()

print_title('Size')
print(arr.size)
print()

# NumPy Q3
print_title('NumPy Q3', 1)
print_title('Top-left 2x2 block')
print(arr[0:2, 0:2])
print()

# NumPy Q4
print_title('NumPy Q4', 1)
print_title('3x4 array of zeros')
print(np.zeros((3, 4)))
print_title('2x5 array of ones')
print(np.ones((2, 5)))
print()

# NumPy Q5
print_title('NumPy Q5', 1)
arr = np.arange(0, 50, 5)
print_title('Array from 0 to 50 with step 5')
print(arr)
print()
print_title('Shape of the array')
print(arr.shape)
print()
print_title('Mean of the array')
print(np.mean(arr))
print()
print_title('Sum of the array')
print(np.sum(arr))
print()
print_title('SD of the array')
print(np.std(arr))
print()

# NumPy Q6
print_title('NumPy Q6', 1)
arr = np.random.normal(0, 1, 200)
print_title('Mean of the array')
print(np.mean(arr))
print()
print_title('SD of the array')
print(np.std(arr))
print()

# --- Matplotlib ---
import matplotlib.pyplot as plt
print_title('Matplotlib', 2)

# Matplotlib Q1
print_title('Matplotlib Q1', 1)
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Squares')
plt.show()

# Matplotlib Q2
print_title('Matplotlib Q2', 1)
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]
plt.bar(subjects, scores)
plt.xlabel('Subjects')
plt.ylabel('Scores')
plt.title('Subject Scores')
plt.show()

# Matplotlib Q3
print_title('Matplotlib Q3', 1)
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
plt.scatter(x1, y1, color='blue', label='Group 1')
plt.scatter(x2, y2, color='red', label='Group 2')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot')
plt.legend()
plt.show()

# Matplotlib Q4
print_title('Matplotlib Q4', 1)
fig, ax = plt.subplots(1, 2)
ax[0].plot(x, y)
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_title('Squares')
ax[1].bar(subjects, scores)
ax[1].set_xlabel('Subjects')
ax[1].set_ylabel('Scores')
ax[1].set_title('Subject Scores')
plt.tight_layout()
plt.show()

# --- Descriptive Stats ---
print_title('Descriptive Stats', 2)

# Descriptive Stats Q1
print_title('Descriptive Stats Q1', 1)
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print(f'Mean: {np.mean(data)}')
print(f'Median: {np.median(data)}')
print(f'Variance: {np.var(data)}')
print(f'Standard Deviation: {np.std(data)}')

# Descriptive Stats Q2
print_title('Descriptive Stats Q2', 1)
rand_val = np.random.normal(65, 10, 500)
plt.hist(rand_val, bins=20, edgecolor='black')
plt.title('Distribution of Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()

# Descriptive Stats Q3
print_title('Descriptive Stats Q3', 1)
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
fig = plt.boxplot((group_a, group_b), label=['Group A', 'Group B'], patch_artist=True)
colors = ['orange', 'lightblue']
for patch, color in zip(fig['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
plt.legend()
plt.show()

# Descriptive Stats Q4
print_title('Descriptive Stats Q4', 1)
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
fig = plt.boxplot((normal_data, skewed_data), label=['Normal', 'Skewed'], patch_artist=True)
colors = ['orange', 'lightblue']
for patch, color in zip(fig['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
plt.legend()
plt.show()
# The exponential data is more skewed, hence we should use median to describe the central tendency.
# In contrast, the normal distribution is more symmetric, hence a mean can be used.

# Descriptive Stats Q5
from scipy import stats
print_title('Descriptive Stats Q5', 1)
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]
print_title('Data 1')
print(f'Mean: {np.mean(data1)}')
print(f'Median: {np.median(data1)}')
print(f'Mode: {stats.mode(data1).mode}')

print_title('Data 2')
print(f'Mean: {np.mean(data2)}')
print(f'Median: {np.median(data2)}')
print(f'Mode: {stats.mode(data2).mode}')

# Since data2 is heavily right skewed, the mean is being "pulled" to the right. Mean is highly sensitive to outliers, while median is not.

# --- Hypothesis Testing ---
print_title('Hypothesis', 2)

# Hypothesis Q1
print_title('Hypothesis Q1', 1)
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]
res1 = stats.ttest_ind(group_a, group_b)
print(f'T-test result:')
print(f't-statistic: {res1.statistic}')
print(f'p-value: {res1.pvalue}')

# Hypothesis Q2
print_title('Hypothesis Q2', 1)
if res1.pvalue <= 0.05:
    print('p-value smaller than 0.05, result is not significant.')
else:
    print('p-value larger than 0.05, result is not significant.')

# Hypothesis Q3
print_title('Hypothesis Q3', 1)
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]
res3 = stats.ttest_rel(before, after)
print(f'T-test result:')
print(f't-statistic: {res3.statistic}')
print(f'p-value: {res3.pvalue}')

# Hypothesis Q4
print_title('Hypothesis Q4', 1)
scores = [72, 68, 75, 70, 69, 74, 71, 73]
res4 = stats.ttest_1samp(scores, 70)
print(f'T-test result:')
print(f't-statistic: {res4.statistic}')
print(f'p-value: {res4.pvalue}')

# Hypothesis Q5
print_title('Hypothesis Q5', 1)
res5 = stats.ttest_ind(group_a, group_b, alternative='greater')
print(f'T-test result:')
print(f't-statistic: {res5.statistic}')
print(f'p-value: {res5.pvalue}')

print('From the result of Hypothesis test Q2 and Q5, we see that it is highly likely the mean of group a is lower than that of group b is not by chance.')

# --- Correlation ---
print_title('Correlation', 2)

# Correlation Q1
print_title('Correlation Q1', 1)
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
print(f'Correlation matrix: {np.corrcoef(x, y)}')
print(f'Correlation Coefficient: {np.corrcoef(x, y)[0, 1]}')
# The correlation is expected to be 1 as y is just double that of x.

# Correlation Q2
print_title('Correlation Q2', 1)
from scipy.stats import pearsonr
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]
pnr = pearsonr(x, y)
print(f'Correlation Coefficient: {pnr.statistic}')
print(f'p-value: {pnr.pvalue}')

# Correlation Q3
print_title('Correlation Q3', 1)
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
df_corr = df.corr()
print(df_corr)

# Correlation Q4
print_title('Correlation Q4', 1)
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]
plt.scatter(x, y)
plt.title('Negative Correlation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Correlation Q5
print_title('Correlation Q5', 1)
import seaborn as sns

sns.heatmap(df_corr, annot=True)
plt.title('Correlation Heatmap')
plt.show()

# --- Pipelines ---
print_title('Pipeline', 2)

# Pipelines Q1
print_title('Pipelines Q1', 1)
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    return pd.Series(arr, name='values')

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    return {
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'mode': series.mode()[0]
    }

def data_pipeline(arr):
    X = create_series(arr)
    X = clean_data(X)
    summary = summarize_data(X)
    return summary

print(f'Summary dict: {data_pipeline(arr)}')