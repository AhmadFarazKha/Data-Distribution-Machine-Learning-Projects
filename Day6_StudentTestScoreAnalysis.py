import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns
import statsmodels.formula.api as smf

# Sample Data (Replace with your actual data)
np.random.seed(42)
n_students = 200
subjects = ['Math', 'Science', 'English', 'History']
test_dates = pd.to_datetime(['2023-01-15', '2023-04-15', '2023-07-15', '2023-10-15'])

df = pd.DataFrame({
    'Student ID': np.repeat(range(1, n_students + 1), len(subjects)*len(test_dates)),
    'Subject': np.tile(np.repeat(subjects, len(test_dates)), n_students),
    'Test Date': np.tile(test_dates, n_students*len(subjects)),
    'Score': np.random.normal(75, 10, n_students * len(subjects)*len(test_dates))
})
df['Test Date'] = pd.to_datetime(df['Test Date'])
df['Test Month'] = df['Test Date'].dt.to_period('M')

# --- Normality Testing ---
print("\n--- Normality Testing ---")
for subject in df['Subject'].unique():
    subject_scores = df[df['Subject'] == subject]['Score']
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(subject_scores, bins=10, alpha=0.6, label=subject)
    plt.title(f"Score Distribution in {subject}")
    plt.subplot(1, 2, 2)
    sm.qqplot(subject_scores, line='s')
    plt.title(f"Q-Q Plot of Scores in {subject}")
    plt.tight_layout()
    plt.show()
    shapiro_test = stats.shapiro(subject_scores)
    print(f"Shapiro-Wilk Test for {subject}: {shapiro_test}")
    k2, p = stats.normaltest(subject_scores)
    print(f"D'Agostino's K-squared test for {subject}: statistics=%.3f, p=%.3f" % (k2, p))

# --- Comparing Distributions Across Subjects ---
print("\n--- Comparing Distributions Across Subjects ---")
plt.figure(figsize=(8, 6))
sns.boxplot(x='Subject', y='Score', data=df)
plt.title("Score Distributions Across Subjects")
plt.show()

plt.figure(figsize=(8, 6))
sns.violinplot(x='Subject', y='Score', data=df)
plt.title("Score Distributions Across Subjects (Violin Plot)")
plt.show()

model = smf.ols('Score ~ C(Subject)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\nANOVA Test:")
print(anova_table)

# --- Identifying Correlations Between Subjects ---
print("\n--- Identifying Correlations Between Subjects ---")

df_pivot = df.pivot(index='Student ID', columns='Subject', values='Score').reset_index()
correlation_matrix = df_pivot.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Matrix")
plt.show()

# Example Scatter Plot
if 'Math' in df_pivot.columns and 'Science' in df_pivot.columns: #Check if the columns exist before plotting to avoid errors
    plt.figure(figsize=(6, 6))
    plt.scatter(df_pivot['Math'], df_pivot['Science'])
    plt.xlabel("Math Score")
    plt.ylabel("Science Score")
    plt.title("Scatter Plot: Math vs. Science")
    plt.show()
else:
    print("Math or Science scores not found for correlation scatter plot.")

# --- Tracking Score Trends Over Time ---
print("\n--- Tracking Score Trends Over Time ---")
trends = df.groupby(['Test Month', 'Subject'])['Score'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(x='Test Month', y='Score', hue='Subject', data=trends, marker='o')
plt.title("Score Trends Over Time")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()