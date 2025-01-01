import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm

# Sample Data for "Inlights Company" (Replace with your actual data)
np.random.seed(42)  # for reproducibility
n_employees = 150
departments = ['Marketing', 'Sales', 'Engineering', 'HR', 'Finance']
job_levels = ['Junior', 'Mid-Level', 'Senior', 'Manager']

df = pd.DataFrame({
    'Employee ID': range(1, n_employees + 1),
    'Department': np.random.choice(departments, n_employees),
    'Years of Experience': np.random.randint(0, 20, n_employees),
    'Job Level': np.random.choice(job_levels, n_employees),
    'Salary': 0  # Initialize salaries
})

# Generate salaries based on experience and job level with some random variation
for index, row in df.iterrows():
    base_salary = 40000  # Base salary
    experience_bonus = row['Years of Experience'] * 2000
    level_bonus = {'Junior': 0, 'Mid-Level': 15000, 'Senior': 30000, 'Manager': 50000}[row['Job Level']]
    df.loc[index, 'Salary'] = base_salary + experience_bonus + level_bonus + np.random.normal(0, 10000)

# --- Salary Distribution Across Departments ---
print("\n--- Salary Distribution Across Departments ---")

for dept in df['Department'].unique():
    dept_salaries = df[df['Department'] == dept]['Salary']

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(dept_salaries, bins=10, alpha=0.6, label=dept)
    plt.title(f"Salary Distribution in {dept}")

    plt.subplot(1, 2, 2)
    stats.probplot(dept_salaries, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of Salary in {dept}")
    plt.tight_layout()
    plt.show()

    shapiro_test = stats.shapiro(dept_salaries)
    print(f"Shapiro-Wilk Test for {dept}: {shapiro_test}")

    print(f"Descriptive Statistics for {dept}:\n{dept_salaries.describe()}\n")

plt.figure(figsize=(8,6))
sns.boxplot(x='Department', y='Salary', data=df)
plt.title("Salary Distribution across all Departments")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- Experience vs. Salary ---
print("\n--- Experience vs. Salary ---")

plt.figure(figsize=(8, 6))
plt.scatter(df['Years of Experience'], df['Salary'])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs. Salary")
plt.show()

correlation = df['Years of Experience'].corr(df['Salary'])
print(f"Correlation between Experience and Salary: {correlation}")

df['Experience_Group'] = pd.cut(df['Years of Experience'], bins=[0, 5, 10, float('inf')], labels=['0-5', '6-10', '10+'])

plt.figure(figsize=(8,6))
sns.boxplot(x='Experience_Group', y='Salary', data=df)
plt.title("Salary Distribution across Experience Groups")
plt.show()

# --- Salary Bands ---
print("\n--- Salary Bands ---")
df['Salary_Band'] = pd.qcut(df['Salary'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

plt.figure(figsize=(8,6))
sns.countplot(x='Salary_Band', data=df, order=['Q1', 'Q2', 'Q3', 'Q4'])
plt.title("Number of Employees in Each Salary Band")
plt.show()

plt.figure(figsize=(8,6))
sns.violinplot(x='Salary_Band', y='Salary', data=df, order=['Q1', 'Q2', 'Q3', 'Q4'])
plt.title("Salary Distribution within Salary Bands")
plt.show()

# --- Regression Analysis (for pay inequities) ---
print("\n--- Regression Analysis ---")
df = pd.get_dummies(df, columns=['Department', 'Job Level'])
X = df.drop(['Employee ID', 'Salary', 'Experience_Group', 'Salary_Band'], axis=1)
y = df['Salary']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())