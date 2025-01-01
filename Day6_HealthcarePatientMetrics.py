import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample Data (CMH Rawalpindi Example - Patient Recovery)
np.random.seed(42)
n_patients = 400
conditions = ['Appendicitis', 'Pneumonia', 'Fracture', 'Surgery']
genders = ['Male', 'Female']

df = pd.DataFrame({
    'Patient ID': range(1, n_patients + 1),
    'Age': np.random.randint(18, 80, n_patients),
    'Gender': np.random.choice(genders, n_patients),
    'Condition': np.random.choice(conditions, n_patients),
    'Days to Recovery': np.random.randint(3, 21, n_patients) + np.random.normal(0, 3, n_patients), # Recovery time with some variation
    'Number of Visits': np.random.randint(1, 10, n_patients), # Number of visits during recovery
    'Medications Prescribed': np.random.randint(1, 5, n_patients)
})

df['Days to Recovery'] = df['Days to Recovery'].apply(lambda x: max(0,x)) #Ensure no negative recovery days

# --- Recovery Time Distribution ---
print("\n--- Recovery Time Distribution ---")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['Days to Recovery'], bins=15, alpha=0.6, color='mediumseagreen')
plt.title("Distribution of Recovery Times")
plt.xlabel("Days to Recovery")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sm.qqplot(df['Days to Recovery'], line='s')
plt.title("Q-Q Plot of Recovery Times")
plt.tight_layout()
plt.show()

shapiro_test = stats.shapiro(df['Days to Recovery'])
print(f"Shapiro-Wilk Test: {shapiro_test}")

# --- Factors Affecting Recovery (Linear Regression) ---
print("\n--- Factors Affecting Recovery ---")
df = pd.get_dummies(df, columns=['Gender', 'Condition'])
X = df.drop(['Patient ID', 'Days to Recovery'], axis=1)
y = df['Days to Recovery']

X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# --- Predicting Resource Requirements (Using Number of Visits as a proxy) ---
print("\n--- Predicting Resource Requirements ---")

X_visits = df.drop(['Patient ID', 'Number of Visits'], axis=1)
y_visits = df['Number of Visits']

X_visits = sm.add_constant(X_visits)
X_train_visits, X_test_visits, y_train_visits, y_test_visits = train_test_split(X_visits, y_visits, test_size=0.2, random_state=42)
model_visits = sm.OLS(y_train_visits, X_train_visits).fit()
print("Visits Model Summary:")
print(model_visits.summary())

y_pred_visits = model_visits.predict(X_test_visits)
mse_visits = mean_squared_error(y_test_visits, y_pred_visits)
r2_visits = r2_score(y_test_visits, y_pred_visits)
print(f"Mean Squared Error (Visits): {mse_visits}")
print(f"R-squared (Visits): {r2_visits}")

# --- Outlier Analysis (Unusual Recovery Patterns) ---
print("\n--- Outlier Analysis (Unusual Recovery Patterns) ---")
plt.figure(figsize=(8, 6))
sns.boxplot(y='Days to Recovery', data=df)
plt.title("Boxplot of Recovery Times (Outlier Detection)")
plt.show()

Q1 = df['Days to Recovery'].quantile(0.25)
Q3 = df['Days to Recovery'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Days to Recovery'] < (Q1 - 1.5 * IQR)) | (df['Days to Recovery'] > (Q3 + 1.5 * IQR))]
print(f"Number of Outliers (IQR Method): {len(outliers)}")
print("Outliers:")
print(outliers.head())