import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sm
import scipy.stats as stats
import seaborn as sns

# Sample Data (Replace with your actual data)
np.random.seed(0)
n_samples = 200
df = pd.DataFrame({
    'Price': np.exp(np.random.normal(12, 0.8, n_samples)),  # Prices (simulated log-normally distributed)
    'Size': np.random.randint(1000, 5000, n_samples),      # Size in sq ft
    'Bedrooms': np.random.randint(2, 6, n_samples),        # Number of bedrooms
    'Bathrooms': np.random.randint(1, 4, n_samples),       # Number of bathrooms
    'Location': np.random.choice(['DHA', 'Bahria Town', 'Gulberg', 'Model Town'], n_samples) # Example Societies
})

# --- Location Encoding (One-Hot Encoding) ---
print("\n--- Location Encoding ---")
df = pd.get_dummies(df, columns=['Location'], prefix='Location')
print(df.head())

# --- Normality Testing ---
print("\n--- Normality Testing ---")
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(df['Price'], bins=20, density=True, alpha=0.6, color='b')
plt.title("Histogram of House Prices")

plt.subplot(1, 2, 2)
sm.qqplot(df['Price'], line='s')
plt.title("Q-Q Plot of House Prices")

plt.tight_layout()
plt.show()

shapiro_test = stats.shapiro(df['Price'])
print("Shapiro-Wilk Test:", shapiro_test)

k2, p = stats.normaltest(df['Price'])
print("D'Agostino's K-squared test: statistics=%.3f, p=%.3f" % (k2, p))

# --- Log Transformation and Check Normality ---
print("\n--- Log Transformation and Check Normality ---")
df['LogPrice'] = np.log1p(df['Price'])

plt.figure(figsize=(6, 6))
sm.qqplot(df['LogPrice'], line='s')
plt.title("Q-Q Plot of Log Transformed House Prices")
plt.show()

shapiro_test_log = stats.shapiro(df['LogPrice'])
print("Shapiro-Wilk Test (Log Transformed):", shapiro_test_log)

# --- Feature Correlation ---
print("\n--- Feature Correlation ---")
correlation_matrix = df.corr()
print(correlation_matrix['Price'].sort_values(ascending=False))

plt.figure(figsize=(10, 8)) # increased figure size for better visualization of correlation matrix with location dummies
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Matrix")
plt.show()

# Example Scatter Plot
plt.figure(figsize=(6, 6))
plt.scatter(df['Size'], df['Price'])
plt.xlabel("Size")
plt.ylabel("Price")
plt.title("Scatter Plot: Size vs. Price")
plt.show()

# --- Handling Outliers (IQR Method) ---
print("\n--- Handling Outliers (IQR Method) ---")
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Price'] < (Q1 - 1.5 * IQR)) | (df['Price'] > (Q3 + 1.5 * IQR))]
print("Number of outliers:", len(outliers))

df_no_outliers = df[~((df['Price'] < (Q1 - 1.5 * IQR)) | (df['Price'] > (Q3 + 1.5 * IQR)))]

# --- Building the Linear Regression Model ---
print("\n--- Building the Linear Regression Model ---")

X = df_no_outliers.drop(['Price', 'LogPrice'], axis=1) # Include location dummies
y = df_no_outliers['Price']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())

# Example with Log Transformed Price
y_log = df_no_outliers['LogPrice']
model_log = sm.OLS(y_log, X).fit()
print("\n--- Linear Regression Model with Log Transformed Price ---")
print(model_log.summary())