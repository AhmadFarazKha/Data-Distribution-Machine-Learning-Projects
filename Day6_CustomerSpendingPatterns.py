import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample Data (Pakistani Retail Chain - Hypermarket Example)
np.random.seed(42)
n_customers = 500
locations = ['Karachi', 'Lahore', 'Islamabad', 'Peshawar']
product_categories = ['Grocery', 'Electronics', 'Clothing', 'Household']
dates = pd.date_range('2022-01-01', '2023-12-31')
n_days = len(dates)

df = pd.DataFrame({
    'Customer ID': np.repeat(range(1, n_customers + 1), n_days),
    'Date': np.tile(dates, n_customers),
    'Location': np.random.choice(locations, n_customers * n_days),
    'Total Spending (PKR)': np.random.randint(500, 10000, n_customers * n_days) + np.sin(np.linspace(0, 4*np.pi, n_customers * n_days))*2000 + np.random.normal(0, 1000, n_customers * n_days),
    'Product Category': np.random.choice(product_categories, n_customers * n_days)
})
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter

# --- Spending Distribution ---
print("\n--- Spending Distribution ---")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['Total Spending (PKR)'], bins=20, alpha=0.6, color='skyblue')
plt.title("Spending Distribution")
plt.xlabel("Total Spending (PKR)")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sm.qqplot(df['Total Spending (PKR)'], line='s')
plt.title("Q-Q Plot of Spending")
plt.tight_layout()
plt.show()

shapiro_test = stats.shapiro(df['Total Spending (PKR)'])
print(f"Shapiro-Wilk Test: {shapiro_test}")

# --- Spending Patterns Across Locations ---
print("\n--- Spending Patterns Across Locations ---")
plt.figure(figsize=(8, 6))
sns.boxplot(x='Location', y='Total Spending (PKR)', data=df)
plt.title("Spending by Location")
plt.show()

# --- Seasonal Trends ---
print("\n--- Seasonal Trends ---")
seasonal_trends = df.groupby('Month')['Total Spending (PKR)'].mean()
plt.figure(figsize=(10, 6))
seasonal_trends.plot(kind='line', marker='o')
plt.title("Seasonal Spending Trends")
plt.xlabel("Month")
plt.ylabel("Average Spending (PKR)")
plt.xticks(range(1,13))
plt.show()

# --- Predicting Future Spending (Linear Regression with Seasonality and Location) ---
print("\n--- Predicting Future Spending ---")
df = pd.get_dummies(df, columns=['Location', 'Product Category'])
X = df.drop(['Customer ID', 'Date', 'Total Spending (PKR)'], axis=1)
y = df['Total Spending (PKR)']

X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

#Example forecast for next month for each location
last_date = df['Date'].max()
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=1, freq='M')

future_df = pd.DataFrame({'Date': future_dates})
future_df['Month'] = future_df['Date'].dt.month
future_df['Quarter'] = future_df['Date'].dt.quarter

for location in locations:
    for category in product_categories:
      future_df[f'Location_{location}'] = 0
      future_df[f'Product Category_{category}'] = 0
future_df['Location_Karachi'][0] = 1
future_df['Product Category_Grocery'][0] = 1
X_future = sm.add_constant(future_df.drop(['Date'], axis=1))
future_predictions = model.predict(X_future)
print("\nExample Forecast for next month for Karachi for Grocery:")
print(future_predictions)

future_df['Location_Karachi'][0] = 0
future_df['Location_Lahore'][0] = 1
future_df['Product Category_Grocery'][0] = 0
future_df['Product Category_Electronics'][0] = 1
X_future = sm.add_constant(future_df.drop(['Date'], axis=1))
future_predictions = model.predict(X_future)
print("\nExample Forecast for next month for Lahore for Electronics:")
print(future_predictions)

# --- Outlier Analysis (Boxplot and IQR) ---
print("\n--- Outlier Analysis ---")
plt.figure(figsize=(8, 6))
sns.boxplot(y='Total Spending (PKR)', data=df)
plt.title("Boxplot of Spending (Outlier Detection)")
plt.show()

Q1 = df['Total Spending (PKR)'].quantile(0.25)
Q3 = df['Total Spending (PKR)'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Total Spending (PKR)'] < (Q1 - 1.5 * IQR)) | (df['Total Spending (PKR)'] > (Q3 + 1.5 * IQR))]
print(f"Number of Outliers (IQR Method): {len(outliers)}")
print("Outliers:")
print(outliers.head())