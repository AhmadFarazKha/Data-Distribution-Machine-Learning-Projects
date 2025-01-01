import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample Data (Daraz Website Traffic)
np.random.seed(42)
dates = pd.date_range('2022-01-01', '2023-12-31')
n_days = len(dates)

df = pd.DataFrame({
    'Date': dates,
    'Daily Traffic': np.random.randint(5000, 20000, n_days) + np.sin(np.linspace(0, 4*np.pi, n_days))*5000 + np.random.normal(0, 1000, n_days), # Simulate seasonality and noise
    'Marketing Spend (PKR)': np.random.randint(0, 100000, n_days)
})

#Hourly data for peak analysis
n_hours = n_days * 24
timestamps = pd.date_range('2022-01-01 00:00:00', '2023-12-31 23:00:00', freq='H')

df_hourly = pd.DataFrame({
    'Timestamp': timestamps,
    'Hourly Traffic': np.random.randint(100, 800, n_hours) + np.sin(np.linspace(0, 8*np.pi, n_hours))*200 + np.random.normal(0, 50, n_hours)
})

# --- Daily Traffic Distribution ---
print("\n--- Daily Traffic Distribution ---")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['Daily Traffic'], bins=20, alpha=0.6, color='orange')
plt.title("Daily Traffic Distribution")
plt.xlabel("Daily Traffic")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sm.qqplot(df['Daily Traffic'], line='s')
plt.title("Q-Q Plot of Daily Traffic")
plt.tight_layout()
plt.show()

shapiro_test = stats.shapiro(df['Daily Traffic'])
print(f"Shapiro-Wilk Test: {shapiro_test}")

# --- Predicting Traffic (Linear Regression with Seasonality) ---
print("\n--- Predicting Traffic (Linear Regression with Seasonality) ---")
df['DayOfWeek'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter

X = df[['DayOfWeek', 'Month', 'Quarter', 'Marketing Spend (PKR)']]
y = df['Daily Traffic']

X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# --- Identifying Peak Usage Patterns ---
print("\n--- Identifying Peak Usage Patterns ---")
df_hourly['Hour'] = df_hourly['Timestamp'].dt.hour
df_hourly['DayOfWeek'] = df_hourly['Timestamp'].dt.dayofweek

peak_traffic = df_hourly.groupby(['DayOfWeek', 'Hour'])['Hourly Traffic'].mean().unstack()

plt.figure(figsize=(10, 6))
sns.heatmap(peak_traffic, cmap="YlGnBu")
plt.title("Average Hourly Traffic by Day of the Week")
plt.xlabel("Hour of the Day")
plt.ylabel("Day of the Week (0=Monday, 6=Sunday)")
plt.show()

# --- Example Forecast for next 7 days ---
print("\n--- Example Forecast for next 7 days ---")
last_date = df['Date'].max()
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)
future_df = pd.DataFrame({'Date': future_dates})
future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
future_df['Month'] = future_df['Date'].dt.month
future_df['Quarter'] = future_df['Date'].dt.quarter
future_df['Marketing Spend (PKR)'] = 50000 #Example marketing spend for the next 7 days
X_future = sm.add_constant(future_df[['DayOfWeek', 'Month', 'Quarter', 'Marketing Spend (PKR)']])
future_predictions = model.predict(X_future)
print(future_predictions)

plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Daily Traffic'], label='Historical Traffic')
plt.plot(future_df['Date'], future_predictions, label='Predicted Traffic', color='red')
plt.xlabel("Date")
plt.ylabel("Daily Traffic")
plt.title("Traffic Forecast")
plt.legend()
plt.show()