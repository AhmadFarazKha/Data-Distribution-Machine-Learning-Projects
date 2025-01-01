import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample Data (Pakistani Textile Mill Example - Fabric Quality)
np.random.seed(42)
n_samples = 300

df = pd.DataFrame({
    'Temperature (°C)': np.random.normal(30, 2, n_samples),  # Temperature during weaving
    'Humidity (%)': np.random.normal(60, 5, n_samples),     # Humidity during weaving
    'Spinning Speed (RPM)': np.random.normal(1500, 50, n_samples), # Speed of the spinning machines
    'Tensile Strength (N)': np.random.normal(250, 15, n_samples) + 0.5 * np.random.normal(0, 10, n_samples)*(np.random.normal(30, 2, n_samples)-30) - 0.2*np.random.normal(0, 10, n_samples)*(np.random.normal(60, 5, n_samples)-60), # Tensile strength of the fabric
    'Defects/100m': np.random.poisson(abs(0.1 * (np.random.normal(30, 2, n_samples)-30) + 0.05*(np.random.normal(60, 5, n_samples)-60) + 0.001*(np.random.normal(1500, 50, n_samples)-1500)+2), n_samples)  # Number of defects per 100 meters of fabric
})

# --- Distribution of Product Measurements (Tensile Strength) ---
print("\n--- Distribution of Tensile Strength ---")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['Tensile Strength (N)'], bins=20, alpha=0.6, color='skyblue')
plt.title("Distribution of Tensile Strength")
plt.xlabel("Tensile Strength (N)")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sm.qqplot(df['Tensile Strength (N)'], line='s')
plt.title("Q-Q Plot of Tensile Strength")
plt.tight_layout()
plt.show()

# --- Predicting Defect Rates (Linear Regression) ---
print("\n--- Predicting Defect Rates ---")
X = df[['Temperature (°C)', 'Humidity (%)', 'Spinning Speed (RPM)']]
y = df['Defects/100m']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# --- Identifying Influencing Factors (Correlation) ---
print("\n--- Identifying Influencing Factors ---")
correlation_matrix = df.corr()
print(correlation_matrix['Defects/100m'].sort_values(ascending=False)) #Correlation with Defects/100m

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Matrix")
plt.show()

# --- Quality Control Thresholds (Example using standard deviation) ---
print("\n--- Quality Control Thresholds ---")
mean_tensile = df['Tensile Strength (N)'].mean()
std_tensile = df['Tensile Strength (N)'].std()
lower_threshold = mean_tensile - 2 * std_tensile
upper_threshold = mean_tensile + 2 * std_tensile

print(f"Mean Tensile Strength: {mean_tensile:.2f} N")
print(f"Standard Deviation of Tensile Strength: {std_tensile:.2f} N")
print(f"Lower Threshold (2 std deviations): {lower_threshold:.2f} N")
print(f"Upper Threshold (2 std deviations): {upper_threshold:.2f} N")

#Example of plotting the thresholds
plt.figure(figsize=(8,6))
plt.hist(df['Tensile Strength (N)'], bins=20, alpha=0.6, color='skyblue')
plt.axvline(lower_threshold, color='red', linestyle='dashed', linewidth=1, label = "Lower Threshold")
plt.axvline(upper_threshold, color='red', linestyle='dashed', linewidth=1, label = "Upper Threshold")
plt.title("Distribution of Tensile Strength with Thresholds")
plt.xlabel("Tensile Strength (N)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

#Example of using the model to predict defects based on thresholds
temp_threshold_low = 27
temp_threshold_high = 33
humidity_threshold_low = 50
humidity_threshold_high = 70
spinning_speed_threshold_low = 1400
spinning_speed_threshold_high = 1600

new_data = pd.DataFrame({
    'Temperature (°C)': [temp_threshold_low,temp_threshold_high, 30, 30],
    'Humidity (%)': [60, 60, humidity_threshold_low, humidity_threshold_high],
    'Spinning Speed (RPM)': [1500, 1500, 1500, 1500]
})
predicted_defects = model.predict(new_data)
print("\nPredicted defects based on thresholds:")
print(predicted_defects)