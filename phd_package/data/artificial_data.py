import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
np.random.seed(42)
n_points = 1000
time = np.arange(n_points)
# trend = 0.1 * time
seasonality = 10 * np.sin(2 * np.pi * time / 50)
noise = np.random.normal(scale=5, size=n_points)

# Generate time series data
# data = trend + seasonality + noise
data = seasonality + noise

# Create a DataFrame
df = pd.DataFrame({"Time": time, "Value": data})

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(df["Time"], df["Value"], label="Artificial Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Artificial Time Series Data")
plt.legend()
plt.show()

# Display the DataFrame
df.head()
