import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_data.csv' with your actual data file)
# Assuming you have a CSV file with features (columns 1 to n) and labels in the last column
data = pd.read_csv('datos3.csv')

# Extract the "BMI" column
bmi_data = data['BMI']

# Create a time axis for your data (assuming you have a time dimension)
time_axis = range(len(bmi_data))

# Define the time step for labeling the Y-axis
time_step = 100

# Create a list of years for Y-axis labels
years = [str(year) for year in range(0, len(bmi_data), time_step)]

# Separate the target labels (y) from the features (X)
X = data.iloc[:, :-1]  # Assuming the last column contains labels (target)
y = data.iloc[:, -1]  # Assuming the last column contains labels (target)

# Split the dataset into features (X) and target labels (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Standardize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a neural network model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Binary classification (ill or not ill)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Calculate the trendline with a diagonal slope using linear regression
# In this example, we are creating a trendline that slopes upwards
trendline_coefficients = np.polyfit(time_axis, bmi_data, 1)
trendline = np.poly1d(trendline_coefficients)

# Modify the trendline coefficients to achieve the desired slope
# Increase the first coefficient for an upward-sloping line, decrease for downward-sloping
new_slope = 0.003 # Adjust as needed to control the slope
trendline[1] = new_slope

# Adjust the width and height as needed for the BMI data plot
plt.figure(figsize=(53, 18))

# Plot the "BMI" values over time
plt.plot(time_axis, bmi_data, marker='.', label='BMI Data')
plt.xlabel('Time (Years)', fontsize=22)
plt.ylabel('BMI', fontsize=22)
plt.xticks(range(0, len(bmi_data), time_step), years, fontsize=32)
plt.yticks(fontsize=32)
plt.title('BMI Over Time', fontsize=22)
plt.grid(True)

# Add an intersection point at a specific time point
intersection_time = 10  # Adjust as needed
intersection_point = bmi_data[intersection_time]
plt.scatter(intersection_time, intersection_point, color='red', label=f'Intersection Point ({years[intersection_time]} Years)', s=500)

# Plot the modified diagonal trendline
plt.plot(time_axis, trendline(time_axis), linestyle='--', color='yellow', linewidth=10, label='Trendline')

plt.legend()
plt.show()