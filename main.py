import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make results reproducible
np.random.seed(42)
tf.random.set_seed(42)

# Features: Area (m²), Rooms, Age (years), Distance from city center (km)
X = np.array([
    [60, 2, 15, 12],
    [75, 3, 10, 8],
    [90, 3, 8, 6],
    [110, 4, 5, 5],
    [130, 4, 3, 4],
    [150, 5, 2, 3],
    [80, 2, 20, 10],
    [100, 3, 12, 7],
    [120, 4, 8, 5],
    [140, 4, 6, 3],
    [160, 5, 4, 2],
    [70, 2, 18, 15],
    [85, 3, 14, 9],
    [95, 3, 7, 6],
    [115, 4, 9, 8],
    [125, 4, 5, 4],
    [145, 5, 3, 3],
    [170, 5, 1, 2],
    [65, 2, 25, 14],
    [105, 3, 11, 6],
    [135, 4, 7, 5],
    [155, 5, 5, 4],
    [180, 6, 2, 2],
    [90, 2, 16, 11],
    [100, 4, 4, 3],
    [120, 3, 10, 7],
    [145, 4, 12, 9],
    [165, 5, 6, 3],
    [75, 2, 9, 5],
    [110, 3, 15, 10]
], dtype=float)

# House prices in thousands of dollars
y = np.array([
    91, 120, 145, 176, 209, 239, 95, 135, 165, 195,
    230, 73, 112, 148, 151, 193, 225, 255, 62, 139,
    177, 213, 268, 91, 171, 157, 174, 234, 126, 133
], dtype=float)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1)
])

# Configure the training process
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss="mse"
)

# Train the neural network
model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=4,
    verbose=0
)

# Evaluate the model
predictions = model.predict(X_test, verbose=0).flatten()

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nNEURAL NETWORK RESULTS")
print("-" * 30)
print(f"MAE:  ${mae:.2f} thousand")
print(f"RMSE: ${rmse:.2f} thousand")
print(f"R2 Score: {r2:.3f}")

# Predict the price of a new house
new_house = np.array([[120, 4, 8, 5]])
new_house_scaled = scaler.transform(new_house)

predicted_price = model.predict(new_house_scaled, verbose=0)[0][0]

print("\nNEW HOUSE PREDICTION")
print("-" * 30)
print("Area: 120 m²")
print("Rooms: 4")
print("Age: 8 years")
print("Distance from city center: 5 km")
print(f"Predicted price: ${predicted_price:.2f} thousand")
print("\nHOUSE PRICE PREDICTOR")
print("Enter 'q' to quit.")

while True:
    area = input("\nEnter area in m²: ")

    if area.lower() == "q":
        print("Goodbye!")
        break

    try:
        area = float(area)
        rooms = int(input("Enter number of rooms: "))
        age = float(input("Enter house age in years: "))
        distance = float(input("Enter distance from city center in km: "))

        if area <= 0 or rooms <= 0 or age < 0 or distance < 0:
            print("Please enter valid positive values.")
            continue

        new_house = np.array([[area, rooms, age, distance]])
        new_house_scaled = scaler.transform(new_house)

        predicted_price = model.predict(
            new_house_scaled, verbose=0
        )[0][0]

        print(f"Predicted house price: ${predicted_price:.2f} thousand")

    except ValueError:
        print("Invalid input. Please enter numbers.")