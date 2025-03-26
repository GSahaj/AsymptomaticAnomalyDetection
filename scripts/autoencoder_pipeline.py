import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load preprocessed data (features and labels)
data = pd.read_csv('data_output/processed_data.csv')
x = data.drop(columns=['label']).values  # Features
y = data['label'].values  # Labels (1 = Normal, 0=Abnormal)

# Normalize Data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Split data ino training(80%) and test (20%) sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train autoencoder on normal breathing and introduce some abnormal data
x_train_normal = x_train[y_train == 1]  # Only normal data
x_train_abnormal = x_train[y_train == 0]  # Abnormal data
x_train_combined = np.vstack((x_train_normal, x_train_abnormal[:len(x_train_normal) // 2]))  # Mix a little abnormal data

# Autoencoder model
input_dim = x_train_combined.shape[1]
encoding_dim = 16

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(x_train_combined, x_train_combined, epochs=50, batch_size=32, shuffle=True,validation_data=(x_test, x_test))


# Compute reconstruction error threshold
def compute_threshold(autoencoder, x_train_normal):
    reconstructions = autoencoder.predict(x_train_normal)
    reconstruction_errors = np.mean(np.abs(reconstructions - x_train_normal), axis=1)
    return np.percentile(reconstruction_errors, 95)  # Set threshold at 95th percentile


threshold = compute_threshold(autoencoder, x_train_normal)


# Evalute on test data
def detect_anomalies(autoencoder, x_test, threshold):
    reconstructions = autoencoder.predict(x_test)
    reconstruction_errors = np.mean(np.abs(reconstructions - x_test), axis=1)
    return reconstruction_errors > threshold


anomaly_predictions = detect_anomalies(autoencoder, x_test, threshold)

# Print anomaly detection results
print("Asymptomatic Patients Detected: ", np.sum(anomaly_predictions))

# Save model and scaler
autoencoder.save('models/autoencoder.h5')
pd.to_pickle(scaler, 'models/scaler.pk1')


