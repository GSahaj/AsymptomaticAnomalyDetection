import numpy as np
import pandas as pd
import tensorflow as tf
from tf.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle

# Load test data
data = pd.read_csv('data_output/processed_data.csv')
x = data.drop(columns=['label']).values  # Features
y = data['label'].values  # Labels (1 = Normal, 0 = Abnormal)

# Load scaler
with open('models/scaler.pkl', 'rb') as f:  # Fixed file extension
    scaler = pickle.load(f)

x_scaled = scaler.transform(x)

# Load trained autoencoder model
autoencoder = load_model('models/autoencoder.h5')

# Compute reconstruction errors
reconstructions = autoencoder.predict(x_scaled)
reconstruction_errors = np.mean(np.abs(reconstructions - x_scaled), axis=1)

# Compute threshold (95th percentile of normal reconstruction errors)
def compute_threshold(model, normal_data):
    normal_reconstructions = model.predict(normal_data)  # Fixed typo and scope issue
    normal_errors = np.mean(np.abs(normal_reconstructions - normal_data), axis=1)
    return np.percentile(normal_errors, 95)

# Identify normal data to compute threshold
x_normal = x_scaled[y == 1]
if len(x_normal) == 0:
    raise ValueError("No normal samples found in the data to compute threshold")

threshold = compute_threshold(autoencoder, x_normal)
anomaly_prediction = reconstruction_errors > threshold

# Evaluate results
num_anomalies = np.sum(anomaly_prediction)
print(f"Total anomalies detected: {num_anomalies}")
print(f"Threshold used: {threshold:.4f}")

# Save predictions
data['Anomaly'] = anomaly_prediction.astype(int)
data['Reconstruction_Error'] = reconstruction_errors  # Add error values for analysis
data.to_csv('data_output/evaluation_results.csv', index=False)