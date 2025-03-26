import os
from glob import glob
import numpy as np
import pandas as pd
import librosa
import subprocess

from preprocessing import load_and_preprocess_audio
from feature_extraction import extract_features
from visualization import plot_mfcc, plot_mel_spectrogram, plot_waveform, plot_spectrogram

# Path to the datasets
normal_breathing_files = glob('dataset/normal_breathing/*wav')  # Input data files
abnormal_breathing_files = glob('dataset/abnormal_breathing/*wav')  # Input data files

# Stores features
normal_feature = []
abnormal_feature = []

for file in normal_breathing_files:
    audio, sr = load_and_preprocess_audio(file)
    features = extract_features(audio, sr)
    normal_feature.append(features)

    # Visualize for Normal Breathing
    plot_waveform(audio, sr)
    plot_spectrogram(audio, sr)
    plot_mfcc(audio, sr)
    plot_mel_spectrogram(audio, sr)

print("normal breathing file processed....")

for file in abnormal_breathing_files:
    audio, sr = load_and_preprocess_audio(file)
    features = extract_features(audio, sr)
    normal_feature.append(features)

    # Visualize for Abnormal breathing
    plot_waveform(audio, sr)
    plot_spectrogram(audio, sr)
    plot_mfcc(audio, sr)
    plot_mel_spectrogram(audio, sr)

print("abnormal breathing file processed....")

# Combine features from both normal and abnormal data
normal_features = np.array(normal_feature)
abnormal_feature = np.array(abnormal_feature)

x = np.concatenate([normal_features, abnormal_feature], axis=0)
y = np.concatenate([np.ones(len(normal_features)), np.zeros(len(abnormal_feature))], axis=0)

# Store the features and labels into a pandas Dataframe
df = pd.DataFrame(x)
df['label'] = y

# Save the processed data as CSV for future use
output_file = '../data_output/processed_data.csv'
df.to_csv(output_file, index=False)

# Print the first few rows of the processed data
print("Processed data saved: ", output_file)
print(df.head())

# Script to run the subprocess for the autoencoder
autoencoder_script_path = r'C:\Users\Sahaj\PycharmProjects\AnomalyDetectionModel\scripts\autoencoder_pipeline.py'
print("Running autoencoder script...")
subprocess.run(['python', autoencoder_script_path])

print("Autoencoder script execution completed")

# Script to run the subprocess for the evaluate.py
evaluate_script_path = r'C:\Users\Sahaj\PycharmProjects\AnomalyDetectionModel\scripts\evaluate.py'
print("Running evaluate script...")
subprocess.run(['python', evaluate_script_path])

print("Evaluate script execuation completed")