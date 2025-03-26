import librosa
import numpy as np


def extract_features(audio, sr=16000):
    # Extracts features from the audio, such as MFCCs, mean and std

    # Extracts MFCCs (Mel-frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Compute mean and std of MFCCS for each frame
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, acis=1)

    # Combine features (mean and std)
    features = np.concatenate([mfccs_mean, mfccs_std])
    return features
