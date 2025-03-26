import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import cycle

#Make the visualization, spectrogram and graphs look nice
sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

def plot_waveform(audio, sr=16000):
    #Visualize the waveform of the audio signal
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Waveform of Audio")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.show()

def plot_spectrogram(audio, sr=16000):
    #Visualize the spectogram of the audio signal
    d = librosa.amplitude_to_db(librosa.stft(audio), ref=np.max)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(d, x_axis='time', y_axis='log', sr=sr)
    plt.colorbar(format="%+0.2f dB")
    plt.title("Spectrogram of Audio")
    plt.show()

def plot_mfcc(audio, sr=16000, n_mfcc=13):
    #Visualize the MFCCs (Mel-Frequency Ceptral Coefficents) of the audio signal

    #Extract MFCCs
    mfccs = librosa.features.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    #Plot MFCCs
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar(format="%+0.2f dB")
    plt.title("MFCCs of Audio")
    plt.xlabel("Time (seconds)")
    plt.ylabel("MFCC Coefficients")
    plt.show()

def plot_mel_spectrogram(audio, sr=16000):
    #Visualize the Mel Spectrogram of the audio signal
    s = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    s_dB = librosa.power_to_dB(s, ref=np.max)

    #Plot Mel Spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(s_dB, x_axis='time', y_axis='log', sr=sr)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram of Audio")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.show()