import noisereduce as nr
import librosa

def reduce_noise(audio, sr=160000):
    #Reduce background noise from the audio signal

    #Apply noise reducion
    clean_audio = nr.reduce_noise(y=audio, sr=sr)
    return clean_audio