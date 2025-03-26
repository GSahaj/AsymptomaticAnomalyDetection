import librosa
from convert_audio import convert_audio
from noise_reduction import reduce_noise

def load_and_preprocess_audio(file_path, sr=16000):
    #Load, converts (if necessary), reduces noise and preprocesses audio

    #Convert to .wav if the file is not in .wav format
    if not file_path.endswith(".wav"):
        convert_audio(file_path, file_path.replace(".mp3", ".wav"))

    #Load the audio file
    audio, sr = librosa.load(file_path, sr=sr)

    #Apply noise reduction
    audio = reduce_noise(audio, sr)

    #Trim silence
    audio, _ = librosa.effects.trim(audio)

    return audio, sr