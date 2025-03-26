from pydub import AudioSegment

def convert_audio(input_path, output_path):
    #Converts audio files to .wav format using pydub

    #Load the audio file(supports .mp3, .ogg, etc)
    audio = AudioSegment.from_file(input_path)

    #Export the audio as .wav
    audio.export(output_path, format="wav")