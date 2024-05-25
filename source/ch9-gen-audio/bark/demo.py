import os
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
#from IPython.display import Audio

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
[MAN]The sculpture is ALREADY complete within the marble block, before I start my work.
It is already there, I just have to CHISEL AWAY the superfluous material.
"""
audio_array = generate_audio(text_prompt)

# save audio to disk
write_wav("bark_generation_4.wav", SAMPLE_RATE, audio_array)
  
# play text in notebook
#Audio(audio_array, rate=SAMPLE_RATE)



