# Import necessary libraries and configure settings
import torch
import torchaudio
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

import ChatTTS
#from IPython.display import Audio

# Initialize and load the model:
chat = ChatTTS.Chat()
chat.load_models() # Set to True for better performance

# Define the text input for inference (Support Batching)
texts = ["""
To introduce myself, I have 20 year working experience, where 10 years are related to AI and machine learning. Especially recently 6 years, I have completely focused on Generative AI, Computer Vision, NLP, LLM and so on. I also have some experience on interviewing candidates and holding debrief meetings.

If you have any questions, please contact me at henry wu 2016 at gmail dot com 

To learn this course effectively, you are better off having a good understanding of deep neural network, like knowing what is input, hidden and output layer.

We will use several simple dataset for demonstration purpose, such as MNIST, Fashion-MNIST and Cifar10.

"""]

# Perform inference and play the generated audio
wavs = chat.infer(texts)
#Audio(wavs[0], rate=24_000, autoplay=True)

# Save the generated audio
torchaudio.save("output_2.wav", torch.from_numpy(wavs[0]), 24000)

