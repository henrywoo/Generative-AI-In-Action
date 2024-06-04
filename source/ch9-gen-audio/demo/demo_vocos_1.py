import torchaudio
from vocos import Vocos
from hiq.vis import print_model
import torch

"""
🌳 Vocos<all params:13531650>
├── MelSpectrogramFeatures(feature_extractor)
│   └── MelSpectrogram(mel_spec)
├── VocosBackbone(backbone)
│   ├── Conv1d(embed)|weight[512,100,7]|bias[512]
│   ├── 💠 LayerNorm(norm,final_layer_norm)<🦜:1024x2>|weight[512]|bias[512]
│   └── ModuleList(convnext)
│       └── 💠 ConvNeXtBlock(0-7)<🦜:1580544x8>|gamma[512]
│           ┣━━ Conv1d(dwconv)|weight[512,1,7]|bias[512]
│           ┣━━ LayerNorm(norm)|weight[512]|bias[512]
│           ┣━━ Linear(pwconv1)|weight[1536,512]|bias[1536]
│           ┗━━ Linear(pwconv2)|weight[512,1536]|bias[512]
└── ISTFTHead(head)
    └── Linear(out)|weight[1026,512]|bias[1026]
"""

vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
print_model(vocos)

y, sr = torchaudio.load("/home/wukong/Music/GAN_final.wav")
if y.size(0) > 1:  # mix to mono
    y = y.mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)
y_hat = vocos(y)
print(y_hat.shape)

# Reconstruct audio from EnCodec tokens
audio_tokens = torch.randint(low=0, high=1024, size=(8, 200))  # 8 codeboooks, 200 frames
features = vocos.codes_to_features(audio_tokens)
bandwidth_id = torch.tensor([2])  # 6 kbps

audio = vocos.decode(features, bandwidth_id=bandwidth_id)