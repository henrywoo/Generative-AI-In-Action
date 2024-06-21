# VQGAN Training Models Architecture


## VQGAN

```python
🌳 VQGAN<all params:70720515>
├── Encoder(encoder)
│   └── Sequential(model)
│       ├── Conv2d(0)|weight[128,3,3,3]|bias[128]
│       ├── 💠 ResidualBlock(1-2,4-5)<🦜:295680x4>
│       │   ┗━━ Sequential(block)
│       │       ┣━━ 💠 GroupNorm(0-0,3-3)<🦜:256x2>
│       │       ┃   ┗━━ GroupNorm(gn)|weight[128]|bias[128]
│       │       ┗━━ 💠 Conv2d(2-2,5-5)<🦜:147584x2>|weight[128,128,3,3]|bias[128]
│       ├── 💠 DownSampleBlock(3-3,6-6)<🦜:147584x2>
│       │   ┗━━ Conv2d(conv)|weight[128,128,3,3]🇸 -(2, 2)|bias[128]🇸 -(2, 2)
│       ├── ResidualBlock(7)
│       │   ├── Sequential(block)
│       │   │   ├── GroupNorm(0)
│       │   │   │   └── GroupNorm(gn)|weight[128]|bias[128]
│       │   │   ├── Conv2d(2)|weight[256,128,3,3]|bias[256]
│       │   │   ├── GroupNorm(3)
│       │   │   │   └── GroupNorm(gn)|weight[256]|bias[256]
│       │   │   └── Conv2d(5)|weight[256,256,3,3]|bias[256]
│       │   └── Conv2d(channel_up)|weight[256,128,1,1]|bias[256]
│       ├── 💠 ResidualBlock(8-8,10-11)<🦜:1181184x3>
│       │   ┗━━ Sequential(block)
│       │       ┣━━ 💠 GroupNorm(0-0,3-3)<🦜:512x2>
│       │       ┃   ┗━━ GroupNorm(gn)|weight[256]|bias[256]
│       │       ┗━━ 💠 Conv2d(2-2,5-5)<🦜:590080x2>|weight[256,256,3,3]|bias[256]
│       ├── 💠 DownSampleBlock(9-9,12-12)<🦜:590080x2>
│       │   ┗━━ Conv2d(conv)|weight[256,256,3,3]🇸 -(2, 2)|bias[256]🇸 -(2, 2)
│       ├── ResidualBlock(13)
│       │   ├── Sequential(block)
│       │   │   ├── GroupNorm(0)
│       │   │   │   └── GroupNorm(gn)|weight[256]|bias[256]
│       │   │   ├── Conv2d(2)|weight[512,256,3,3]|bias[512]
│       │   │   ├── GroupNorm(3)
│       │   │   │   └── GroupNorm(gn)|weight[512]|bias[512]
│       │   │   └── Conv2d(5)|weight[512,512,3,3]|bias[512]
│       │   └── Conv2d(channel_up)|weight[512,256,1,1]|bias[512]
│       ├── 💠 NonLocalBlock(14-14,16-16,18-18)<🦜:1051648x3>
│       │   ┣━━ GroupNorm(gn)
│       │   ┃   ┗━━ GroupNorm(gn)|weight[512]|bias[512]
│       │   ┗━━ 💠 Conv2d(q,k,v,proj_out)<🦜:262656x4>|weight[512,512,1,1]|bias[512]
│       ├── 💠 ResidualBlock(15-15,17-17,19-19)<🦜:4721664x3>
│       │   ┗━━ Sequential(block)
│       │       ┣━━ 💠 GroupNorm(0-0,3-3)<🦜:1024x2>
│       │       ┃   ┗━━ GroupNorm(gn)|weight[512]|bias[512]
│       │       ┗━━ 💠 Conv2d(2-2,5-5)<🦜:2359808x2>|weight[512,512,3,3]|bias[512]
│       ├── GroupNorm(20)
│       │   └── GroupNorm(gn)|weight[512]|bias[512]
│       └── Conv2d(22)|weight[256,512,3,3]|bias[256]
├── Decoder(decoder)
│   └── Sequential(model)
│       ├── Conv2d(0)|weight[512,256,3,3]|bias[512]
│       ├── 💠 ResidualBlock(1-1,3-4,6-6,8-8)<🦜:4721664x5>
│       │   ┗━━ Sequential(block)
│       │       ┣━━ 💠 GroupNorm(0-0,3-3)<🦜:1024x2>
│       │       ┃   ┗━━ GroupNorm(gn)|weight[512]|bias[512]
│       │       ┗━━ 💠 Conv2d(2-2,5-5)<🦜:2359808x2>|weight[512,512,3,3]|bias[512]
│       ├── 💠 NonLocalBlock(2-2,5-5,7-7,9-9)<🦜:1051648x4>
│       │   ┣━━ GroupNorm(gn)
│       │   ┃   ┗━━ GroupNorm(gn)|weight[512]|bias[512]
│       │   ┗━━ 💠 Conv2d(q,k,v,proj_out)<🦜:262656x4>|weight[512,512,1,1]|bias[512]
│       ├── ResidualBlock(10)
│       │   ├── Sequential(block)
│       │   │   ├── GroupNorm(0)
│       │   │   │   └── GroupNorm(gn)|weight[512]|bias[512]
│       │   │   ├── Conv2d(2)|weight[256,512,3,3]|bias[256]
│       │   │   ├── GroupNorm(3)
│       │   │   │   └── GroupNorm(gn)|weight[256]|bias[256]
│       │   │   └── Conv2d(5)|weight[256,256,3,3]|bias[256]
│       │   └── Conv2d(channel_up)|weight[256,512,1,1]|bias[256]
│       ├── 💠 NonLocalBlock(11-11,13-13,15-15)<🦜:263680x3>
│       │   ┣━━ GroupNorm(gn)
│       │   ┃   ┗━━ GroupNorm(gn)|weight[256]|bias[256]
│       │   ┗━━ 💠 Conv2d(q,k,v,proj_out)<🦜:65792x4>|weight[256,256,1,1]|bias[256]
│       ├── 💠 ResidualBlock(12-12,14-14,17-19)<🦜:1181184x5>
│       │   ┗━━ Sequential(block)
│       │       ┣━━ 💠 GroupNorm(0-0,3-3)<🦜:512x2>
│       │       ┃   ┗━━ GroupNorm(gn)|weight[256]|bias[256]
│       │       ┗━━ 💠 Conv2d(2-2,5-5)<🦜:590080x2>|weight[256,256,3,3]|bias[256]
│       ├── 💠 UpSampleBlock(16-16,20-20)<🦜:590080x2>
│       │   ┗━━ Conv2d(conv)|weight[256,256,3,3]|bias[256]
│       ├── ResidualBlock(21)
│       │   ├── Sequential(block)
│       │   │   ├── GroupNorm(0)
│       │   │   │   └── GroupNorm(gn)|weight[256]|bias[256]
│       │   │   ├── Conv2d(2)|weight[128,256,3,3]|bias[128]
│       │   │   ├── GroupNorm(3)
│       │   │   │   └── GroupNorm(gn)|weight[128]|bias[128]
│       │   │   └── Conv2d(5)|weight[128,128,3,3]|bias[128]
│       │   └── Conv2d(channel_up)|weight[128,256,1,1]|bias[128]
│       ├── 💠 ResidualBlock(22-23,25-27)<🦜:295680x5>
│       │   ┗━━ Sequential(block)
│       │       ┣━━ 💠 GroupNorm(0-0,3-3)<🦜:256x2>
│       │       ┃   ┗━━ GroupNorm(gn)|weight[128]|bias[128]
│       │       ┗━━ 💠 Conv2d(2-2,5-5)<🦜:147584x2>|weight[128,128,3,3]|bias[128]
│       ├── 💠 UpSampleBlock(24-24,28-28)<🦜:147584x2>
│       │   ┗━━ Conv2d(conv)|weight[128,128,3,3]|bias[128]
│       ├── GroupNorm(29)
│       │   └── GroupNorm(gn)|weight[128]|bias[128]
│       └── Conv2d(31)|weight[3,128,3,3]|bias[3]
├── Codebook(codebook)
│   └── Embedding(embedding)|weight[1024,256]
└── 💠 Conv2d(quant_conv,post_quant_conv)<🦜:65792x2>|weight[256,256,1,1]|bias[256]
```

## Discriminator

```python
🌳 Discriminator<all params:2765633>
└── Sequential(model)
    ├── Conv2d(0)|weight[64,3,4,4]🇸 -(2, 2)|bias[64]🇸 -(2, 2)
    ├── Conv2d(2)|weight[128,64,4,4]🇸 -(2, 2)
    ├── BatchNorm2d(3)|weight[128]|bias[128]
    ├── Conv2d(5)|weight[256,128,4,4]🇸 -(2, 2)
    ├── BatchNorm2d(6)|weight[256]|bias[256]
    ├── Conv2d(8)|weight[512,256,4,4]
    ├── BatchNorm2d(9)|weight[512]|bias[512]
    └── Conv2d(11)|weight[1,512,4,4]|bias[1]
```