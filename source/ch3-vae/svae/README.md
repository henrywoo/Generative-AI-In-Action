# S-VAE and VQ-sVAE


## VQ-sVAE

### 1. Hard Traditional Vector Quantization

- Randomly select 512 super points in hyperball and doing k-means like clustering. The super points moves as the training goes.
- Hard quantization
- STE


### 2. Soft Traditional Vector Quantization

- Uniformally select 512 super points in hyperball and doing k-means like clustering. The super points moves as the training goes.
- Soft quantization


### 3. Soft Hyperball Vector Quantization

- Evenly generate 512 super points in hyperball surface. The super points are fixed!
- Soft quantization
- Entropy loss


### 4. Hard Hyperball Vector Quantization

- Evenly generate 512 super points in hyperball surface. The super points are fixed!
- Hard quantization
- Entropy loss


## Conclusion

- Soft quantization is good for using less Q points; More Quant points are worse sometimes.
- Add one more linear layer improved the performance obviously (better feature extraction)
- Add contrastive loss is good for reconstruction perf too
- Simple Linear Model seems better than 3-layer CNN for simple dataset MNIST
- [VQ loss should be optimized and it boost recon loss perf](https://wandb.ai/henrywu/vq_svae_v2/runs/f34lkbot?nw=nwuserhenrywu)