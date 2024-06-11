import torch
from vector_quantize_pytorch import FSQ

def set_seed(seed=42):
    import random
    import numpy as np
    import tensorflow as tf
    import torch
    # Setting seed for random
    random.seed(42)
    # Setting seed for numpy
    np.random.seed(42)
    # Setting seed for tensorflow
    tf.random.set_seed(42)
    # Setting seed for torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
set_seed()

quantizer = FSQ(
    levels = [8, 5, 5, 5]
)

x = torch.randn(1, 1024, 4) # 4 since there are 4 levels
xhat, indices = quantizer(x)

print(torch.dist(xhat, x))
print(xhat.shape)
print(indices.shape)
# (1, 1024, 4)
# (1, 1024)

assert torch.all(xhat == quantizer.indices_to_codes(indices))