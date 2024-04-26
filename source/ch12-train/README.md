# Let's Train The Model

<div style="float: right; margin-left: 20px;">
    <img src="cd_release.png" width="300" height="460" alt="CD Release">
</div>

[Youtube Music📀▶️](https://music.youtube.com/playlist?list=OLAK5uy_mxQFu_bdbL9nplWkvmsBfowqsgzXoJKCc) | 
[Amazon Music📀▶](https://music.amazon.com/tracks/B0D2M61V61?marketplaceId=ATVPDKIKX0DER&musicTerritory=US&ref=dm_sh_2DPOGOaQWe6BMUmCjVcMP5lpi)

## Initialization

## The Vanishing/Exploding Gradients Problems

### Vanishing

As gradients propagate back through each layer during backpropagation, they can get smaller and smaller due to multiplication with weights and derivatives of activation functions. When gradients are very small, the weights in the early layers of the network do not change significantly, which can slow down or completely halt the learning process in these layers.

Gradients tend to vanish towards the beginning of the network (i.e., the layers closer to the input). This is especially true in networks using activation functions like sigmoid or tanh, where derivative values are small and can cause the gradients to become smaller as they are propagated back through the network during training.

### Explosion

The exploding gradient problem occurs when the gradients of a model's parameters become excessively large during training, which can lead to unstable network behavior and divergence in the loss function, often causing model parameters to overshoot optimal values. This problem is particularly common in deep networks with many layers and certain types of recurrent neural networks (RNNs).

Unlike vanishing gradients, exploding gradients can occur with activation functions like ReLU if not properly managed because ReLU does not inherently limit the size of the output (it does not saturate as sigmoid or tanh does).

Exploding gradients are more common in layers closer to the output, or generally in networks where there are long sequences or very deep networks without any form of gradient normalization (like gradient clipping). This might also happen with improper weight initialization or high learning rates.

> Exploding can Occur in Any Layer but More Likely in the Output Side.

To demonstrate the exploding gradient problem, let's set up a simple recurrent neural network example using PyTorch. RNNs are notorious for experiencing exploding gradients, especially with long sequences and without any gradient clipping.

![](gradient_explosion_deepnet.png)

## Activation

![](activations.png)

## Normalization

## Regularization

## Optimizer

## LR Scheduler








