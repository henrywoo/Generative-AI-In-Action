import abc
import torch.nn as nn

class Latent(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
        self.is_continuous = False
        self.num_latents = 0
        self.num_inputs = 0

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

import torch
import torch.nn.functional as F

class QuantizedLatent(Latent):
    def __init__(self, num_latents, num_values_per_latent, optimize_values, device='cpu'):
        super().__init__()
        self.is_continuous = False
        self.num_latents = num_latents
        self.num_inputs = num_latents
        self.optimize_values = optimize_values
        self.device = device

        if isinstance(num_values_per_latent, int):
            self.num_values_per_latent = [num_values_per_latent] * num_latents
        else:
            self.num_values_per_latent = num_values_per_latent

        self._values_per_latent = [torch.linspace(-0.5, 0.5, self.num_values_per_latent[i], device=self.device) for i in range(num_latents)]

    @property
    def values_per_latent(self):
        if self.optimize_values:
            return self._values_per_latent
        else:
            return [v.detach() for v in self._values_per_latent]

    @staticmethod
    def quantize(x, values):
        distances = torch.abs(x - values)
        index = torch.argmin(distances)
        return values[index], index

    def forward(self, x):
        quantized_and_indices = [self.quantize(x_i, values_i) for x_i, values_i in zip(x, self.values_per_latent)]
        quantized = torch.stack([qi[0] for qi in quantized_and_indices])
        indices = torch.stack([qi[1] for qi in quantized_and_indices])
        quantized_sg = x + (quantized - x).detach()
        outs = {
            'z_continuous': x,
            'z_quantized': quantized,
            'z_hat': quantized_sg,
            'z_indices': indices
        }

        return outs

    def sample(self):
        ret = []
        for values in self.values_per_latent:
            ret.append(torch.choice(values))
        return torch.tensor(ret, device=self.device)


def verify_quantized_latent():
    # Test parameters
    num_latents = 3
    num_values_per_latent = 5
    optimize_values = False
    device = 'cpu'

    # Create an instance of QuantizedLatent
    latent = QuantizedLatent(num_latents, num_values_per_latent, optimize_values, device=device)
    print(latent._values_per_latent)

    # Generate some random input data
    x = torch.randn(num_latents, device=device)

    # Perform a forward pass
    output = latent(x)
    print(x)
    print(output)

    # Check the output structure
    assert 'z_continuous' in output, "Output missing 'z_continuous'"
    assert 'z_quantized' in output, "Output missing 'z_quantized'"
    assert 'z_hat' in output, "Output missing 'z_hat'"
    assert 'z_indices' in output, "Output missing 'z_indices'"

    # Check the types and shapes of the outputs
    assert isinstance(output['z_continuous'], torch.Tensor), "'z_continuous' is not a torch.Tensor"
    assert isinstance(output['z_quantized'], torch.Tensor), "'z_quantized' is not a torch.Tensor"
    assert isinstance(output['z_hat'], torch.Tensor), "'z_hat' is not a torch.Tensor"
    assert isinstance(output['z_indices'], torch.Tensor), "'z_indices' is not a torch.Tensor"

    assert output['z_continuous'].shape == (num_latents,), "'z_continuous' shape is incorrect"
    assert output['z_quantized'].shape == (num_latents,), "'z_quantized' shape is incorrect"
    assert output['z_hat'].shape == (num_latents,), "'z_hat' shape is incorrect"
    assert output['z_indices'].shape == (num_latents,), "'z_indices' shape is incorrect"

    print("All tests passed!")


if __name__ == "__main__":
    verify_quantized_latent()
