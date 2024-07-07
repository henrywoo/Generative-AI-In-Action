import torch
import numpy as np

Codeword = torch.Tensor
Indices = torch.Tensor

def round_ste(z):
    """Round with straight through gradients."""
    zhat = torch.round(z)
    return z + (zhat - z).detach()

class FSQ:
    """Quantizer."""

    def __init__(self, levels: list[int], eps: float = 1e-3):
        self._levels = levels
        self._eps = eps
        self._levels_np = np.asarray(levels)
        self._basis = np.concatenate(
            ([1], np.cumprod(self._levels_np[:-1]))).astype(np.uint32)

        self._implicit_codebook = self.indexes_to_codes(
            torch.arange(self.codebook_size, dtype=torch.int32))

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return len(self._levels)

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return np.prod(self._levels)

    @property
    def codebook(self):
        """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
        return self._implicit_codebook

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = torch.tensor((self._levels_np - 1) * (1 - self._eps) / 2, dtype=z.dtype, device=z.device)
        offset = torch.tensor(np.where(self._levels_np % 2 == 1, 0.0, 0.5), dtype=z.dtype, device=z.device)
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z: torch.Tensor) -> Codeword:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))

        # Renormalize to [-1, 1].
        half_width = torch.tensor(self._levels_np // 2, dtype=quantized.dtype, device=quantized.device)
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        # Scale and shift to range [0, ..., L-1]
        half_width = torch.tensor(self._levels_np // 2, dtype=zhat_normalized.dtype, device=zhat_normalized.device)
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = torch.tensor(self._levels_np // 2, dtype=zhat.dtype, device=zhat.device)
        return (zhat - half_width) / half_width

    def codes_to_indexes(self, zhat: Codeword) -> Indices:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.num_dimensions
        zhat = self._scale_and_shift(zhat)
        return (zhat * torch.tensor(self._basis, dtype=zhat.dtype, device=zhat.device)).sum(dim=-1).to(torch.int32)

    def indexes_to_codes(self, indices: Indices) -> Codeword:
        """Inverse of `codes_to_indexes`."""
        indices = indices[..., None]
        basis_tensor = torch.tensor(self._basis, dtype=indices.dtype, device=indices.device)
        levels_tensor = torch.tensor(self._levels_np, dtype=indices.dtype, device=indices.device)
        codes_non_centered = (indices // basis_tensor) % levels_tensor
        return self._scale_and_shift_inverse(codes_non_centered)

# Example usage and tests
fsq = FSQ(levels=[3, 6, 4])
print(fsq.codebook)

z = torch.tensor([10.25, 0.6, -7.0])
zhat = fsq.quantize(z)
print(f"Quantized {z} -> {zhat}")

# We can map to an index in the codebook.
idx = fsq.codes_to_indexes(zhat)
print(f"Code {zhat} is the {idx}-th index.")

# Back to code
code_out = fsq.indexes_to_codes(idx)
print(f"Index {idx} mapped back to {code_out}.")

fsq = FSQ(levels=[5, 4, 3])

d = fsq.num_dimensions
z = torch.rand((3, 8, 8, d))
zhat = fsq.quantize(z)
assert zhat.shape == (3, 8, 8, d)

indices = fsq.codes_to_indexes(zhat)
assert indices.shape == (3, 8, 8)

zhat_out = fsq.indexes_to_codes(indices)
assert zhat_out.shape == zhat.shape

torch.testing.assert_close(zhat, zhat_out)

fsq = FSQ(levels=[3, 4])
print(fsq.codebook)
