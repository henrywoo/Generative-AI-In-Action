import numpy as np
import matplotlib.pyplot as plt
from points import *

d_choices = [2, 3, 4, 6]  # we can't go much higher with 5x5 uniform grids!
K_choices = [1, 2, 3, 4]  # variable numbers of codebooks

npoints_hd = 1000  # points in high-dim spaces

errors = np.zeros((len(d_choices), len(K_choices)))
savings_factors = np.zeros((len(d_choices), len(K_choices)))

print("Here we show the error for high-dimensional datasets using various levels of RVQ.")
print("'cost savings factor' refers to the ratio of using regular VQ (at uniform resolution)\nvs RVQ.")

for i, d in enumerate(d_choices):
    print(f"\nd = {d}:")
    np.random.seed(1)
    data_hd = DATA_MIN + (DATA_MAX - DATA_MIN) * np.random.rand(npoints_hd, d)
    codebook0 = generate_codebook(n_grid, n_dim=d)
    codebooks = [codebook0 / (n_grid ** level) for level in range(max(K_choices))]
    for j, K in enumerate(K_choices):
        indices = quantizer(data_hd, codebooks)
        recon = data_hd * 0
        for lil_k in range(K):  # reconstruct using all codebooks
            recon += codebooks[lil_k][indices[lil_k]]
        error = ((recon - data_hd) ** 2).mean()
        grid_0_points = n_grid ** d
        rvq_points = grid_0_points * K
        uni_res = grid_0_points ** K  # comparable uniform resolution
        savings = uni_res / rvq_points
        errors[i, j] = error
        savings_factors[i, j] = savings
        print(f"  K = {K}, error = {error:.2e}, cost savings factor = {savings:.1f}")

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(6, 4.8))

# Plot 1: d, k vs. error
for i, d in enumerate(d_choices):
    axs[0].plot(K_choices, errors[i, :], marker='o', label=f'd={d}', alpha=0.3)
axs[0].set_title('Error vs. K for different dimensions (d)', fontsize=10)
axs[0].set_xlabel('K', fontsize=8)
axs[0].set_ylabel('Error', fontsize=8)
axs[0].set_yscale('log')
axs[0].legend()
axs[0].grid(True)

# Plot 2: d, k vs. cost savings factor
for i, d in enumerate(d_choices):
    axs[1].plot(K_choices, savings_factors[i, :], marker='o', label=f'd={d}')
axs[1].set_title('Cost Savings Factor vs. K for different dimensions (d)', fontsize=10)
axs[1].set_xlabel('K', fontsize=8)
axs[1].set_ylabel('Cost Savings Factor', fontsize=8)
axs[1].set_yscale('log')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig('images/error_k_d.png')
plt.show()
