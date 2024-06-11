from points import *

d_choices = [2, 3, 4, 6]  # we can't go much higher with 5x5 uniform grids!
K_choices = [1, 2, 3, 4]  # variable numbers of codebooks

npoints_hd = 1000  # points in high-dim spaces

print("Here we show the error for high-dimensional datasets using various levels of RVQ.")
print("'cost savings factor' refers to the ratio of using regular VQ (at uniform resolution)\nvs RVQ.")

for d in d_choices:
    print(f"\nd = {d}:")
    np.random.seed(1)
    data_hd = DATA_MIN + (DATA_MAX - DATA_MIN) * np.random.rand(npoints_hd, d)
    codebook0 = generate_codebook(n_grid, n_dim=d)
    codebooks = [codebook0 / (n_grid ** level) for level in range(max(K_choices))]
    for K in K_choices:
        indices = quantizer(data_hd, codebooks)
        recon = data_hd * 0
        for lil_k in range(K):  # reconstruct using all codebooks
            recon += codebooks[lil_k][indices[lil_k]]
        error = ((recon - data_hd) ** 2).mean()
        grid_0_points = n_grid ** (d)
        rvq_points = grid_0_points * K
        uni_res = grid_0_points ** K  # comparable uniform resolution
        savings = uni_res / rvq_points
        print(f"  K = {K}, error = {error:.2e}, cost savings factor = {savings:.1f}")
