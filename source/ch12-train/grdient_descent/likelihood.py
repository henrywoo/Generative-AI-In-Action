import numpy as np

def gaussian_likelihood(mu, sigma, x):
    n = len(x)
    prefactor = (1 / (np.sqrt(2 * np.pi) * sigma)) ** n
    exponent = np.exp(-np.sum((x - mu)**2) / (2 * sigma**2))
    return prefactor * exponent

def gaussian_log_likelihood(mu, sigma, x):
    n = len(x)
    log_prefactor = -n * np.log(np.sqrt(2 * np.pi) * sigma)
    log_exponent = -np.sum((x - mu)**2) / (2 * sigma**2)
    return log_prefactor + log_exponent

# 示例数据
np.random.seed(0)
mu_true = 0.0
sigma_true = 1.0
x = np.random.normal(mu_true, sigma_true, size=100)

# 计算似然函数和对数似然函数
mu_est = np.mean(x)
sigma_est = np.std(x)
likelihood = gaussian_likelihood(mu_est, sigma_est, x)
log_likelihood = gaussian_log_likelihood(mu_est, sigma_est, x)

print(f"Likelihood: {likelihood}")
print(f"Log-Likelihood: {log_likelihood}")
