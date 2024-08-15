import numpy as np

P = np.array([0.5, 0.5])
Q = np.array([1e-10, 1.0-(1e-10)])

def kld(P, Q):
    kl_divergence = np.sum(P * np.log(P / Q))
    print("KL Divergence:", kl_divergence)

kld(P, Q)  # 10.8
kld(Q, P)  # 0.69


def info(x):
    kl_divergence = np.sum(P * np.log(P / Q))
    print("KL Divergence:", kl_divergence)
