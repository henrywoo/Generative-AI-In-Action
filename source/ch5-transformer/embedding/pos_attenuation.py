import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

d = 128  # Dimension
theta = lambda t: 10000 ** (-2 * t / d)

def f(m):
    total = 0
    for j in range(d // 2):
        inner_sum = np.sum(np.exp(1j * m * theta(np.arange(j + 1))))
        total += np.linalg.norm(inner_sum)
    return total / (d / 2)

# Range of m values to evaluate
m_values = np.arange(0, 1024)  # Up to 256

# Calculate f(m) for each m
f_values = np.array([f(m) for m in m_values])

# Plotting
plt.plot(m_values, f_values)
plt.xlabel('Relative Distance')
plt.ylabel('Relative Magnitude')
plt.title('RoPE Positional Encoding Attenuation')
#plt.grid(axis='y')
plt.savefig('pos_attenuation.png')
plt.show()
