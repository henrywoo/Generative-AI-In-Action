import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')
# Sample Input Data
x = np.array([-1.0, 1.0, 3.0, 0.5, -1.5])

# Calculate Intermediate Values
x_squared = x ** 2
mean_squared = np.mean(x_squared)
rms_value = np.sqrt(mean_squared)
x_normalized = x / rms_value

print(x_normalized)

# Create Figure
fig, axs = plt.subplots(2, 1, figsize=(7, 8))
fig.suptitle('RMSNorm Visualization (Single Neuron)')

# Plot Original Inputs
axs[0].bar(range(len(x)), x, color='skyblue', label='Original Inputs (x)')
axs[0].set_xticks(range(len(x)))
#axs[0].set_xticklabels([f'$x_{i+1}$' for i in range(len(x))])
axs[0].set_ylabel('$x$')
axs[0].legend()
axs[0].grid(axis='y', alpha=0.5)

# Plot Intermediate Calculations
for i, val in enumerate(x):
    axs[0].text(i, 0.2 if val>0 else -0.4, f'$x_{i+1}={val}$', ha='center')

axs[0].axhline(rms_value, color='orange', linestyle='dashed', linewidth=1,
               label=f'mean squared root ({rms_value:.2f})')
axs[1].axhline(rms_value, color='orange', linestyle='dashed', linewidth=1,
               label=f'Mean of Squares ({rms_value:.2f})')
axs[0].legend()

# Plot RMS Value
axs[1].text(len(x) - 1.5, rms_value + 0.4, f'RMSNorm: $x\'=x/sqrt(mean(x^2))$', ha='center')

# Plot Normalized Outputs
axs[1].bar(range(len(x_normalized)), x_normalized, color='pink', label='Normalized Outputs')
axs[1].set_xticks(range(len(x_normalized)))
axs[1].set_xticklabels([f'${{x_{i+1}}}^\prime$' for i in range(len(x_normalized))])
axs[1].set_ylabel('$x\'$')
for i, val in enumerate(x_normalized):
    axs[1].text(i, 0.1 if val>0 else -0.2, f'$x\'_{i+1}={val:.1f}$', ha='center')
axs[1].legend()
axs[1].grid(axis='y', alpha=0.5)

plt.savefig('rmsnorm.png')
plt.show()
