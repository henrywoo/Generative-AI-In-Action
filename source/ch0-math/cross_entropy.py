import numpy as np
import matplotlib.pyplot as plt

# Define binary cross-entropy function
def binary_cross_entropy(t, p):
    t = np.float_(t)
    p = np.float_(p)
    losses = []
    for tt, pp in zip(t, p):
        loss = -(tt * np.log(pp) + (1 - tt) * np.log(1 - pp))
        losses.append(loss)
        print(f'true_val = {tt}, predicted_val = {pp}, loss = {loss}')
    return losses

# Define categorical cross-entropy function
def categorical_cross_entropy(t_list, p_list):
    t_list = np.float_(t_list)
    p_list = np.float_(p_list)
    losses = []
    for t, p in zip(t_list, p_list):
        loss = -np.sum(t * np.log(p))
        losses.append(loss)
        print(f't:{t}, p:{p},loss:{loss}')
    return losses

# Example data for binary cross-entropy
true_vals_bin = [0, 1, 0, 1]
pred_vals_bin = [0.1, 0.9, 0.2, 0.8]

# Example data for categorical cross-entropy
true_vals_cat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
pred_vals_cat = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]]

# Calculate losses
bin_losses = binary_cross_entropy(true_vals_bin, pred_vals_bin)
cat_losses = categorical_cross_entropy(true_vals_cat, pred_vals_cat)

# Plot data and losses
plt.figure(figsize=(12, 10))
plt.style.use('ggplot')

# Plot raw data for binary cross-entropy
plt.subplot(2, 2, 1)
plt.plot(range(len(true_vals_bin)), true_vals_bin, marker='o', label='True Values', linestyle='--')
plt.plot(range(len(pred_vals_bin)), pred_vals_bin, marker='o', label='Predicted Values', linestyle='--')
plt.title('Binary Raw Data')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()

# Plot raw data for categorical cross-entropy
plt.subplot(2, 2, 2)
bar_width = 0.35
indices = np.arange(len(true_vals_cat))
for i, (true_vals, pred_vals) in enumerate(zip(true_vals_cat, pred_vals_cat)):
    plt.bar(indices + i * bar_width, true_vals, bar_width, label=f'True Values {i}', alpha=0.6)
    plt.bar(indices + i * bar_width, pred_vals, bar_width, label=f'Predicted Values {i}', alpha=0.3)
plt.title('Categorical Raw Data')
plt.xlabel('Class Index')
plt.ylabel('Probability')
plt.xticks(indices + bar_width, ['Class 1', 'Class 2', 'Class 3'])
plt.legend()

# Plot binary cross-entropy losses
plt.subplot(2, 2, 3)
plt.plot(range(len(bin_losses)), bin_losses, marker='o', color='blue', alpha=0.5)
plt.title('Binary Cross-Entropy Losses')
plt.xlabel('Sample Index')
plt.ylabel('Loss')

# Plot categorical cross-entropy losses
plt.subplot(2, 2, 4)
plt.plot(range(len(cat_losses)), cat_losses, marker='o', alpha=0.5, color='red')
plt.title('Categorical Cross-Entropy Losses')
plt.xlabel('Sample Index')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()
