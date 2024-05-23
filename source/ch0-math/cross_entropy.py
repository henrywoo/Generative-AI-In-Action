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
        print(f't:{t}, p:{p}, loss:{loss}')
    return losses

# Example data for binary cross-entropy
true_vals_bin = [0, 1, 1, 0, 0, 1, 1]
pred_vals_bin = [0.07, 0.91, 0.74, 0.23, 0.85, 0.17, 0.94]

# Example data for categorical cross-entropy
true_vals_cat = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
pred_vals_cat = [[0.91, 0.04, 0.05], [0.11, 0.8, 0.09], [0.3, 0.1, 0.6], [0.25, 0.4, 0.35]]

# Calculate losses
bin_losses = binary_cross_entropy(true_vals_bin, pred_vals_bin)
total_bce_loss = np.sum(bin_losses)
cat_losses = categorical_cross_entropy(true_vals_cat, pred_vals_cat)
total_cce_loss = np.sum(cat_losses)

# Set font size globally
plt.rcParams.update({'font.size': 10})

# Plot BCE data and losses
plt.figure(figsize=(9, 3))
plt.style.use('ggplot')

# Plot raw data for binary cross-entropy
plt.subplot(1, 2, 1)
plt.plot(range(len(true_vals_bin)), true_vals_bin, marker='o', label='True', linestyle='--')
plt.plot(range(len(pred_vals_bin)), pred_vals_bin, marker='o', label='Predicted', linestyle='--')
plt.title('True and Prediction Data', fontsize=10)
#plt.xlabel('Sample Index', fontsize=10)
plt.ylabel('Probability', fontsize=10)
plt.legend(fontsize=8)

# Plot binary cross-entropy losses
plt.subplot(1, 2, 2)
plt.plot(range(len(bin_losses)), bin_losses, marker='o', color='blue', alpha=0.5)
plt.title(f'Total BCE: {total_bce_loss:.4f}', fontsize=10)
#plt.xlabel('Sample Index', fontsize=10)
plt.ylabel('BCE', fontsize=10)

plt.tight_layout()
plt.savefig("cross_entropy_bce.png")
plt.show()

# Plot CCE data and losses
plt.figure(figsize=(15, 3))
plt.style.use('ggplot')

categories = ['Seal', 'Panda', 'Duck']
for i, (true_vals, pred_vals) in enumerate(zip(true_vals_cat, pred_vals_cat)):
    plt.subplot(1, 5, i+1)
    bar_width = 0.35
    indices = np.arange(len(categories))
    plt.bar(indices, true_vals, bar_width, label='True', alpha=0.6)
    plt.bar(indices + bar_width, pred_vals, bar_width, label='Predicted', alpha=0.3)
    cce = categorical_cross_entropy([true_vals], [pred_vals])
    plt.title(f'CCE - {np.sum(cce):.4f} for Sample {i+1}', fontsize=10)
    plt.xlabel(f'Prediction {i+1}', fontsize=10)
    plt.ylabel('Probability', fontsize=10)
    plt.xticks(indices + bar_width / 2, ['Seal', 'Panda', 'Duck'], fontsize=8)
    plt.legend(fontsize=8)

# Plot CCE losses
plt.subplot(1, 5, 5)
plt.plot(range(len(cat_losses)), cat_losses, marker='o', color='red', alpha=0.5)
plt.title(f'Total CCE: {total_cce_loss:.4f}', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.xticks(range(len(cat_losses)), [1, 2, 3, 4], fontsize=8)  # Set x-axis to [1, 2, 3, 4]

plt.tight_layout()
plt.savefig("cross_entropy_cce.png")
plt.show()
