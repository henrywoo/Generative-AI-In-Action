import seaborn as sns
import matplotlib.pyplot as plt

# Load the built-in 'diamonds' dataset
diamonds = sns.load_dataset('diamonds')

# Display the first few rows of the dataset
print(diamonds.head())

# Plotting with seaborn
sns.pairplot(diamonds, hue='cut')
plt.show()
