import numpy as np
from scipy.spatial.distance import cosine


def calculate_cosine_similarity_excluding_outliers(a, b):
    def remove_outliers(data):
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return np.where((data >= lower_bound) & (data <= upper_bound), data, np.nan)

    a_cleaned = remove_outliers(a)
    b_cleaned = remove_outliers(b)

    # Keep only the indices where neither a_cleaned nor b_cleaned is NaN
    valid_indices = ~np.isnan(a_cleaned) & ~np.isnan(b_cleaned)
    a_filtered = a_cleaned[valid_indices]
    b_filtered = b_cleaned[valid_indices]

    if len(a_filtered) == 0 or len(b_filtered) == 0:
        raise ValueError("No valid data points remain after outlier removal.")

    # Compute cosine similarity
    similarity = 1 - cosine(a_filtered, b_filtered)
    return similarity


# Example usage with dummy data
a = np.array([0.1, 0.2, 0.3, 0.4, 5.0, 6])  # Example vector with an outlier
b = np.array([0.1, 0.2, 0.3, 0.4, -5.0, 6], )  # Example vector with an outlier

similarity = calculate_cosine_similarity_excluding_outliers(a, b)
print(f"Cosine similarity excluding outliers: {similarity}")
