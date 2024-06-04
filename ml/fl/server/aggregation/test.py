from typing import List, Tuple
import numpy as np


def is_outlier(data: np.ndarray, threshold: float = 3.0) -> bool:
    """
    Check if the data contains outliers using z-score method.

    Parameters:
    - data: numpy array of weights.
    - threshold: z-score threshold to identify outliers.

    Returns:
    - True if data contains outliers, False otherwise.
    """
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    return np.any(z_scores > threshold)


def median_aggregate(results: List[Tuple[List[np.ndarray], int]], num_iterations: int = 3) -> np.ndarray:
    """
    Compute the median across weights with outlier detection and iterative median computation.

    Parameters:
    - results: A list of tuples, where each tuple contains a list of numpy arrays (weights for each layer)
               and an integer (usually representing the number of samples for the client).
    - num_iterations: Number of iterations for computing the median to ensure stability.

    Returns:
    - A list of numpy arrays representing the median weights for each layer.
    """
    if not results:
        raise ValueError("The input 'results' cannot be empty.")

    # Extract weights from results and apply outlier detection
    weights = []
    for weight_list, _ in results:
        filtered_weights = []
        for layer in weight_list:
            if not is_outlier(layer):
                filtered_weights.append(layer)
        if filtered_weights:
            weights.append(filtered_weights)

    if not weights or not weights[0]:
        raise ValueError("The weights in 'results' are malformed or empty.")

    # Iteratively compute the median for each layer
    for _ in range(num_iterations):
        weights_prime: List[np.ndarray] = [
            np.median(layer_updates, axis=0)
            for layer_updates in zip(*weights)
        ]
        # Update weights with the new median
        weights = [weights_prime] * len(weights)

    return weights_prime

# Example usage:
results = [
    ([np.array([1, 2, 3]), np.array([4, 5, 6])], 10),
    ([np.array([3, 2, 1]), np.array([6, 5, 4])], 20)
]
median_weights = median_aggregate(results)
print(median_weights)