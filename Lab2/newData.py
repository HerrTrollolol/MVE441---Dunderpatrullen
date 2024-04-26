import numpy as np


def generate_dataset_with_clusters(
    n, l, h, cluster_prob=0.1, cluster_size_range=(2, 5), cluster_range=(1, 3)
):
    """
    Generate a matrix of size n x n with random integers between l and h,
    and clusters of repeated numbers at random locations.

    Parameters:
        n (int): Size of the matrix (number of rows and columns).
        l (int): Lower bound of the random integers.
        h (int): Higher bound of the random integers.
        cluster_prob (float): Probability of generating a cluster at any given location.
        cluster_size_range (tuple): Range of cluster sizes (min_size, max_size).
        cluster_range (tuple): Range of cluster range sizes (min_range, max_range).

    Returns:
        numpy.ndarray: The generated matrix.
    """
    matrix = np.random.randint(low=l, high=h + 1, size=(n, n))

    list = []
    num_cluster = 0
    last = None
    counter = 0
    for i in range(1000):
        last = np.randint(0, 1)
        if last == 0:
            counter += 1
            if counter == 3:
                num_cluster += 1
        else:
            counter += 1
            if counter == 3:
                num_cluster += 1

    for i in range(n):
        for j in range(n):
            if np.random.random() < cluster_prob:
                cluster_size = np.random.randint(
                    cluster_size_range[0], cluster_size_range[1] + 1
                )
                cluster_row_range = np.random.randint(
                    cluster_range[0], cluster_range[1] + 1
                )
                cluster_col_range = np.random.randint(
                    cluster_range[0], cluster_range[1] + 1
                )
                number = np.random.randint(l, h + 1)
                if i + cluster_row_range <= n and j + cluster_col_range <= n:
                    matrix[i : i + cluster_row_range, j : j + cluster_col_range] = (
                        number
                    )

    return matrix


# Example usage:
n = 5  # Size of the matrix
l = 0  # Lower bound
h = 10  # Higher bound
cluster_prob = 0.1  # Probability of generating a cluster
cluster_size_range = (2, 5)  # Range of cluster sizes
cluster_range = (1, 3)  # Range of cluster range sizes

dataset = generate_dataset_with_clusters(
    n, l, h, cluster_prob, cluster_size_range, cluster_range
)
print(dataset)
