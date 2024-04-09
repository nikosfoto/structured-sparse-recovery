import numpy as np

def structured_omp(A, y, K, pattern):
    """
    Structured Orthogonal Matching Pursuit (S-OMP) algorithm for structured sparse recovery.

    Parameters:
        A (ndarray): CS matrix (M x N).
        y (ndarray): Measurement vector (M x 1).
        K (int): Sparsity level.
        pattern (ndarray): Structured sparsity pattern indicating the indices of non-zero elements.

    Returns:
        ndarray: Recovered sparse signal (N x 1).
    """
    N = A.shape[1]
    x_hat = np.zeros(N)  # Initialize the estimated sparse signal
    residual = y.copy()  # Initialize the residual

    # Initialize support set
    support_set = []

    # Main loop
    for _ in range(K):
        # Compute inner products
        inner_products = np.abs(np.dot(A.T, residual))

        # Consider only the indices corresponding to the structured sparsity pattern
        inner_products_filtered = inner_products[pattern]

        # Find index of maximum inner product
        max_index = np.argmax(inner_products_filtered)

        # Map back to original indices
        original_index = np.where(pattern)[0][max_index]

        # Add selected index to support set
        support_set.append(original_index)

        # Update submatrix of A and compute least squares solution
        A_s = A[:, support_set]
        x_hat_s = np.linalg.lstsq(A_s, y, rcond=None)[0]

        # Update estimated signal
        x_hat[support_set] = x_hat_s

        # Compute new residual
        residual = y - np.dot(A, x_hat)

    return x_hat
