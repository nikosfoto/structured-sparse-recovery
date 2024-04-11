import numpy as np

def block_omp(A, y, block_size, sparsity_level, tolerance=1e-6):
    """
    Block Orthogonal Matching Pursuit (Block OMP) algorithm
    
    Parameters:
        A (ndarray): Measurement matrix of shape (m, n).
        y (ndarray): Measurement vector of shape (m, ).
        block_size (int): Size of each block in the block sparse signal.
        sparsity_level (int): Desired signal sparsity level (NOT block sparsity level).
        tolerance (float): Tolerance parameter for early stopping (default: 1e-6).
        
    Returns:
        x_hat (ndarray): Estimated block sparse signal of shape (n, ).
    """
    m, n = A.shape
    num_blocks = n // block_size
    
    # Initialize residual and support set
    r = y.copy()
    support = set()
    
    # Main loop
    for _ in range(sparsity_level):
        # Compute inner products of residual with each block
        inner_products = np.abs(A.T @ r)
        
        # Flatten inner products for block-level selection
        flattened_inner_products = np.sum(inner_products.reshape(num_blocks, block_size), axis=1)
        
        # Find the block with maximum inner product
        max_block_idx = np.argmax(flattened_inner_products)
        
        # Add the indices of the selected block to the support set
        selected_block_indices = list(range(max_block_idx * block_size, (max_block_idx + 1) * block_size))
        support.update(selected_block_indices)
        
        # Update the least squares estimate using the selected blocks
        A_selected = A[:, sorted(list(support))]
        x_hat_selected = np.linalg.lstsq(A_selected, y, rcond=None)[0]
        
        # Compute the new residual
        r = y - A_selected @ x_hat_selected

        # Check for convergence
        if np.linalg.norm(r) < tolerance:
            break
    
    # Reconstruct the block sparse signal
    x_hat = np.zeros(n)
    x_hat[list(support)] = x_hat_selected
    
    return x_hat

# Example usage:
# A is the measurement matrix, y is the measurement vector
# block_size is the size of each block in the block sparse signal
# sparsity is the desired sparsity level of the solution
# x_hat = block_omp(A, y, block_size, sparsity)
