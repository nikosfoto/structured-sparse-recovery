import numpy as np
np.random.seed(2024)

def generate_structured_data(num_measurements, num_variables, num_samples, sparsity_half, noise_var):

    V = np.random.normal(loc=0, scale=np.sqrt(noise_var), size=(num_measurements,num_samples))
    A = np.random.randn(num_measurements, num_variables)
    A = A / np.linalg.norm(A, axis = 0)

    X = np.zeros((num_variables, num_samples))
    for a in range(num_samples):
        x = np.zeros((num_variables,))
        non_zero_indices = np.random.choice(
            np.linspace(0, num_variables // 2 -1 , num=num_variables // 2, dtype=int, endpoint=False),
            size=sparsity_half,
            replace=False,
        )
        x[non_zero_indices] = np.random.uniform(size=(sparsity_half,))
        x[non_zero_indices + num_variables//2] = x[2 * non_zero_indices] + np.random.uniform(size=(sparsity_half,))
        X[:,a] = x

    Y = A @ X + V
    return Y, A, X