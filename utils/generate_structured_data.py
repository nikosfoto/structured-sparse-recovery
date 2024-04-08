import numpy as np
from scipy.linalg import hadamard
np.random.seed(2024)

def generate_structured_data(num_measurements, num_variables, num_samples, sparsity_half, noise_var, sensing_matrix_choice='normal'):

    V = np.random.normal(loc=0, scale=np.sqrt(noise_var), size=(num_measurements,num_samples))

    if sensing_matrix_choice == 'normal':
        A = np.random.randn(num_measurements, num_variables)
        A = A / np.linalg.norm(A, axis = 0)
    elif sensing_matrix_choice == 'hadamard':
        A = np.array(hadamard(num_variables))[:num_measurements, :] / np.sqrt(num_measurements)
    else:
        raise ValueError('Invalid sensing matrix method')

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