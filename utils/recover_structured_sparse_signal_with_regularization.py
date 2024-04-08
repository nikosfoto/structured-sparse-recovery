import numpy as np
from scipy.optimize import minimize


def objective_function(x, CS_mat, y, alpha):
    return np.linalg.norm(y - np.dot(CS_mat, x))**2 + alpha * np.linalg.norm(x, 1)

# Function to recover the structured sparse signal using CS with l1 norm regularization
def recover_structured_sparse_signal_with_regularization(CS_mat, y, alpha):
    n = CS_mat.shape[1]
    x0 = np.zeros(n)  # Initial guess for optimization

    # Define support constraints to enforce symmetry
    def support_constraints(x):
        x1 = x[:n//2]
        x2 = x[n//2:]
        return np.linalg.norm(x1 - x2)

    bounds = [(None, None)] * n  # No constraints on entries
    constraints = [{'type': 'eq', 'fun': support_constraints}]  # Symmetry constraint
    res = minimize(objective_function, x0, args=(CS_mat, y, alpha), bounds=bounds, constraints=constraints)
    return res.x