import cvxpy as cp
import numpy as np
from scipy.special import gamma

def norm_2_I(x, m, d):
    x_i_all = x.reshape((m, d))
    x_i_all_norm = cp.norm2(x_i_all, axis=1)
    x_norm_2_I = cp.sum(x_i_all_norm)
    return x_norm_2_I

def norm_2_I_i(x, m, d):
    x_i_all = x.reshape((m, d))
    x_i_norm_all = cp.norm2(x_i_all, axis=1)
    return x_i_norm_all


def l2l1_algorithm(A, y, N, d, sigma):
    m = N // d

    # Define the problem variables
    x = cp.Variable(A.shape[1])
    t = cp.Variable(m, nonneg=True)

    M = A.shape[0]
    mu = np.sqrt(2) * gamma((M + 1) / 2) / gamma(M / 2)
    var = M - mu**2
    eps = (mu + 3 * np.sqrt(var)) * sigma

    # Define the problem constraints
    x_i_norm_all = norm_2_I_i(x, m, d)
    constraints = [cp.norm2(y - A @ x) <= eps]
    for i in range(m):
        M_i = np.zeros((d, N))
        for j in range(d):
            M_i[j, d*i+j] = 1
        # constraints += [cp.SOC(t[i], M_i @ x)]
        constraints += [cp.SOC(t[i], x.reshape((m, d))[i, :])]
        constraints += [t[i] >= 0]

    # Define the problem objective
    objective = cp.Minimize(cp.sum(t))

    # Define the problem and solve it
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)


    pass

    # Return the optimal solution
    return x.value
