import numpy as np

def reorder_columns(matrix):
    """
    Reorder the columns of matrix like this: 1st column, n//2-th column, 2nd column, n//2+1-th column, ..., n//2-th column, n-th column
    """
    n_cols = matrix.shape[1]
    reordered_cols = []
    for i in range(n_cols // 2):
        reordered_cols.append(matrix[:, i])
        reordered_cols.append(matrix[:, n_cols // 2 + i])
    if n_cols % 2 != 0:
        reordered_cols.append(matrix[:, n_cols // 2])
    return np.column_stack(reordered_cols)