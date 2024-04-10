import numpy as np

def stack_odd_even(x):
    """
    Stack the odd-indexed elements followed by the even-indexed elements of an array.
    """
    even_elements = x[::2]
    odd_elements = x[1::2]
    return np.hstack((even_elements, odd_elements))