import numpy as np

def check_structure(vector):
    length = vector.shape[0]
    half_length = length // 2

    for i in range(half_length):
        if vector[i] != 0 and vector[i + half_length] == 0:
            return False
        elif vector[i] == 0 and vector[i + half_length] != 0:
            return False

    return True