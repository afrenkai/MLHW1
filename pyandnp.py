import numpy as np

def problem1 (A, B, C):

    return A @ B - C

def problem2 (A):
    np.ones_like(A)

def problem3 (A):
    return np.fill_diagonal(A, 0)

def problem4 (A, i):
    return np.sum(A[i, :])

def problem5 (A, c, d):
    # define bool mask to filter vals
    mask = (A > c) & (A <= d)
    return np.mean(A[mask]) if np.any(mask) else np.nan # not sure how to handle, will return nan for now., 

def problem6 (A, k):
    vals, vecs = np.linalg.eig(A)
    ind = np.argsort(np.abs(vals))[-k:]
    return vecs[:, ind]

def problem7 (A, x):
    return np.linalg.solve(A, x)

def problem8 (x, k):
    return np.tile(x[:, np.newaxis], k)

def problem9 (A):
    return A[np.random.permutation(A.shape[0]), :]

def problem10 (A):
    return np.mean(A, axis = 1)
def problem11 (n, k):
    A = np.random.randint(0, k+1, size = n)
    A[A % 2 == 0] = -1
    return A
def problem12 (A, b):
    return A + b[:, np.newaxis]
def problem13 (A):
    n, m, _ = A.shape
    return A.reshape(n, m * m).T
 
