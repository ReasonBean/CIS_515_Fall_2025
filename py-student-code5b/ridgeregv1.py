import numpy as np

def ridgeregv1(X: np.ndarray, y: np.ndarray, K: float) -> tuple[np.ndarray, float, float, np.ndarray, float]:
    """
    Ridge regression with centered data
    b is not penalized
    X is an m x n matrix, y a m x 1 colum vector
    weight vector w, intercept b
    Solution in terms of the primal variables

    Warning: in Python, handling both n=1 and n>1 together takes some trickery - you may need to use np.newaxis to expand dims
    """
    raise Exception("Implement me")
    return np.zeros(0), 0, 0, np.zeros(0), 0
