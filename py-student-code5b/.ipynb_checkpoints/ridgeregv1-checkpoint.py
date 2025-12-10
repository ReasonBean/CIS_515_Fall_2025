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

    # Basic setup
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]

    m, n=X.shape

    # Centering
    y_mean = np.mean(y) # the mean of y
    X_mean = np.mean(X, axis=0, keepdims=True)

    # Centered data
    X_centered = X-X_mean
    y_centered = y-y_mean

    # Caculate w (fomula *6)
    A=X_centered@X_centered.T+K*np.eye(m)
    temp=np.linalg.solve(A, y_centered) # I didn't hardcoding for computing the inverse matrix
    w=X_centered.T@temp

    # Calculate bias
    b=y_mean-(X_mean@w).item() # to scalar

    xi=y-(X@w+b)

    nw=np.linalg.norm(w)
    nxi=np.linalg.norm(xi)
    
    #raise Exception("Implement me")
    #return np.zeros(0), 0, 0, np.zeros(0), 0
    return w, nw, b, xi, nxi
