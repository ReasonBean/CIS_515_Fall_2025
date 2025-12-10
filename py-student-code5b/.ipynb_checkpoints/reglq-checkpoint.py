import numpy as np

def reglq(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray, float]:
    """
    Regression minimizing w and b
    X is an m x n matrix, y a m x 1 colum vector
    weight vector w, intercept b
    Computes the least squares solution using the pseudo inverse
    """
    # Basic setup
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]

    m,n=X.shape

    ones_col=np.ones((m, 1))
    X_aug =np.hstack([X, ones_col]) # m X n+1

    # Pseudo inverse, pinv
    X_aug_pinv=np.linalg.pinv(X_aug)
    w_aug=X_aug_pinv@y

    # Extract param
    w=w_aug[:n]
    b=w_aug[n].item()

    # xi
    xi = y - (X_aug @ w_aug)

    # norm
    nw=np.linalg.norm(w)
    nxi=np.linalg.norm(xi)
    
    #raise Exception("Implement me")
    
    #return (np.zeros(0), 0, 0 , np.zeros(0), 0)
    return w, nw, b, xi, nxi
