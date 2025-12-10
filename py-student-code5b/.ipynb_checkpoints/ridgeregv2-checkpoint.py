import numpy as np

def ridgeregv2(X: np.ndarray, y: np.ndarray, K: float) -> tuple[np.ndarray, float, float, np.ndarray, float]:
    """
    Ridge regression minimizing w and b
    b is penalized
    X is an m x n matrix, y a m x 1 colum vector
    weight vector w, intercept b
    Solution in terms of the primal variables
    And also in terms of the dual variable alph
    """
    # Basic setup
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]

    m,n=X.shape

    # Augmentation
    ones_col = np.ones((m,1))
    X_tilde=np.hstack([X, ones_col]) # m X n+1

    # dual alpha
    A=X_tilde@X_tilde.T+K*np.eye(m)
    alpha=np.linalg.solve(A,y)

    # Calculate w and b
    w_aug=X_tilde.T@alpha
    w=w_aug[:n]
    b=w_aug[n].item()

    # loss xi
    xi=K*alpha

    # Norm, not b
    nw=np.linalg.norm(w)
    nxi=np.linalg.norm(xi)
    
    #raise Exception("Implement me")
    #return np.zeros(0), 0, 0, np.zeros(0), 0
    return w, nw, b, xi, nxi
