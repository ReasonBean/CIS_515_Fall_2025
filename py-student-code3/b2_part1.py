import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

def get_similitude_matrix(scale, theta_deg, tx, ty):
    """
    Construct a 2D similitude matrix

    A similitude matrix in SIM(2) is an affine transformation consisting of 
    a uniform magnification, a rotation by a specific angle, and a translation vector
    In other words, from the professor's note, SIM(2) encodes all transformations that preserve shape 
    but change size (magnification), orientation, and position.

    Mathmatically, a similitude is represented as
    A=[[a*R(theta) | t],
        [theta | 1]], where
        R(theta) = [[cos(theta), -sin(theta)],
                    [sin(theta), cos(theta)]], and
        t = (tx, ty)^T

    A point (x, y) in 2D
    Thus, (x', y') a*R(theta)(x,y)+t.
    This means that magnification by a, rotation by theta, translation by (tx, ty)

    Return will be SIM(2) matrix.
    SIM(2) matrix A is 
    [[a*cos(theta), -a*sin(theta), tx],
     [a*sin(theta), a*cos(theta), ty],
     [0, 0, 1]]
    """
    theta=np.radians(theta_deg)
    c, s=np.cos(theta), np.sin(theta) # cos(theta), sin(theta)
    A = np.eye(3) # 3x3 matrix

    # Magnification and rotation
    A[0, 0], A[0, 1] = scale* c, -scale* s
    A[1, 0], A[1, 1] = scale* s,  scale* c

    # Translation
    A[0, 2], A[1,2]= tx, ty
    return A

def similitude_log(A):
    """
    Compute the logarithm of a SIM(2) matrix and return its corresponding element in sim(2)
    A= [[a*cos(theta), -a*sin(theta), tx],
         [a*sin(theta), a*cos(theta), ty],
         [0, 0, 1]]
    
    The logarithm(A) lies in the sim(2), consisting of matrices
    B= [[lambda, -theta, u],
        [theta, lambda, v],
        [theta, theta, theta]], where lambda is natural log (a),
    and (u, v) encodes the infinitesimal translation after converting through the matrix V defined in B1.
    V=Omega^{-1}*(e^{Omega}-I), where Omega= [[lambda, -theta],
                                                [theta, lambda]]
    """
    # Extract magnification a from the 2×2 rotational block
    alpha = np.sqrt(A[0,0]**2 + A[0,1]**2)
    
    # lambda = ln(a)
    lam = np.log(alpha)
    
    # Extract rotation angle theta
    theta = np.arctan2(A[1, 0], A[0, 0])
    
    # Form Omega
    Omega = np.array([[lam, -theta],
                      [theta, lam]])
    
    # Extract translation vector W
    W = np.array([A[0, 2], A[1, 2]])
    
    # Compute V using the expression from Problem B1 part e
    if abs(lam) < 1e-10 and abs(theta) < 1e-10:
        # Case lambda = rotation angle = 0, so V = I
        V = np.eye(2)
    else:
        # V
        exp_Omega = np.array([[alpha * np.cos(theta), -alpha * np.sin(theta)],
                              [alpha * np.sin(theta),  alpha * np.cos(theta)]])
        V = np.linalg.solve(Omega, exp_Omega - np.eye(2))
    
    # Compute for U from W=VU
    U = np.linalg.solve(V, W)
    
    # Assemble the sim(2) matrix B.
    B = np.zeros((3, 3))
    B[0, 0], B[0, 1], B[0, 2] = lam, -theta, U[0]
    B[1, 0], B[1, 1], B[1, 2] = theta, lam, U[1]
    
    return B

def transform_shape(shape, A):
    """
    Apply a SIM(2) matrix to a 2D shape.

    A 2D point (x, y) is first lifted into coordinates as (x, y, 1).
    And then, (x', y', 1)^T=A*(x, y, 1)^T, where A is SIM(2) matrix

    Therefore, the transformed point is (x', y')=a*R(theta)(x, y)+t.

    This function performs the transformation on an entire set of points. 
    """
    
    ones = np.ones((shape.shape[0], 1)) # Convert points (x, y, 1) form
    points_h = np.hstack([shape, ones]) 

    # Apply the transformation A*(x, y, 1)^T
    transformed_h = (A @ points_h.T).T 

    return transformed_h[:, :2] # Return should be the (x',y') part

# Setup the base shape
# We choose a 2D pentagon shape.
base_shape = np.array([[0, 0], [2, 0], [2, 2], [1, 3], [0, 2], [0, 0]])
base_shape = base_shape - np.mean(base_shape, axis=0)

# Define two endpoint similitudes
start_params = (1.0, 0, 2, 2) # a, rotation degree, translation (2, 2)
end_params   = (0.5, 90, 8, 8) # a, rotation degree, translation (8, 8)

A_start = get_similitude_matrix(*start_params)
A_end   = get_similitude_matrix(*end_params)

# Compute log (SIM(2) -> sim(2))
L_start = similitude_log(A_start)
L_end   = similitude_log(A_end)

#check
print("Start: ", L_start)
print("\nEnd: ", L_end)

# Linear interpolation in sim(2) and visualization
plt.figure(figsize=(10, 8))
plt.title("B2 Part (1): Linear interpolation between two deformations")
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.5)

steps = 20
cmap = plt.cm.viridis

for i, t in enumerate(np.linspace(0, 1, steps)):
    # Start interpolation in sim(2)
    L_t = (1 - t) * L_start + t * L_end
    
    # Exponential map back to SIM(2)
    A_t = expm(L_t)
    
    # Apply the interpolated transformation to the shape
    deformed_shape = transform_shape(base_shape, A_t)
    
    # Plot
    if i == 0 or i == steps - 1:
        plt.plot(deformed_shape[:, 0], deformed_shape[:, 1], 'k-', 
                linewidth=2.5, marker='o', markersize=4,
                label="Keyframe" if i == 0 else "")
    else:
        color = cmap(t)
        plt.plot(deformed_shape[:, 0], deformed_shape[:, 1], 
                color=color, alpha=0.6, linewidth=1)

plt.legend()
plt.savefig("output/b2_part1.png")
plt.show()