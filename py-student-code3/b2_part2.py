import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.linalg import expm # Used as the exponential map: sim(2) -> SIM(2)
from interpatxy import interpatxy  # Cubic spline interpolation function from project 3

# Import the manual functions defined in part 1
from b2_part1 import get_similitude_matrix, similitude_log, transform_shape

def eval_cubic_bezier(t, P0, P1, P2, P3):
    """
    Re-evaluate a cubic spline at parameter t in [0,1] with control points.
    B(t)=(1-t)^3 *P0+ 3(1-t)^2 *t*P1+ 3(1-t)t^2 *P2 + t^3*P3
    """
    return (1-t)**3 * P0 + 3*(1-t)**2*t * P1 + 3*(1-t)*t**2 * P2 + t**3 * P3

# Setup the base shape
# We choose a 2D pentagon shape as in part 1.
base_shape = np.array([[0, 0], [2, 0], [2, 2], [1, 3], [0, 2], [0, 0]])
base_shape = base_shape - np.mean(base_shape, axis=0)

# Define deformation sequence
# We let the user provide arbitrary keyframes.
# User inputs (scale, angle_degree, tx, ty) for each keyframe, so the program can interpolate any number of similitudes A0 ... Am.
keyframes_params = []

print("=== Similitude motion interpolation Input ===")
try:
    num_str = input("Enter number of keyframes (m+1): ")
    if not num_str.strip():
        print("Error: Invalid input.")
        sys.exit(1)
        
    num_frames_input = int(num_str)
    
    if num_frames_input < 2:
        print("Error: Need at least 2 keyframes for interpolation.")
        sys.exit(1)

    print("Input format: scale angle_deg tx ty (e.g., 1.0 45 2 2)")
    for i in range(num_frames_input):
        while True:
            try:
                line = input(f"Enter keyframe #{i} (scale angle_deg tx ty): ")
                parts = list(map(float, line.strip().split()))
                if len(parts) != 4:
                    print("Error: Please enter exactly 4 numbers.")
                    continue
                keyframes_params.append(tuple(parts))
                break
            except ValueError:
                print("Error: Invalid input.")

except ValueError:
    print("Invalid input. Program terminated.")
    sys.exit(1)

# Build SIM(2) matrices A_i from the user parameters.
matrices = [get_similitude_matrix(*p) for p in keyframes_params]
N_frames = len(matrices)

print(f"\nProcessing {N_frames} keyframes...")

# Important note
# When extracting logarithm parameters from SIM(2) matrices, we face a critical issue with angle representation.

# For keyframes with angles like degree 0, 90, 180, 270, 360, we get [0, pi/2, pi, -pi/2, 0] <- Notice the jump from pi to -pi/2.
# This causes the spline interpolation to take a shortcut by rotating backwards instead of continuing smoothly forward.
# Thus, it happened abrupt direction changes and unnatural motion

# Troubleshoot
# 1. We use np.unwrap() to convert angles to a continuous representation. 
# The original degree becomes [0, pi/2, pi, 3pi/2, 2pi] <- Monotone increasing.

# 2. After unwrapping theta, we MUST recompute (u, v)
# Since in sim(2), the relationship is translation = V(lambda, theta)*(u, v), 
# where V depends on both lambda and theta. 
# If we unwrap theta but keep the old (u,v), the sim(2) exp(B) will produce wrong translations.
# So, first of all, we extracted a -> lambda = ln(a)
# and extract theta (wrapped) using arctan2
# and unwrap theta to get continuous angles
# and for each keyframe with (lambda, theta_unwrapped, t), we solved V(lambda, theta_unwrapped)*u =t =>  u = V^{-1}*t
# This ensures exp(B) correctly reconstructs the original SIM(2) matrix

print("Computing Logarithms and correcting u, v for unwrapped angles...")

# 1. Extract raw parameters first (scale lambda, wrapped theta, and translation t)
raw_alphas = []
raw_thetas = []
translations = []

for M in matrices:
    # Extract scale a from rotation scaling block, then lambda = ln(a)
    a = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
    raw_alphas.append(np.log(a)) # natural log
    
    # Extract theta (wrapped in -pi, pi)
    raw_thetas.append(np.arctan2(M[1, 0], M[0, 0]))
    
    # Extract translation t=(tx, ty)
    translations.append([M[0, 2], M[1, 2]])

alphas = np.array(raw_alphas)
thetas = np.array(raw_thetas)

# 2. Unwrap angles so theta_i varies continuously without 2pi jumps
thetas = np.unwrap(thetas)

# 3. Recompute (u, v) for each keyframe using the unwrapped thetas
# We must ensure t = V(theta_unwrapped)*u holds true.
us = []
vs = []

for i in range(N_frames):
    lam = alphas[i]
    th = thetas[i]
    tx, ty = translations[i]
    
    # Calculate V matrix for the specific (unwrapped) theta
    # V = Omega^-1*(e^Omega-I)
    if abs(lam) < 1e-10 and abs(th) < 1e-10:
        # When both lambda and theta are near zero, Omega ~~ 0 and V ~~ I
        V = np.eye(2)
    else:
        Omega = np.array([[lam, -th], [th, lam]])
        scale = np.exp(lam)
        # exp_Omega matches the rotation/scale part of the group matrix
        exp_Omega = np.array([[scale * np.cos(th), -scale * np.sin(th)],
                              [scale * np.sin(th),  scale * np.cos(th)]])
        
        # V = Omega^-1 * (exp_Omega - I) => solve Omega * V = exp(Omega)-I for V
        V = np.linalg.solve(Omega, exp_Omega - np.eye(2))
    
    # Solve t = V * u for u -> u = V^-1 * t
    t_vec = np.array([tx, ty])
    u_vec = np.linalg.solve(V, t_vec)
    
    us.append(u_vec[0])
    vs.append(u_vec[1])

us = np.array(us)
vs = np.array(vs)

# 4. Compute spline control points using interpatxy
print("Computing Spline Control Points.")
# We use interpatxy to get control points for the parameters in SIM(2)
B_alpha, B_theta = interpatxy(alphas, thetas)[2:]
B_u, B_v = interpatxy(us, vs)[2:]
plt.close('all') 

# 5. Animation Setup
# We referred matplotlib document(https://matplotlib.org/stable/users/explain/animations/animations.html), and
# a website (https://spatialthoughts.com/2022/01/14/animated-plots-with-matplotlib/) 
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("B2 Part (2): Motion interpolation (similitude spline)")
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)

# Calculate bounds dynamically based on input movement
# Increased margin to widen the grid view
# Dynamically choose axis limits based on keyframe centers (tx, ty)
all_centers = np.array([(p[2], p[3]) for p in keyframes_params])
margin = 10  # set 10 for an appropriate view
min_x, max_x = np.min(all_centers[:, 0]) - margin, np.max(all_centers[:, 0]) + margin
min_y, max_y = np.min(all_centers[:, 1]) - margin, np.max(all_centers[:, 1]) + margin
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)

# Draw each keyframe shape as a dashed outline, A_0, A_1, ..., A_m
for i, M in enumerate(matrices):
    deformed = transform_shape(base_shape, M)
    draw_s = np.vstack([deformed, deformed[0]])
    ax.plot(draw_s[:, 0], draw_s[:, 1], 'k--', linewidth=1, alpha=0.3)
    ax.text(deformed[0, 0], deformed[0, 1], f"A{i}", fontsize=10, alpha=0.5)

# Dynamic elements (moving shape)
# We also add a trace line
line, = ax.plot([], [], 'r-', linewidth=3, label='Moving Body')
path_x, path_y = [], []
trace, = ax.plot([], [], 'b:', linewidth=1, alpha=0.5, label='Trace') 

# Animation settings (Default)
fps = 30 # Lower FPS to reduce file size
duration_sec = 5
total_frames = fps * duration_sec
segments = N_frames - 1

def update(frame):
    """
    For FuncAnimation
    """
    # Reset path trace at the start of each loop to avoid connecting end to start
    if frame==0:
        path_x.clear()
        path_y.clear()
    # Global normalized time in [0, 1]
    global_t = frame / (total_frames - 1)
    
    # Map global time (global_t) to segment index (i) and local time (t)
    u = global_t * segments
    i = int(u)
    if i >= segments:
        i = segments - 1
        t = 1
    else:
        t = u - i
        
    # Spline Evaluation (in sim(2))
    a_t = eval_cubic_bezier(t, *B_alpha[i])
    w_t = eval_cubic_bezier(t, *B_theta[i])
    u_t = eval_cubic_bezier(t, *B_u[i])
    v_t = eval_cubic_bezier(t, *B_v[i])
    

    # Build the element L_t in sim(2)
    # And then, apply the matrix exponential to obtain A(t) in SIM(2)
    L_t = np.zeros((3, 3))
    L_t[0, 0], L_t[0,1], L_t[0,2] = a_t, -w_t,u_t 
    L_t[1,0], L_t[1,1], L_t[1,2] = w_t,  a_t, v_t
    
    # sim(2) -> SIM(2)
    A_t = expm(L_t)
    
    # Shape transformation using imported helper
    deformed = transform_shape(base_shape, A_t)
    draw_s = np.vstack([deformed, deformed[0]])
    
    # update Plot
    line.set_data(draw_s[:, 0], draw_s[:,1])
    
    # Update Trace
    center = np.mean(deformed, axis=0)
    path_x.append(center[0])
    path_y.append(center[1])
    trace.set_data(path_x, path_y)
    
    return line, trace

# 6. Create animation and Save
anim = FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=True)
save_path = "output/b2_part2.gif"
print(f"Saving animation to {save_path}")
anim.save(save_path, writer=PillowWriter(fps=fps))

plt.legend()
plt.show()