import numpy as np

def makeline(ww: np.ndarray, bb: float, ll: np.ndarray, mm: np.ndarray, nw: float) -> np.ndarray:
    """
    Function to make the line segment
    of the intersection of the line
    of equation  (x,y). ww = b with
    the box defined by y = ll(2) (lower side), x = mm(1) (right side),
    y = mm(2) (upper side), x = ll(1) (left side)
    nw = ||w||


    Test where the line of equation  (x,y). w = b
    intersects the box defined by y = ll(2) (lower side), x = mm(1) (right side),
    y = mm(2) (upper side), x = ll(1) (left side)

    Normalizes equation of the line with ||w|| = 1.
    """
    w = ww/nw; b = bb/nw
    w = w[:, 0] if len(w.shape) > 1 else w
    # print(f"Normalized w[0] = {w[0]}")
    # print(f"Normalized w[1] = {w[1]}")
    # print(f"Normalized b = {b}")
    if abs(w[1]) < 1e-11: # vertical line
        p1 = np.array([b/w[0], ll[1]]); p2 = np.array([b/w[0], mm[1]])
    elif abs(w[0]) < 1e-11: # horizontal line
        p1 = np.array([ll[0], b/w[1]]); p2 = np.array([mm[0], b/w[1]])
    else:
        xll2 = (b - w[1]*ll[1])/w[0]; ymm1 = (b - w[0]*mm[0])/w[1]
        xmm2 = (b - w[1]*mm[1])/w[0]; yll1 = (b - w[0]*ll[0])/w[1]
        if ll[0] <= xll2 and xll2 <= mm[0]: # Intersects lower side of box
            if ll[1] <= ymm1 and ymm1 <= mm[1]: # intersects right side of box
                p1 = np.array([xll2, ll[1]]); p2 = np.array([mm[0], ymm1])
            elif ll[0] <= xmm2 and xmm2 <= mm[0]: #intersects upper side of box
                p1 = np.array([xll2, ll[1]]); p2 = np.array([xmm2, mm[1]])
            else: # intersects left side of box
                p1 = np.array([xll2, ll[1]]); p2 = np.array([ll[0], yll1])
        else: # Does not intersect lower side of box
            if ll[1] <= ymm1 and ymm1 <= mm[1]: # intersects right side of box
                if ll[0] <= xmm2 and xmm2 <= mm[0]: # intersects upper side of box
                    p1 = np.array([mm[0], ymm1]); p2 = np.array([xmm2, mm[1]])
                else: # not lower side, right side, not upper side; intersects left side of box
                    p1 = np.array([mm[0], ymm1]); p2 = np.array([ll[0], yll1])
            else: # not lower side, not right side; intersects the line x = ll[0] (left line) and the line y = mm[1] (upper line)
                p1 = np.array([ll[0], yll1]); p2 = np.array([xmm2, mm[1]])
    
    l = np.block([[p1.T], [p2.T]]).T
    return l