import numpy as np


def fit_polynormial(x, y, niter=10, deg=2, kth=0.5):
    p = 10
    N = x.shape[0]
    Kth = int(N * kth)
    inx = range(0, N, 1)
    x_inx = np.random.choice(inx, p, replace=False)

    for i in range(0, niter):
        poly_coef = np.polyfit(x[x_inx], y[x_inx], deg=deg)
        y_ = np.polyval(poly_coef, x)
        r2 = (y - y_) ** 2
        r2s_inx = sorted(range(len(r2)), key=lambda k: r2[k])
        x_inx = r2s_inx[Kth-p:Kth]

    y_ = np.polyval(poly_coef, x)
    return poly_coef, y_
