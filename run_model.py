import numpy as np
import matplotlib.pyplot as plt

import AtmosphericBlocking


def noboru_cx(x, Lx, alpha, time=None):
    # The background conditions used in Noboru's paper
    A0 = 10 * (1 - np.cos(4 * np.pi * x / Lx))
    cx = 60 - 2 * alpha * A0
    return cx, A0


def gaussforce(
    x: np.ndarray,
    t: float,
    peak=2,
    inject: bool = True,
    tw: float = 2.5,
    xw: float = 2800.0e3,
    xc: float = 16800.0e3,
    tc: float = 277.8,
) -> np.ndarray:
    # Gaussian centered at 277.8 days and 16,800 km
    ti = t / 86400.0
    sx = 1.852e-5 * np.ones_like(x)
    if inject:
        sx *= 1 + peak * np.exp(-(((x - xc) / xw) ** 2) - ((ti - tc) / tw) ** 2)
    return sx


model = AtmosphericBlocking.Model(
    nx=2048,
    Lx=28000e3,
    dt=0.005 * 86400,
    alpha=0.55,
    tmax=3.5 * 86400,
    D=3.26e5,
    tau=10 * 86400,
    logfile=None,
    # cfunc=noboru_cx,
    # sfunc=gaussforce,
    # path="nc_output/",
    # io_backend="xr",
)

tmaxes = np.linspace(1, 45, 75) * 86400


A = 0
S = 0
for tmax in tmaxes:
    model.tmax = tmax
    model.run()

    try:
        A = np.vstack([A, model.A[None]])
        S = np.vstack([S, model.S[None]])
    except ValueError:
        A = model.A
        S = model.S

lons = model.x / 1e3 * 360 / 28000.0 + 100.0
lons[np.where(lons > 180)] -= 360.0

plt.figure(figsize=(4.0, 8))
plt.contourf(
    model.x / 1e3, tmaxes / 86400, A + model.A0[None, :], np.linspace(0, 50, 21)
)
plt.colorbar()
plt.contour(model.x / 1e3, tmaxes / 86400, S, np.linspace(3e-5, 6e-5, 5), colors="0.5")
plt.xlabel(r"Distance [km]")
plt.ylabel(r"Day")
plt.title("A")
plt.ylim(5, 45)
plt.xlim(13500, model.Lx / 1e3)
plt.show()

#
# plt.figure(figsize=(5,8))
# plt.contourf(lons,tmaxes/86400,S)
# plt.xlabel(r"Longitude")
# plt.ylabel(r"Day")
# plt.title("S")
#
# plt.ylim(1,27)
# plt.xlim(-100,100)
