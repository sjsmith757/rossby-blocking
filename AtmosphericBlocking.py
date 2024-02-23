from __future__ import annotations
import numpy as np
from typing import Protocol, Optional, Literal, Tuple, List, TYPE_CHECKING, Union
from Saving import IOInterface
from functools import wraps

if TYPE_CHECKING:
    try:
        import xarray as xr
    except ImportError:
        pass


class SFuncType(Protocol):
    def __call__(  # noqa: E704
        self, x: np.ndarray, t: float, inject: bool = ..., peak: float = ...
    ) -> np.ndarray: ...


class CFuncType(Protocol):
    def __call__(  # noqa: E704
        self, x: np.ndarray, Lx: float, alpha: float, time: float = ...
    ) -> Tuple[np.ndarray, np.ndarray]: ...


class Model(IOInterface):
    """A minimalistic model for Atmospheric blocking"""

    def __init__(
        self,
        nx: int = 128,
        Lx: float = 28000e3,
        dt: float = 0.1 * 86400,
        tmax: float = 100.0,
        tmin: float = 0.0,
        alpha: float = 0.55,
        D: float = 3.26e5,
        tau: float = 10 * 86400.0,
        injection: bool = True,
        forcingpeak: float = 2.0,
        sfunc: Optional[SFuncType] = None,
        cfunc: Optional[CFuncType] = None,
        beta: float = 60.0,
        logfile: Optional[str] = "model.out",
        printcadence: int = 1000,
        loglevel: Union[int, str] = "WARNING",
        save_to_disk: bool = True,
        overwrite: bool = True,
        tsave_snapshots: int = 50,
        tsave_start: float = 0.0,
        verbose: Optional[bool] = None,
        path: str = "output/",
        io_backend: Literal["h5", "xr"] = "h5",
    ):
        r"""

        A minimal model for solving the 1D wave activity equation

        The equation the model solves

        .. math::

        \frac{\partial}{\partial t}\hat{A}(x,t) = -\frac{\partial}{\partial x}\left[\left(C(x) - \alpha\hat{A}\right)\hat{A}\right] + \hat{S} - \frac{\hat{A}}{\tau}+ D\frac{\partial^2\hat{A}}{\partial x^2}


        Parameters
        ----------
        nx : int, optional
            grid size, by default 128
        Lx : float, optional
            the physical distance spanned by the grid, by default 28000e3
        dt : float, optional
            the model integration timestep, in s, by default 0.1*86400
        tmax : float, optional
            how long to run the model, in s, by default 100.0
        tmin : float, optional
            the time to begin integration, by default 0.0
        alpha : float, optional
            the alpha parameter, or the regression slope between zonal wind and wave
            activity, by default 0.55
        D : float, optional
            the D parameter, corresponding to the diffusivity of wave activity, by
            default 3.26e5
        tau : float, optional
            the damping timescale for wave activity, in s, by default 10*86400.0
        injection : bool, optional
            whether to inject a temporally and spatially localized forcing, by default
            True
        forcingpeak : float, optional
            how large the injected forcing should be, by default 2.0
        sfunc : Callable, optional
            the functional form of the stochastic forcing. The default uses values from
            Nakamura and Huang 2018. :func:`sfunc` should accept the following parameters:

                * x (:py:class:`np.ndarray`): the model grid
                * t (float): the model time step
                * inject (bool): whether to inject a one-time forcing
                * peak (float): the scaling factor for the forcing

            :func:`sfunc` should return a :py:class:`np.ndarray[nx]`.

            The default :func:`sfunc` used is

        .. math::

        \hat{S} = \hat{S_0} \left\{ 1+\hat{S}_{\text{max}}\exp \left[ -\left(\frac{x-x_c}{x_w}\right)^2 -\left(\frac{t-t_c}{t_w}\right)^2 \right] \right\}

        cfunc : Callable, optional
            the functional form of the wave group velocity. The default uses values from
            Nakamura and Huang 2018. :func:`cfunc` should accept the following parameters:

                * x (:py:class:`np.ndarray`): the model grid
                * Lx (float): the model length scale
                * alpha (float): the model alpha
                * time (float): the model time step

            :func:`cfunc` should return two :py:class:`np.ndarray[nx]`, corresponding to
            the group velocity (C) and the stationary wave amplitude (A0), respectively.

            The default function used is

        .. math::

        C(x) = \beta - 2\alpha A_0(x)

            where

        .. math::

        A_0(x) = 10.0*\left[1-\cos\left(\frac{4\pi x}{L_x}\right)\right]

        beta : float, optional
            the beta to be used by the model if using the default :func:`cfunc`,
            by default 60.0.

        IOInterface Parameters
        ----------------------
        The following parameters determine how the model logs and outputs its data.

        logfile : str, optional
            the path to the log file to create, by default "model.out". Setting the
            logfile to none will output the log information to sys.stdout
        printcadence : int, optional
            the frequency to log the model's status, by default 1000
        loglevel : int or ["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   optional
            the logging level to utilize, by default "INFO"
        save_to_disk : bool, optional
            whether to save the results of a model run, by default True
        overwrite : bool, optional
            whether to overwrite previous cases of the same name, by default True
        tsave_snapshots : int, optional
            how often to save snapshots of the model's internal state, by default 50
        tsave_start : float, optional
            when to begin saving snapshots, by default 0.0
        verbose : bool, optional
            deprecated parameter. loglevel should be used instead, by default None
        path : str, optional
            the root directory for the model experiment, by default "output/"
        io_backend : ["h5", "xr"], optional
            the backedn to use for writing data, by default "h5"
        """

        self.nx = nx
        self.Lx = Lx
        self.dt = dt
        self.tmax = tmax
        self.tmin = tmin
        self.t: float = 0.0
        self.tc: int = 0
        self.tau = tau
        self.D = D
        self.Smax = forcingpeak
        self.inject = injection
        self.alpha = alpha
        self.beta = beta
        self.sfunc = sfunc
        self.cfunc = cfunc

        # initializations
        super().__init__(
            logfile=logfile,
            printcadence=printcadence,
            loglevel=loglevel,
            save_to_disk=save_to_disk,
            overwrite=overwrite,
            tsave_snapshots=tsave_snapshots,
            tsave_start=tsave_start,
            verbose=verbose,
            path=path,
            io_backend=io_backend,
        )
        self._allocate_variables()
        self._initialize_grid()
        self._initialize_C()
        self._initialize_etdrk4()
        # self._initialize_rk3w()

        self.save_setup(
            fields=["grid/nx", "grid/x", "grid/k"], dtypes=["int", "float", "float"]
        )
        self.save_parameters(
            fields=[
                "nx",
                "Lx",
                "dt",
                "tmax",
                "tmin",
                "tau",
                "D",
                "Smax",
                "inject",
                "alpha",
                "beta",
            ],
        )

    def run(self) -> None:
        """Run the model forward until the end."""
        while self.t < self.tmax:
            self._step_forward()

            if self.save_to_disk:
                self.save_snapshots(fields=["t", "A", "F", "S", "C", "beta"])

    #
    # private methods
    #

    def run_some_years(self) -> None:
        """Run the model forward for 1 year."""
        self.t = self.tmin
        while self.t < self.tmax:
            self._step_forward()

            if self.save_to_disk:
                self.save_snapshots(fields=["t", "A", "F", "S", "C", "beta"])

        if self._verbose:
            print(str(self.t / 86400.0) + " days since start")

    def reset(self) -> None:
        """
        Reset the model's internal time and wave activity so that it can be run again
        from the initial conditions. Otherwise, the model will always continue from its
        previous state.
        """
        self.t = 0.0
        self._allocate_variables()

    def _step_forward(self) -> None:
        # status
        self._print_status()

        # time step
        # self._step_euler()   # Simple-minded Euler
        self._step_etdrk4()  # Exponential time differencing RK4
        # self._step_rk3w()

        self.tc += 1
        self.t += self.dt

    def _step_euler(self) -> None:
        self._update_S()
        self.rhs = self.calc_rhs()
        self.Ah += self.rhs * self.dt

    def _initialize_rk3w(self) -> None:
        """This pre-computes coefficients to a low storage implicit-explicit
        Runge Kutta time stepper.
        See Spalart, Moser, and Rogers. Spectral methods for the navier-stokes
            equations with one infinite and two periodic directions. Journal of
            Computational Physics, 96(2):297 - 324, 1991."""

        self.a1, self.a2, self.a3 = 29.0 / 96.0, -3.0 / 40.0, 1.0 / 6.0
        self.b1, self.b2, self.b3 = 37.0 / 160.0, 5.0 / 24.0, 1.0 / 6.0
        self.c1, self.c2, self.c3 = 8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0
        self.d1, self.d2 = -17.0 / 60.0, -5.0 / 12.0

        self.Linop = -(1.0 / self.tau + self.D * self.k2) * self.dt

        self.L1 = (1.0 + self.a1 * self.Linop) / (1.0 - self.b1 * self.Linop)
        self.L2 = (1.0 + self.a2 * self.Linop) / (1.0 - self.b2 * self.Linop)
        self.L3 = (1.0 + self.a2 * self.Linop) / (1.0 - self.b3 * self.Linop)

    def _step_rk3w(self) -> None:
        self._update_S()
        self.nl1h: np.ndarray = self.calc_nonlin() + self.Sh
        self.Ah0: np.ndarray = self.Ah.copy()
        self.Ah = (self.L1 * self.Ah0 + self.c1 * self.dt * self.nl1h).copy()

        self.nl2h = self.nl1h.copy()
        self._update_S()
        self.nl1h = self.calc_nonlin() + self.Sh
        self.qh = (
            self.L2 * self.Ah0
            + self.c2 * self.dt * self.nl1h
            + self.d1 * self.dt * self.nl2h
        ).copy()

        self.nl2h: np.ndarray = self.nl1h.copy()
        self.Ah0 = self.Ah.copy()
        self._update_S()
        self.nl1h = self.calc_nonlin() + self.Sh
        self.Ah = (
            self.L3 * self.Ah0
            + self.c3 * self.dt * self.nl1h
            + self.d2 * self.dt * self.nl2h
        ).copy()

    def _initialize_etdrk4(self) -> None:
        """This performs pre-computations for the Expotential Time Differencing
        Fourth Order Runge Kutta time stepper. The linear part is calculated
        exactly.

        See Cox and Matthews, J. Comp. Physics., 176(2):430-455, 2002.
            Kassam and Trefethen, IAM J. Sci. Comput., 26(4):1214-233, 2005."""

        # the exponent for the linear part
        self.c = -(self.D * self.k2 + 1.0 / self.tau)

        ch = self.c * self.dt
        self.expch = np.exp(ch)
        self.expch_h = np.exp(ch / 2.0)
        # self.expch2 = np.exp(2.0 * ch)

        M = 32.0  # number of points for line integral in the complex plane
        rho = 1.0  # radius for complex integration
        r = rho * np.exp(
            2j * np.pi * ((np.arange(1.0, M + 1)) / M)
        )  # roots for integral

        # l1,l2 = self.ch.shape
        # LR = np.repeat(ch,M).reshape(l1,l2,M) + np.repeat(r,l1*l2).reshape(M,l1,l2).T
        # Assume L is diagonal in Fourier space
        LR = ch[..., np.newaxis] + r[np.newaxis, ...]
        LR2 = LR * LR
        LR3 = LR2 * LR

        self.Qh: np.ndarray = self.dt * (((np.exp(LR / 2.0) - 1.0) / LR).mean(axis=1))
        self.f0: np.ndarray = self.dt * (
            ((-4.0 - LR + (np.exp(LR) * (4.0 - 3.0 * LR + LR2))) / LR3).mean(axis=1)
        )
        self.fab: np.ndarray = self.dt * (
            ((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR3).mean(axis=1)
        )
        self.fc: np.ndarray = self.dt * (
            ((-4.0 - 3.0 * LR - LR2 + np.exp(LR) * (4.0 - LR)) / LR3).mean(axis=1)
        )

    def _step_etdrk4(self) -> None:
        self._update_S()
        self._update_C()

        self.Ah0 = self.Ah.copy()

        Fn0 = self.calc_nonlin() + self.Sh
        self.Ah = self.expch_h * self.Ah0 + Fn0 * self.Qh
        self.Ah1 = self.Ah.copy()

        Fna = self.calc_nonlin() + self.Sh
        self.Ah = self.expch_h * self.Ah0 + Fna * self.Qh

        Fnb = self.calc_nonlin() + self.Sh
        self.Ah = self.expch_h * self.Ah1 + (2.0 * Fnb - Fn0) * self.Qh

        Fnc = self.calc_nonlin() + self.Sh

        self.Ah = (
            self.expch * self.Ah0
            + Fn0 * self.f0
            + 2.0 * (Fna + Fnb) * self.fab
            + Fnc * self.fc
        )

    def calc_rhs(self) -> np.ndarray:
        """Calculate the right hand side of the 1D wave activity equation"""
        NonLin = self.calc_nonlin()
        # linear terms
        Lin = -(1.0 / self.tau + self.D * self.k2) * self.Ah

        return Lin + NonLin + self.Sh

    def _update_F(self) -> None:
        self.A = np.fft.irfft(self.Ah)
        self.F = (self.C - self.alpha * self.A) * self.A

    def calc_nonlin(self) -> np.ndarray:
        """Calculate the nonlinear portion of the equation"""
        self._update_F()
        return -1j * self.k * np.fft.rfft(self.F)

    def _initialize_C(self) -> None:
        self.A0: np.ndarray
        self.C: np.ndarray
        if self.cfunc:
            self.C, self.A0 = self.cfunc(self.x, self.Lx, self.alpha, time=self.t)
        else:
            self.A0 = 10 * (1 - np.cos(4 * np.pi * self.x / self.Lx))
            self.C = self.beta - 2 * self.alpha * self.A0

    def _update_C(self) -> None:
        if self.cfunc:
            self.C, self.A0 = self.cfunc(self.x, self.Lx, alpha=self.alpha, time=self.t)
        else:
            self.A0 = 10 * (1 - np.cos(4 * np.pi * self.x / self.Lx))
            self.C = 60 - 2 * self.alpha * self.A0

    def _update_S(self) -> None:
        if self.sfunc:
            self.S = self.sfunc(self.x, self.t, inject=self.inject, peak=self.Smax)
        else:
            self.S = np.ones_like(self.x) * 1.852e-5
            if self.inject:
                xc, Rx = 16800e3, 2800e3
                t_c, Rt = 12.3 * 86400, 3.5 * 86400
                self.S *= 1 + self.Smax * np.exp(
                    -(((self.x - xc) / Rx) ** 2) - ((self.t - t_c) / Rt) ** 2
                )
        self.Sh: np.ndarray = np.fft.rfft(self.S)

    def _allocate_variables(self) -> None:
        """Allocate variables in memory"""

        self._dtype_real = np.dtype("float64")
        self._dtype_cplx = np.dtype("complex128")
        self._shape_real = self.nx
        self._shape_cplx = self.nx // 2 + 1

        # Wave activity
        self.A: np.ndarray = np.zeros(self.nx)
        self.Ah: np.ndarray = np.fft.rfft(self.A)

    def _initialize_grid(self) -> None:
        """Initialize lattice and spectral space grid"""

        # physical space grids (the lattice)
        self.dx = self.Lx / (self.nx)

        self.x = np.linspace(0.0, self.Lx - self.dx, self.nx)

        # wavenumber grids
        self.dk = 2.0 * np.pi / self.Lx
        self.nk = self.nx // 2 + 1
        self.k = self.dk * np.arange(0.0, self.nk)

        self.ik = 1j * self.k

        self.k2 = self.k**2

    @wraps(IOInterface.to_dataset)
    def to_dataset(
        self,
        coords: List[Tuple[str, np.ndarray]] = [],
        dvars: List[str] = ["beta", "A", "F", "S", "C", "alpha"],
        params: List[str] = ["inject"],
    ) -> xr.Dataset:
        if not coords:
            coords = [
                ("t", np.array([self.t])),
                ("x", self.x),
                ("k", self.k),
            ]
        dvars += ["nx", "Lx", "dt", "tmax", "tmin", "tau", "Smax", "D"]
        return IOInterface.to_dataset(self, coords=coords, dvars=dvars, params=params)

    # utility methods
    def spec_var(self, ph: np.ndarray):
        """compute variance of p from Fourier coefficients ph"""
        var_dens = 2.0 * np.abs(ph) ** 2 / self.nx**2
        return var_dens.sum()
