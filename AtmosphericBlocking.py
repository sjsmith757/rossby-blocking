import numpy as np
from numpy import pi, cos, sin, exp
import scipy as sp
import logging

class Model(object):
    """  A minimalistic model for Atmospheric blocking """

    def __init__(self,
                nx=128,
                Lx=28000e3,
                dt=0.1*86400,
                tmax=100,
                alpha = 0.55,
                D=3.26e5,
                tau=10*86400,
                logfile="model.out"):

        self.nx = nx
        self.Lx = Lx

        self.dt = dt
        self.tmax = tmax
        self.t = 0.
        self.tc = 0

        self.tau = tau
        self.D = D

        self.alpha = alpha

        self.logfile = logfile

        # initializations
        self._initialize_logger()
        self._allocate_variables()
        self._initialize_grid()
        self._initialize_C()
        self._initialize_rk3w()

        #self._initialize_diagnostics()

    def run(self):
        """Run the model forward until the end."""
        while(self.t < self.tmax):
            self._step_forward()

        return


    #
    # private methods
    #

    def _step_forward(self):

        # status
        self._print_status()

        # time step
        #self._step_euler()
        self._step_rk3w()

        self.tc += 1
        self.t += self.dt

    def _step_euler(self):
        self._update_S()
        self.rhs = self.calc_rhs()
        self.Ah += self.rhs*self.dt




    def _initialize_rk3w(self):

        """ This pre-computes coefficients to a low storage implicit-explicit
            Runge Kutta time stepper.

            See Spalart, Moser, and Rogers. Spectral methods for the navier-stokes
                equations with one infinite and two periodic directions. Journal of
                Computational Physics, 96(2):297 - 324, 1991. """

        self.a1, self.a2, self.a3 = 29./96., -3./40., 1./6.
        self.b1, self.b2, self.b3 = 37./160., 5./24., 1./6.
        self.c1, self.c2, self.c3 = 8./15., 5./12., 3./4.
        self.d1, self.d2 = -17./60., -5./12.

        self.Linop = -(1./self.tau + self.D*self.k2)*self.dt

        self.L1 = ( (1. + self.a1*self.Linop)/(1. - self.b1*self.Linop) )
        self.L2 = ( (1. + self.a2*self.Linop)/(1. - self.b2*self.Linop) )
        self.L3 = ( (1. + self.a2*self.Linop)/(1. - self.b3*self.Linop) )

    def _step_rk3w(self):

        self._update_S()
        self.nl1h = self.calc_nonlin() + self.Sh
        self.Ah0 = self.Ah.copy()
        self.Ah = (self.L1*self.Ah0 + self.c1*self.dt*self.nl1h).copy()

        self.nl2h = self.nl1h.copy()
        self._update_S()
        self.nl1h = self.calc_nonlin() +  self.Sh
        self.qh = (self.L2*self.Ah0 + self.c2*self.dt*self.nl1h +\
                self.d1*self.dt*self.nl2h).copy()

        self.nl2h = self.nl1h.copy()
        self.Ah0 = self.Ah.copy()
        self._update_S()
        self.nl1h = self.calc_nonlin() +  self.Sh
        self.Ah = (self.L3*self.Ah0 + self.c3*self.dt*self.nl1h +\
                self.d2*self.dt*self.nl2h).copy()

    def calc_rhs(self):

        self.NonLin = self.calc_nonlin()

        # linear terms
        self.Lin = -(1./self.tau + self.D*self.k2)*self.Ah

        return self.Lin + self.NonLin + self.Sh

    def calc_nonlin(self):
        self.A = np.fft.irfft(self.Ah)
        return -1j*self.k*np.fft.rfft( (self.C-self.alpha*self.A)*self.A )

    def _initialize_C(self):
        self.A0 = 10*(1-np.cos(4*np.pi*self.x/self.Lx))
        self.C = 60 - 2*self.alpha*self.A0

    def _update_S(self):
        self.xc, self.Rx = 16800e3, 2800e3
        self.tc, self.Rt = 12.3*86400, 3.5*86400
        self.S = 1.852e-5*( 1 + 2*np.exp( - ((self.x-self.xc)/self.Rx)**2 - ((self.t-self.tc)/self.Rt )**2 ) )
        self.Sh = np.fft.rfft(self.S)

    def _allocate_variables(self):
        """ Allocate variables in memory """

        self.dtype_real = np.dtype('float64')
        self.dtype_cplx = np.dtype('complex128')
        self.shape_real = self.nx
        self.shape_cplx = self.nx//2+1

        # Wave activity
        self.A  = np.zeros(self.nx)
        self.Ah = np.fft.rfft(self.A)

    # logger
    def _initialize_logger(self):

        self.logger = logging.getLogger(__name__)


        if self.logfile:
            fhandler = logging.FileHandler(filename=self.logfile, mode='w')
        else:
            fhandler = logging.StreamHandler()

        formatter = logging.Formatter('%(levelname)s: %(message)s')

        fhandler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(fhandler)

        self.logger.setLevel(10)

        # this prevents the logger to propagate into the ipython notebook log
        self.logger.propagate = False

        self.logger.info(' Logger initialized')


    def _initialize_grid(self):
        """ Initialize lattice and spectral space grid """

        # physical space grids (the lattice)
        self.dx = self.Lx/(self.nx)

        self.x = np.linspace(0.,self.Lx-self.dx,self.nx)

        # wavenumber grids
        self.dk = 2.*pi/self.Lx
        self.nk = self.nx//2+1
        self.k = self.dk*np.arange(0.,self.nk)

        self.ik = 1j*self.k

        self.k2 = self.k**2


    def _print_status(self):
        """Output some basic stats."""
        # if (self.loglevel) and ((self.tc % self.printcadence)==0):
        #     self._calc_var()
        #     self.logger.info('Step: %4i, Time: %3.2e, Variance: %3.2e'
        #             , self.tc,self.t,self.var)
        pass

    ## utility methods
    def spec_var(self, ph):
        """ compute variance of p from Fourier coefficients ph """
        var_dens = 2. * np.abs(ph)**2 / self.nx**2
        return var_dens.sum()
