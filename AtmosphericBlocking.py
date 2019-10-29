import numpy as np
from numpy import pi, cos, sin, exp
import scipy as sp
import logging

from Saving import *

class Model(object):
    """  A minimalistic model for Atmospheric blocking """

    def __init__(self,
                nx=128,
                Lx=28000e3,
                dt=0.1*86400,
                tmax=100,
                tmin=0,
                alpha = 0.55,
                D=3.26e5,
                tau=10*86400,
                injection=True,
                forcingpeak = 2,
                sfunc = None,
                cfunc = None,
                logfile="model.out",
                printcadence=1000,
                loglevel=0,
                save_to_disk=True,
                overwrite=True,
                tsave_snapshots=50,
                beta = 60,
                verbose  = False,
                path = 'output/'):

        self.nx = nx
        self.Lx = Lx

        self.dt = dt
        self.tmax = tmax
        self.tmin = tmin
        self.t = 0.
        self.tc = 0
        
        self.printcadence = printcadence
        self.loglevel = loglevel

        self.tau = tau
        self.D = D
        self.Smax = forcingpeak

        self.inject = injection

        self.alpha = alpha
        self.beta  = beta
        self.verbose = verbose

        self.logfile = logfile

        self.sfunc = sfunc
        self.cfunc = cfunc

        self.save_to_disk = save_to_disk
        self.overwrite = overwrite
        self.tsnaps = tsave_snapshots
        self.path = path

        # initializations
        self._initialize_logger()
        self._allocate_variables()
        self._initialize_grid()
        self._initialize_C()
        self._initialize_etdrk4()
        #self._initialize_rk3w()
        initialize_save_snapshots(self,self.path)
        save_setup(self)
        save_parameters(self,fields=['nx','Lx','dt','tmax','tmin','printcadence','loglevel','tau',\
                                    'D','Smax','inject','alpha','beta','verbose','save_to_disk','overwrite',\
                                    'tsnaps','path'])

        #self._initialize_diagnostics()

    def run(self):
        """Run the model forward until the end."""
        while(self.t < self.tmax):
            self._step_forward()

            if self.save_to_disk:
                save_snapshots(self,fields=['t','A','F','S','C', 'beta'])
        return

    #
    # private methods
    #
       
    def run_some_years(self):
        """Run the model forward for 1 year."""
        self.t = self.tmin
        while(self.t < self.tmax):
            self._step_forward()
            
            if self.save_to_disk:
                save_snapshots(self,fields=['t','A','F','S','C', 'beta'])
                
        if (self.verbose==True):
            print (str(self.t/86400.0)+" days since start")
        return

    

    def _step_forward(self):

        # status
        self._print_status()

        # time step
        #self._step_euler()   # Simple-minded Euler
        self._step_etdrk4()   # Exponential time differencing RK4
        #self._step_rk3w()

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



    def _initialize_etdrk4(self):

        """ This performs pre-computations for the Expotential Time Differencing
            Fourth Order Runge Kutta time stepper. The linear part is calculated
            exactly.

            See Cox and Matthews, J. Comp. Physics., 176(2):430-455, 2002.
                Kassam and Trefethen, IAM J. Sci. Comput., 26(4):1214-233, 2005. """

        # the exponent for the linear part
        self.c = - (self.D*self.k2 + 1./self.tau)

        ch = self.c*self.dt
        self.expch = np.exp(ch)
        self.expch_h = np.exp(ch/2.)
        self.expch2 = np.exp(2.*ch)

        M = 32.  # number of points for line integral in the complex plane
        rho = 1.  # radius for complex integration
        r = rho*np.exp(2j*np.pi*((np.arange(1.,M+1))/M))# roots for integral

        #l1,l2 = self.ch.shape
        #LR = np.repeat(ch,M).reshape(l1,l2,M) + np.repeat(r,l1*l2).reshape(M,l1,l2).T
        LR = ch[...,np.newaxis] + r[np.newaxis,...]
        LR2 = LR*LR
        LR3 = LR2*LR

        self.Qh   =  self.dt*(((np.exp(LR/2.)-1.)/LR).mean(axis=1));
        self.f0  =  self.dt*( ( ( -4. - LR + ( np.exp(LR)*( 4. - 3.*LR + LR2 ) ) )/ LR3 ).mean(axis=1) )
        self.fab =  self.dt*( ( ( 2. + LR + np.exp(LR)*( -2. + LR ) )/ LR3 ).mean(axis=1) )
        self.fc  =  self.dt*( ( ( -4. -3.*LR - LR2 + np.exp(LR)*(4.-LR) )/ LR3 ).mean(axis=1) )

    def _step_etdrk4(self):

        self._update_S()
        self._update_C()

        self.Ah0 = self.Ah.copy()

        Fn0 = self.calc_nonlin() + self.Sh
        self.Ah = (self.expch_h*self.Ah0 + Fn0*self.Qh)
        self.Ah1 = self.Ah.copy()

        Fna = self.calc_nonlin() + self.Sh
        self.Ah = (self.expch_h*self.Ah0 + Fna*self.Qh)

        Fnb = self.calc_nonlin() + self.Sh
        self.Ah = (self.expch_h*self.Ah1 + ( 2.*Fnb - Fn0 )*self.Qh)

        Fnc =  self.calc_nonlin() + self.Sh

        self.Ah = (self.expch*self.Ah0 + Fn0*self.f0 +  2.*(Fna+Fnb)*self.fab\
                  + Fnc*self.fc)


    def calc_rhs(self):

        self.NonLin = self.calc_nonlin()

        # linear terms
        self.Lin = -(1./self.tau + self.D*self.k2)*self.Ah

        return self.Lin + self.NonLin + self.Sh

    def calc_nonlin(self):
        self.A = np.fft.irfft(self.Ah)
        self.F = (self.C-self.alpha*self.A)*self.A
        return -1j*self.k*np.fft.rfft( self.F )

    def _initialize_C(self):
        if self.cfunc:
            self.C,self.A0 = self.cfunc(self.x,self.Lx,self.alpha,time=self.t)
        else:
            self.A0 = 10*(1-np.cos(4*np.pi*self.x/self.Lx))
            self.C  = 60 - 2*self.alpha*self.A0

    def _update_C(self):
        if self.cfunc:
            self.C,self.A0 = self.cfunc(self.x,self.Lx,alpha=self.alpha,time=self.t)
        else:
            self.A0 = 10*(1-np.cos(4*np.pi*self.x/self.Lx))
            self.C = 60 - 2*self.alpha*self.A0
            
    def _update_S(self):
        if self.sfunc:
            self.S = self.sfunc(self.x,self.t,inject=self.inject,peak=self.Smax)
        else:
            self.xc, self.Rx = 16800e3, 2800e3
            self.t_c, self.Rt = 12.3*86400, 3.5*86400
            self.S = np.zeros(len(self.x)) + 1.852e-5
            if self.inject:
                self.S *= ( 1 + self.Smax*np.exp( - ((self.x-self.xc)/self.Rx)**2 - ((self.t-self.t_c)/self.Rt )**2 ) )
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
        if (self.loglevel) and ((self.tc % self.printcadence)==0):
            self.logger.info('Step: %4i, Time: %3.2e'
                    , self.tc,self.t)
        pass

    ## utility methods
    def spec_var(self, ph):
        """ compute variance of p from Fourier coefficients ph """
        var_dens = 2. * np.abs(ph)**2 / self.nx**2
        return var_dens.sum()
