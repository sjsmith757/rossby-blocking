import sys,os
import AtmosphericBlocking
import numpy as np
import h5py
import glob
import diagnostic as dg

def gaussforce(x,t,peak=2,inject=True,tw=2.5,xc=16800.0e3,xw=2800.0e3):
  # Gaussian centered at 277.8 days and 16,800 km
    tc = 277.8
    tw = 2.5
    t = t/86400.0
    xc = 16800.0e3
    xw = 2800.0e3
    sx = 1.852e-5 + np.zeros(len(x))
    if inject:
        sx *= (1+peak*np.exp(-((x-xc)/xw)**2 - ((t-tc)/tw)**2))
    return sx

class conditions:
    def __init__(self,peak=2,inject=True,Y=10,beta=60,n=2,alpha=0.55,tau=10.0,sfunc=None,
                 tw=2.5,xc=16800.0e3,xw=2800.0e3):
        self.peak = peak
        self.inject=inject
        self.Y = Y
        self.sfunc=sfunc
        if not sfunc:
            self.sfunc=gaussforce
        self.tau = tau*86400.0
        self.beta = beta
        self.n=n
        self.alpha = alpha
        self.tw = tw
        self.xc = xc
        self.xw = xw
    def forcing(self,x,t,peak=None,inject=None):
        if peak:
            self.peak = peak
        if inject:
            self.inject = inject
        sx = self.sfunc(x,t,peak=self.peak,inject=self.inject,tw=self.tw,xc=self.xc,
                        xw=self.xw)
        return sx
    def getcx(self,x,Lx,alpha=None,t=None):
        if alpha:
            self.alpha = alpha
        A0 = self.Y*(1-np.cos(2*self.n*np.pi*x/Lx))
        cx = self.beta - 2*self.alpha*A0
        return cx,A0
    
if __name__=="__main__":
    n = int(sys.argv[1])
    Y = float(sys.argv[2])
    name = sys.argv[3]
    
    
    Lx = 28000.0e3
    
    ensemble = []
    
    for xc in np.linspace(0.25*Lx/n,0.75*Lx/n,num=10):
    
        initc = conditions(sfunc=gaussforce,n=n,xc=xc,Y=Y)
        os.system("rm -rf output/")
        model = AtmosphericBlocking.Model(nx=2048,Lx = 28000e3,dt=.001*86400,alpha=initc.alpha,
                                                tmax=3.5*86400,D=3.26e5,tau=initc.tau,
                                                sfunc=initc.forcing,cfunc=initc.getcx,
                                                forcingpeak=initc.peak,injection=initc.inject,
                                                save_to_disk=True,
                                                overwrite=True,
                                                tsave_snapshots=50,
                                                path = 'output/')
        model.tmax = 350*86400
        model.run()
        
        setup = h5py.File("output/setup.h5")
        x = setup['grid/x'][:]
        
        fnis = np.array(sorted(glob.glob("output/snapshots/*.h5")))
        
        Ahat, F = 0,0
        t = []
        for fni in fnis[0::2]:
            snap = h5py.File(fni)
            t.append(snap['t'][()])
            try:
                Ahat = np.vstack([Ahat, snap['A'][:]])
                F = np.vstack([F, snap['F'][:]])
            except:
                Ahat = snap['A'][:]
                F = snap['F'][:]

        t = np.array(t)
        
        it0 = np.where(t/86400 > 260)[0][0]
        
        grad = (np.gradient(F[it0:],axis=1))
        
        fmask = ((grad-np.mean(grad))/np.std(grad) > 5)*1.0
        
        ict,dmask = dg.count_blocks(fmask,10,10)
        
        ts = t[it0:][np.where(dmask>0.5)[0]]/86400
        xs = x[np.where(dmask>0.5)[1]]/1e3
        
        nsig = np.sum(fmask)
        
        result = {"onset":(xs,ts),
                  "nblocks":ict}
        
        ensemble.append(result)
        
    np.save("../"+name+".npy",ensemble)
    
    #Clean up
    os.system("rm -rf output/")
        