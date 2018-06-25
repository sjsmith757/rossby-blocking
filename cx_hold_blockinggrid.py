import sys,os
import AtmosphericBlocking
import numpy as np
import h5py
import glob
import diagnostic as dg


def gaussforce(x,t,peak=2,inject=True,tw=2.5,xw=2800.0e3,xc=16800.0e3,tc=277.8):
  # Gaussian centered at 277.8 days and 16,800 km
    tc = tc
    tw = tw
    t = t/86400.0
    xc = xc
    xw = xw
    sx = 1.852e-5 + np.zeros(len(x))
    if inject:
        sx *= (1+peak*np.exp(-((x-xc)/xw)**2 - ((t-tc)/tw)**2))
    return sx


# In[4]:


def force_all(x,t,peak=2,inject=True,tw=2.5,xw=2800.0e3,xc=16800.0e3,tc=277.8):
  # Gaussian centered at 277.8 days
    tc = tc
    tw = tw
    t = t/86400.0
    xc = xc
    xw = xw
    sx = 1.852e-5 + np.zeros(len(x))
    if inject:
        sx *= (1+peak*np.exp(- ((t-tc)/tw)**2))
    return sx


# In[5]:


def noboru_cx(x,Lx,alpha):
  # The background conditions used in Noboru's paper
    A0 = 10*(1-np.cos(4*np.pi*x/Lx))
    cx = 60 - 2*alpha*A0
    return cx,A0


# In[6]:


def noiseforce(x,t,peak=2,inject=True,freqs=np.arange(10),speeds=np.arange(10),
               phases=np.zeros(10),ampls=np.ones(10),Lx=28000.0e3,
               tw=2.5,xw=2800.0e3,xc=16800.0e3,tc=277.8):
    if t/86400<270:
        return np.zeros(x.shape)+1.852e-5
    sx = np.zeros(x.shape)
    wampls = ampls*peak
    for i in range(0,len(freqs)):
        sx += 1.0/len(freqs)*wampls[i]*np.sin(2*np.pi*freqs[i]*x/Lx+speeds[i]*t+phases[i])
    sx = 1.852e-5*np.maximum(1,(1 + sx**3))
    return sx


# In[7]:


class conditions:
    def __init__(self,peak=2,inject=True,Y=10,beta=60,n=2,alpha=0.55,tau=10.0,sfunc=None,
                 xc=16800.0e3,xw=2800.0e3,tw=2.5,tc=277.8,noisy=False):
        self.peak = peak
        self.inject=inject
        self.Y = Y
        self.sfunc=sfunc
        self.tw = tw
        self.tc = tc
        self.xc = xc
        self.xw = xw
        self.noisy=noisy
        if not sfunc:
            self.sfunc=gaussforce
        self.tau = tau*86400.0
        self.beta = beta
        self.n=n
        self.alpha = alpha
    def forcing(self,x,t,peak=None,inject=None):
        if peak:
            self.peak = peak
        if inject:
            self.inject = inject
        sx = self.sfunc(x,t,peak=self.peak,inject=self.inject,tw=self.tw,xc=self.xc,
                        xw=self.xw,tc=self.tc)
        return sx
    def getcx(self,x,Lx,alpha=None,time=None):
        if alpha:
            self.alpha = alpha
        A0 = self.Y*(1-np.cos(2*self.n*np.pi*x/Lx))
        cx = self.beta - 2*self.alpha*A0
        return cx,A0


# In[8]:


class noisyconditions:
    def __init__(self,peak=2,Y=10,beta=60,n=2,background=True,forcing=True,
                 nwforce=26,nwcx=21,maxforcex=20,maxA0x=10,forcedecay=20,A0decay=40,alpha=0.55,
                 tc=277.8,tw=2.5,xc=16800.0e3,xw=2800.0e3,sfunc=None,
                 cfunc=None,inject=True,cxpeak=0.5,tau=10.0):
        self.peak = peak
        self.cxpeak = cxpeak
        self.inject=inject
        self.Y = Y
        self.sfunc=sfunc
        self.tw = tw
        self.tc = tc
        self.xc = xc
        self.xw = xw
        self.background=background
        self.forcingbool=forcing
        self.cfunc=cfunc
        self.tau = tau*86400.0
        if not sfunc and not forcing:
            print(forcing,sfunc)
            self.sfunc=gaussforce
        elif not sfunc and forcing:
            self.sfunc = noiseforce
        self.beta = beta
        self.n=n
        self.alpha = alpha
        if forcing:
            self.ffreqs = np.random.randint(1,maxforcex,size=nwforce)
            self.fspeeds = 2.0*np.pi/(forcedecay*86400.0) -\
                            4*np.pi/(forcedecay*86400.0)*np.random.rand(nwforce)  
            self.fphases = np.random.rand(nwforce)*2*np.pi
            self.fampls = 3.7*np.random.rand(nwforce) #6.8
        if background:
            self.cfreqs = np.random.randint(1,maxA0x,size=nwcx)
            self.cspeeds = 2.0*np.pi/(A0decay*86400.0) - 4*np.pi/(A0decay*86400.0)*np.random.rand(nwcx)  
            self.cphases = np.random.rand(nwcx)*2*np.pi
            self.campls = np.random.rand(nwcx)
        
        
    def forcing(self,x,t,peak=None,inject=None):
        if peak:
            self.peak = peak
        if inject:
            self.inject = inject
        if not self.forcingbool:
            sx = self.sfunc(x,t,peak=self.peak,inject=self.inject,tw=self.tw,xc=self.xc,
                            xw=self.xw,tc=self.tc)
        else:
            sx = self.sfunc(x,t,peak=self.peak,freqs=self.ffreqs,speeds=self.fspeeds,
                            phases=self.fphases,ampls=self.fampls)
        return sx
    def getcx(self,x,Lx,alpha=None,time=None):
        if alpha:
            self.alpha = alpha
        A0 = self.Y*(1-np.cos(2*self.n*np.pi*x/Lx))
        if self.background:
            A0 *= self.cfunc(x,Lx,t=time,freqs=self.cfreqs,speeds=self.cspeeds,
                             phases=self.cphases,ampls=self.cxpeak*self.campls)
        cx = self.beta - 2*self.alpha*A0
        
        return cx,A0


# In[9]:


def noisybackground(x,Lx,t=None,freqs=None,speeds=None,phases=None,ampls=None):
    dcx = np.zeros(len(x))
    for i in range(0,len(freqs)):
        dcx += 1.0/len(freqs)*ampls[i]*np.sin(2*np.pi*freqs[i]*x/Lx+speeds[i]*t+phases[i])
    return (1+dcx)



def blocking_iteration(cxpeak=1.0,fpeak=1.8,beta=60,Y=10,noisy_initc=None):
    
    if not noisy_initc:
        noisy_initc = noisyconditions(cfunc=noisybackground,cxpeak=cxpeak,peak=fpeak,
                                      background=True,forcing=True,beta=beta,Y=Y)
    else:
        noisy_initc.beta=beta
        noisy_initc.Y=Y
        noisy_initc.cxpeak=cxpeak
        noisy_initc.peak=fpeak


    cleanup = True
    if cleanup:
        os.system("rm -rf output/")


    cond = noisy_initc
    wavenum=2

    model = AtmosphericBlocking.Model(nx=1024,Lx = 28000e3,dt=.005*86400,alpha=cond.alpha,
                                            tmax=3.5*86400,D=3.26e5,tau=cond.tau,
                                            sfunc=cond.forcing,cfunc=cond.getcx,
                                            forcingpeak=cond.peak,injection=cond.inject,
                                            save_to_disk=True,
                                            overwrite=True,
                                            tsave_snapshots=50,
                                            path = 'output/')

    model.tmax = 450*86400
    model.run()

    setup = h5py.File("output/setup.h5")
    x = setup['grid/x'][:]

    fnis = np.array(sorted(glob.glob("output/snapshots/*.h5")))

    Ahat, F, S, C = 0,0,0,0
    t = []
    for fni in fnis[0::2]:
        snap = h5py.File(fni)
        t.append(snap['t'][()])
        try:
            Ahat = np.vstack([Ahat, snap['A'][:]])
            F = np.vstack([F, snap['F'][:]])
            S = np.vstack([S, snap['S'][:]])
            C = np.vstack([C, snap['C'][:]])
        except:
            Ahat = snap['A'][:]
            F = snap['F'][:]
            S = snap['S'][:]
            C = snap['C'][:]

    t = np.array(t)

    it0 = np.where(t/86400 > 260)[0][0]
        
    ua = 40-cond.alpha*(Ahat+model.A0[np.newaxis,:])

    gamma = 4*cond.alpha*F/C**2
    
    masked = gamma.copy()
    masked[np.where(gamma<0.99999)]=0
    masked[np.where(gamma>=0.99999)]=1

    threshu = 5
    ugrad = -(np.gradient(ua[it0:],axis=1))
    
    umask = ((ugrad - np.mean(ugrad))/np.std(ugrad) > threshu)*1.0
    
    uct,ucmask = dg.count_blocks(umask,80-2*wavenum,10)
    uts = t[it0:][np.where(ucmask>0.5)[0]]/86400.0
    uxs = x[np.where(ucmask>0.5)[1]]/1.0e3
    
    shock_size = np.sum(umask)
    
    smask = (S[it0:]>0.6e-4)*1.0

    sct,scmask = dg.count_blocks(smask,5,5)
    
    sts = t[it0:][np.where(scmask>0.5)[0]]/86400
    sxs = x[np.where(scmask>0.5)[1]]/1e3
    
    return cond,uct,shock_size,uxs,uts,sct,sts,sxs

if __name__=="__main__":
    
    name = sys.argv[1]
    conditionfile = sys.argv[2]
    beta = float(sys.argv[3])
    
    #beta = 60
    a0y = 10
    
    cxpeak=1.0
    fpeak=1.7
    
    nwforce = 26
    nwcx = 21
    
    ffreqs = np.zeros((16,nwforce))
    fspeeds = np.zeros((16,nwforce))
    fphases = np.zeros((16,nwforce))
    fampls = np.zeros((16,nwforce))
    
    cfreqs = np.zeros((16,nwcx))
    cspeeds = np.zeros((16,nwcx))
    cphases = np.zeros((16,nwcx))
    campls = np.zeros((16,nwcx))
    
    nblockseq = np.zeros(16)
    blockszseq = np.zeros(16)
    
    avgnblocks = 0.0
    avgblocksize = 0.0
    
    avgnblocks_per_event = 0.0
    avgsize_per_event = 0.0
    
    nblockseq_perevent = np.zeros(16)
    blockszseq_perevent = np.zeros(16)
    
    onsets = []
    events = []
    
    initc = np.load(conditionfile).item()
    
    #initc = noisyconditions(cfunc=noisybackground,cxpeak=cxpeak,peak=fpeak,
                                      #background=True,forcing=True,beta=beta,Y=a0y)
    
    n=0
    for a0y in np.logspace(0,np.log10(20.0),num=16):
        conditions,nblocks,blocksize,uxs,uts,sct,sts,sxs = blocking_iteration(cxpeak=cxpeak,
                                                                              fpeak=fpeak,
                                                                              beta=beta,
                                                                              Y=a0y,
                                                                              noisy_initc=initc)
        ffreqs[n,:]  = conditions.ffreqs
        fspeeds[n,:] = conditions.fspeeds
        fphases[n,:] = conditions.fphases
        fampls[n,:]  = conditions.fampls
        
        cfreqs[n,:]  = conditions.cfreqs
        cspeeds[n,:] = conditions.cspeeds
        cphases[n,:] = conditions.cphases
        campls[n,:]  = conditions.campls
        
        nblockseq[n] = nblocks
        blockszseq[n] = blocksize
        
        nblockseq_perevent[n] = nblocks/sct
        blockszseq_perevent[n] = blocksize/sct
        
        onsets.append([uxs,uts])
        events.append([sxs,sts])
        
        n+=1
        
    forcing_waves = {"freqs":ffreqs,
                     "speeds":fspeeds,
                     "phases":fphases,
                     "amplitudes":fampls}
    bg_waves = {"freqs":cfreqs,
                "speeds":cspeeds,
                "phases":cphases,
                "amplitudes":campls}
    output = {"forcing_init":forcing_waves,
              "c(x)_init":bg_waves,
              "raw_nblocks":nblockseq,
              "raw_blocksize":blockszseq,
              "block_coords":onsets,
              "a0y":np.logspace(0,np.log10(20.0),num=16),
              "cx_peak":cxpeak,
              "forcing_peak":fpeak,
              "forcing_coords":events,
              "raw_nblocks_perevent":nblockseq_perevent,
              "raw_blocksize_perevent":blockszseq_perevent}
    
    np.save("../"+name+".npy",output,fix_imports=True)
    
    #Cleanup
    os.system("rm -rf output/")