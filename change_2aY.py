#!/usr/bin/env python
# coding: utf-8

# In[1]:


import AtmosphericBlocking
import numpy as np
import h5py
import sys,os
import glob
import matplotlib.pyplot as plt
import logging
import Saving as saving

log_directory='logs/change_a2Y'
logging.basicConfig(filename=log_directory+'.log',level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')



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




def noboru_cx(x,Lx,alpha):
  # The background conditions used in Noboru's paper
    A0 = 10*(1-np.cos(4*np.pi*x/Lx))
    cx = 60 - 2*alpha*A0
    return cx,A0


# In[2]:


def noiseforce(x,t,peak=2,inject=True,freqs=np.arange(10),speeds=np.arange(10),
               phases=np.zeros(10),ampls=np.ones(10),Lx=28000.0e3,tw=2.5,
               xw=2800.0e3,xc=16800.0e3,tc=277.8):
    if t/86400<270:
        return np.zeros(x.shape)+1.852e-5
    sx = np.zeros(x.shape)
    wampls = ampls*peak
    for i in range(0,len(freqs)):
        sx += 1.0/len(freqs)*wampls[i]*np.sin(2*np.pi*freqs[i]*x/Lx+speeds[i]*t+phases[i])
    sx = 1.852e-5*np.maximum(1,(1 + sx**3))
    return sx


# In[3]:


def noisybackground(x,Lx,t=None,freqs=None,speeds=None,phases=None,ampls=None):
    dcx = np.zeros(len(x))
    for i in range(0,len(freqs)):
        dcx += 1.0/len(freqs)*ampls[i]*np.sin(2*np.pi*freqs[i]*x/Lx+speeds[i]*t+phases[i])
    return (1+dcx)



# In[4]:


class conditions:
    def __init__(self,peak=2,inject=True,Y=10,beta=60,n=2,
                 alpha=0.55,tau=10.0,sfunc=None,xc=16800.0e3,
                 xw=2800.0e3,tw=2.5,tc=277.8,noisy=False):
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
        sx = self.sfunc(x,t,peak=self.peak,inject=self.inject,
                        tw=self.tw,xc=self.xc,
                        xw=self.xw,tc=self.tc)
        return sx
    def getcx(self,x,Lx,alpha=None,time=None):
        if alpha:
            self.alpha = alpha
        A0 = self.Y*(1-np.cos(2*self.n*np.pi*x/Lx))
        cx = self.beta - 2*self.alpha*A0
        return cx,A0
    


# In[5]:


class noisyconditions:
    def __init__(self,peak=2,Y=10,beta=60,n=2,background=True,
                 forcing=True,nwforce=26,nwcx=21,maxforcex=20,
                 maxA0x=10,forcedecay=20,A0decay=40,alpha=0.55,
                 tc=277.8,tw=2.5,xc=16800.0e3,xw=2800.0e3,
                 sfunc=None,cfunc=None,inject=True,
                 cxpeak=0.5,tau=10.0,save_to_disk=False,overwrite=True, path='output/'):
        self.peak = peak
        self.cxpeak = cxpeak
        self.inject=inject
        self.Y  = Y
        self.sfunc=sfunc
        self.tw = tw
        self.tc = tc
        self.xc = xc
        self.xw = xw
        self.background=background
        self.forcingbool=forcing
        self.cfunc=cfunc
        self.tau = tau*86400.0
        self.nwforce = nwforce
        self.nwcx    = nwcx
        self.maxforcex = maxforcex
        self.maxA0x    = maxA0x
        self.forcedecay = forcedecay
        self.A0decay   = A0decay
        self.path      =path
        self.save_to_disk = save_to_disk
        self.overwrite = overwrite
                        
        if not sfunc and not forcing:
            print(forcing,sfunc)
            self.sfunc=gaussforce
        elif not sfunc and forcing:
            self.sfunc = noiseforce
        self.beta = beta
        self.n=n
        self.alpha = alpha
        if forcing:
            self.ffreqs = np.random.randint(1,self.maxforcex,size=self.nwforce)
            self.fspeeds = 2.0*np.pi/(self.forcedecay*86400.0) - 4*np.pi/(forcedecay*86400.0)*  np.random.rand(self.nwforce)  
            self.fphases = np.random.rand(self.nwforce)*2*np.pi
            self.fampls = 3.7*np.random.rand(self.nwforce) #6.8
        if background:
            self.cfreqs = np.random.randint(1,self.maxA0x,size=self.nwcx)
            self.cspeeds = 2.0*np.pi/(self.A0decay*86400.0) -  4*np.pi/(self.A0decay*86400.0)*np.random.rand(self.nwcx)  
            self.cphases = np.random.rand(self.nwcx)*2*np.pi
            self.campls = np.random.rand(self.nwcx)
       
    
        ### This is not working for now!
#         saving.initialize_save_snapshots(self,self.path)
#         saving.save_parameters(self,fields=['peak','n','cxpeak','inject','xc','xw', 'tc','tw','background',\
#                                     'forcingbool','tau','inject','alpha','beta','path', 'nwforce',\
#                                            'nwcx', 'maxforcex', 'maxA0x', 'forcedecay', 'A0decay' ])
        ### This is not working for now!
        
        
    def forcing(self,x,t,peak=None,inject=None):
        if peak:
            self.peak = peak
        if inject:
            self.inject = inject
        if not self.forcingbool:
            sx = self.sfunc(x,t,peak=self.peak,inject=self.inject,
                            tw=self.tw,xc=self.xc,
                            xw=self.xw,tc=self.tc)
        else:
            sx = self.sfunc(x,t,peak=self.peak,freqs=self.ffreqs,
                            speeds=self.fspeeds,phases=self.fphases,
                            ampls=self.fampls)
        return sx
    def getcx(self,x,Lx,alpha=None,time=None):
        if alpha:
            self.alpha = alpha
        A0 = self.Y*(1-np.cos(2*self.n*np.pi*x/Lx))
        if self.background:
            A0 *= self.cfunc(x,Lx,t=time,freqs=self.cfreqs,
                             speeds=self.cspeeds,
                             phases=self.cphases,
                             ampls=self.cxpeak*self.campls)
        cx = self.beta - 2*self.alpha*A0
        
        return cx,A0
    


# #### Intitalizing the model #######




if __name__=="__main__":
    
    a2Y_init  = float(sys.argv[1]) #90
    a2Y_final = float(sys.argv[2]) #20
    decades   = float(sys.argv[3]) #20

    noisy_initc = noisyconditions(cfunc=noisybackground, cxpeak=0.5,Y=10, nwcx=21, n=2, peak=3, 
                                  nwforce=26, background=True,forcing=True,beta=40, alpha=0.55, 
                                  path = 'noise_params/', save_to_disk=True)

    cond = noisy_initc
    

    model = AtmosphericBlocking.Model(nx=1024,Lx = 28000e3,dt=.005*86400,alpha=cond.alpha,
                                                tmax=3.5*86400,D=3.26e5,tau=cond.tau,
                                                sfunc=cond.forcing,cfunc=cond.getcx,
                                                forcingpeak=cond.peak,injection=cond.inject, beta=cond.beta,
                                                save_to_disk=True,
                                                overwrite=True,
                                                tsave_snapshots=50,
                                                verbose=False,
                                                path = 'output_slow_change_a2Y_new/')


    model.verbose=False
    model.save_to_disk = True
    model.tsave_snapshots = 200
    model.tmin = 0*1*86400#(year*360)*86400
    model.tmax = (360)*1*86400 #((year+1)*360)*86400  
    cond.Y     = a2Y_init/(2.0*cond.alpha)
    model.Y    = cond.Y
    model.run_some_years()
    logging.debug(" day = %i / a2Y = %1.2f/ Y=%1.2f"%(model.t/86400, model.Y*(2.0*model.alpha), model.Y))


    logging.debug(" Initialization done ")

    model.verbose=False
    model.save_to_disk = True
    model.tsave_snapshots = 50

    ## How to slowly change Uref ##

    years   = decades*10
    DAYS    = years*360
    for day in range(360,360+int(DAYS)):
        r = (a2Y_init-a2Y_final)/(DAYS-1.0)
        model.tmin = day*1*86400
        model.tmax = (day+1)*1*86400 
        cond.Y     = (a2Y_init - r*(day-360))/(2*cond.alpha)
        # beta changes everyday 0.00097 m/s per day, 0.35 m/s year, 7 m/s per decade, 70 m/s per 20 decades
        model.Y    = cond.Y
        model.run_some_years() 
        if (day%360) == 0:
            logging.debug(" day = %i / year = %i / a2Y = %1.2f / Y=%1.2f / r=%1.6f"%(day, int((day)/360.0), model.Y*(2.0*model.alpha), model.Y, r))

