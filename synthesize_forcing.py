import numpy as np
import matplotlib.pyplot as plt

# Block Shape: (time, xwidth, twidth, phase, (xcoord,tcoord)) (the last only for onsets)

if __name__=="__main__":
    nblocks = np.zeros((5,10,5,20))
    starts = np.zeros((5,10,5,20,2),dtype=object)
    avgdelays = np.zeros((5,10,5,20))
    stddelays = np.zeros((5,10,5,20))
    phase = np.zeros(20)
    
    foundphase = False
    
    t1 = 0.5
    t2 = 10.0
    x1 = 50.0
    x2 = 5000.0
    p1 = 0.5
    p2 = 10.0
    
    peaks = np.logspace(np.log10(p1),np.log10(p2),num=5)
    xwidth = np.logspace(np.log10(x1),np.log10(x2),num=10)
    twidth = np.logspace(np.log10(t1),np.log10(t2),num=5)
    
    nt=0
    for t in twidth:
        nx=0
        for x in xwidth:
            ip=0
            for p in peaks:
                name = "block%02.1f_%04.1f_%02.1f.npy"%(t,x,p)
                try:
                    output = np.load(name)
                except:
                    print(name)
                    raise
                for nphase in range(0,20):
                    nblocks[nt,nx,ip,nphase] = output[nphase]["nblocks"]
                    starts[nt,nx,ip,nphase,0] = output[nphase]["onset"][0]
                    starts[nt,nx,ip,nphase,1] = output[nphase]["onset"][1]
                    dels = np.array(output[nphase]["delay"][1])
                    if len(dels)>0:
                        avgdelays[nt,nx,ip,nphase] = np.mean(dels)
                        stddelays[nt,nx,ip,nphase] = np.std(dels)
                    else:
                        avgdelays[nt,nx,ip,nphase] = np.nan
                        stddelays[nt,nx,ip,nphase] = np.nan
                    if not foundphase:
                        phase[nphase] = output[nphase]["forcing phase"]
                foundphase=True
                ip+=1
            nx+=1
        print("Finished Time %d of %d"%(nt+1,len(twidth)))
        nt+=1
     
    nblkstats = np.zeros((5,10,5,2))
    ndelstats = np.zeros((5,10,5,2))
    
    nblkstats[:,:,:,0] = np.mean(nblocks,axis=3)
    nblkstats[:,:,:,1] =  np.std(nblocks,axis=3)
    
    ndelstats[:,:,:,0] = np.nanmean(avgdelays,axis=3)
    ndelstats[:,:,:,1] =  np.sqrt(np.nansum((stddelays*avgdelays)**2,axis=3))
    
    output = {"raw blocks":nblocks,
              "onset coords":starts,
              "onset delays":(avgdelays,stddelays),
              "block stats":nblkstats,
              "delay stats":ndelstats,
              "forcing peak":peaks,
              "forcing xwidth":xwidth,
              "forcing twidth":twidth,
              "phase":phase,
              "shape":"(peak,xwidth,twidth,phase or (mean,std))"}
    
    np.save("forcingsweep.npy",output)
    