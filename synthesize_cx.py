import numpy as np
import matplotlib.pyplot as plt

# Block Shape: (time, xwidth, twidth, phase, (xcoord,tcoord)) (the last only for onsets)

if __name__=="__main__":
    nblocks = np.zeros((12,20,20))
    starts = np.zeros((12,20,20,2),dtype=object)
    avgdelays = np.zeros((12,20,20))
    stddelays = np.zeros((12,20,20))
    phase = np.zeros(20)
    
    foundphase = False
    
    x1 = 50.0
    x2 = 5000.0

    t1 = 0.5
    t2 = 10.0
    
    p1 = 0.5
    p2 = 10.0
    
    wavenums = range(1,13)
    ampls = np.logspace(0,2,num=20)
    
    nn=0
    for n in wavenums:
        nY=0
        for Y in ampls:
            name="cxblock%02d_%03.1f.npy"%(n,Y)
            output = np.load(name)
            for nphase in range(0,20):
                nblocks[nn,nY,nphase] = output[nphase]["nblocks"]
                starts[nn,nY,nphase,0] = output[nphase]["onset"][0]
                starts[nn,nY,nphase,1] = output[nphase]["onset"][1]
                dels = np.array(output[nphase]["delay"][1])
                if len(dels)>0:
                    avgdelays[nn,nY,nphase] = np.mean(dels)
                    stddelays[nn,nY,nphase] = np.std(dels)
                else:
                    avgdelays[nn,nY,nphase] = np.nan
                    stddelays[nn,nY,nphase] = np.nan
                if not foundphase:
                    phase[nphase] = output[nphase]["forcing phase"]
            foundphase=True
            nY+=1
        nn+=1
        print("Finished Wavenum %d of %d"%(nn,len(wavenums)))
        
    nblkstats = np.zeros((12,20,2))
    ndelstats = np.zeros((12,20,2))
        
    nblkstats[:,:,0] = np.mean(nblocks,axis=2)
    nblkstats[:,:,1] =  np.std(nblocks,axis=2)
    
    ndelstats[:,:,0] = np.nanmean(avgdelays,axis=2)
    ndelstats[:,:,1] =  np.sqrt(np.nansum((stddelays*avgdelays)**2,axis=2))
    
    output = {"raw blocks":nblocks,
              "onset coords":starts,
              "onset delays":(avgdelays,stddelays),
              "block stats":nblkstats,
              "delay stats":ndelstats,
              "wavenums":wavenums,
              "amplitudes":ampls,
              "phase":phase,
              "shape":"(wavenumber,amplitude,phase or (mean,std))"}
    
    np.save("cxsweep.npy",output)
    