import os
import glob
import numpy as np
import sys

if __name__=="__main__":
    
    prefix = sys.argv[1]
    niters = int(sys.argv[2])
    nperjob = int(sys.argv[3])
    
    ntasks = niters//nperjob
    
    files = sorted(glob.glob(prefix+"_*.npy"))
    
    nwforce = 26
    nwcx = 21
    
    output = {"var":np.zeros(len(files)//ntasks),
              "data":[]}
    
    print(len(files)//ntasks)
        
    print(nperjob,ntasks)
    
    l=0
    k=0
    ir = 0
    
    while l<len(files):
        
        ffreqs = np.zeros((niters,nwforce))
        fspeeds = np.zeros((niters,nwforce))
        fphases = np.zeros((niters,nwforce))
        fampls = np.zeros((niters,nwforce))

        cfreqs = np.zeros((niters,nwcx))
        cspeeds = np.zeros((niters,nwcx))
        cphases = np.zeros((niters,nwcx))
        campls = np.zeros((niters,nwcx))

        nblockseq = np.zeros(niters)
        blockszseq = np.zeros(niters)

        avgnblocks = 0.0
        avgblocksize = 0.0

        avgnblocks_per_event = 0.0
        avgsize_per_event = 0.0

        nblockseq_perevent = np.zeros(niters)
        blockszseq_perevent = np.zeros(niters)
        
        onsets = []
        events = []
        
        print(ffreqs.shape)
        print(l)
        k=0
        
        print(ir,files[l].split('_'))
        output["var"][ir] = float(files[l].split('_')[1])
        
        for nf in range(0,ntasks):
            df = np.load(files[l]).item()
            
            print(df)
            
            onsets += df["block_coords"]
            events += df["forcing_coords"]
            
            print(ffreqs[k*nperjob:(k+1)*nperjob,:].shape,df["forcing_init"]["freqs"][:].shape)
            
            ffreqs[k*nperjob:(k+1)*nperjob,:]  = df["forcing_init"]["freqs"][:]
            fspeeds[k*nperjob:(k+1)*nperjob,:] = df["forcing_init"]["speeds"][:]
            fphases[k*nperjob:(k+1)*nperjob,:] = df["forcing_init"]["phases"][:]
            fampls[k*nperjob:(k+1)*nperjob,:]  = df["forcing_init"]["amplitudes"][:]
            
            cfreqs[k*nperjob:(k+1)*nperjob,:]  = df["c(x)_init"]["freqs"][:]
            cspeeds[k*nperjob:(k+1)*nperjob,:] = df["c(x)_init"]["speeds"][:]
            cphases[k*nperjob:(k+1)*nperjob,:] = df["c(x)_init"]["phases"][:]
            campls[k*nperjob:(k+1)*nperjob,:]  = df["c(x)_init"]["amplitudes"][:]
            
            wavenum = df["cx_wavenumber"]
            cx_peak = df["cx_peak"]
            fpeak = df["forcing_peak"]
            
            
            nblockseq[k*nperjob:(k+1)*nperjob] = df["raw_nblocks"][:]
            blockszseq[k*nperjob:(k+1)*nperjob] = df["raw_blocksize"][:]
            nblockseq_perevent[k*nperjob:(k+1)*nperjob] = df["raw_nblocks_perevent"][:]
            blockszseq_perevent[k*nperjob:(k+1)*nperjob] = df["raw_blocksize_perevent"][:]
            
            k+=1
            l+=1
            
        print("ding")
            
        avgnblocks = np.mean(nblockseq)
        avgblocksize = np.mean(blockszseq)
        avgnblocks_per_event = np.mean(nblockseq_perevent)
        avgsize_per_event = np.mean(blockszseq_perevent)
        
        stdnblocks = np.std(nblockseq)
        stdblocksize = np.std(blockszseq)
        stdnblocks_per_event = np.std(nblockseq_perevent)
        stdsize_per_event = np.std(blockszseq_perevent)
        
        forcing_waves = {"freqs":ffreqs,
                        "speeds":fspeeds,
                        "phases":fphases,
                        "amplitudes":fampls}
        bg_waves = {"freqs":cfreqs,
                    "speeds":cspeeds,
                    "phases":cphases,
                    "amplitudes":campls}
        output["data"].append({"forcing_init":forcing_waves,
                               "c(x)_init":bg_waves,
                               "avg_nblocks":avgnblocks,
                               "avg_blocksize":avgblocksize,
                               "std_nblocks":stdnblocks,
                               "std_blocksize":stdblocksize,
                               "raw_nblocks":nblockseq,
                               "raw_blocksize":blockszseq,
                               "block_coords":onsets,
                               "cx_wavenumber":wavenum,
                               "cx_peak":cx_peak,
                               "forcing_peak":fpeak,
                               "forcing_coords":events,
                               "avg_nblocks_perevent":avgnblocks_per_event,
                               "avg_blocksize_perevent":avgsize_per_event,
                               "std_nblocks_perevent":stdnblocks_per_event,
                               "std_blocksize_perevent":stdsize_per_event,
                               "raw_nblocks_perevent":nblockseq_perevent,
                               "raw_blocksize_perevent":blockszseq_perevent})
        
        ir+=1
        
    np.save(prefix+".npy",output,fix_imports=True)
    
    


            
            