import os
import glob
import numpy as np
import sys

if __name__=="__main__":
    
    prefix = sys.argv[1]

    
    files = sorted(glob.glob(prefix+"_*.npy"))
    
        
    
    nblocks = np.zeros((16,16))
    blocksz = np.zeros((16,16))
    nblockspe = np.zeros((16,16))
    blockszpe = np.zeros((16,16))
    
    
    for l in range(0,len(files)):
        
        
        df = np.load(files[l]).item()
        
        print(df)
        
        
        nblocks[l,:] = df["raw_nblocks"][:]
        blocksz[l,:] = df["raw_blocksize"][:]
        
        nblockspe[l,:] = df["raw_nblocks_perevent"][:]
        blockszpe[l,:] = df["raw_blocksize_perevent"][:]
        

        print("ding")


    output={"nblocks":nblocks,
            "blocksize":blocksz,
            "nblocks_perevent":nblockspe,
            "blocksize_perevent":blockszpe,
            "beta":np.linspace(30,90,num=16),
            "a0y":np.logspace(0,np.log10(20.0),num=16),
            "shape":"(beta,a0y)"}
    

        
    np.save(prefix+".npy",output,fix_imports=True)
    
    


            
            