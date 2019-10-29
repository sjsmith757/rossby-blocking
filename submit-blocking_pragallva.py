import os
import numpy as np

_BATCHSCRIPT = ("#!/bin/bash                                                  \n"+
                "#SBATCH --job-name=%s                                        \n"+
                "#SBATCH --output=%%j_%s.out                          \n"+
                "#SBATCH --error=%%j_%s.err                            \n"+
                "#SBATCH --ntasks=1                                          \n"+
                "#SBATCH --mem-per-cpu=2000M                                  \n"+
                "#SBATCH --account=pi-nnn                                       \n"+
                "#SBATCH --time=16:00:00                                      \n"+
                "module load python                   \n"+
                "cd %s                                                        \n")

RUNSTRING = "python noisyblocking.py %d %d %f %f %s  \n"

if __name__=="__main__":
    
    niters = 240
    nperjob =60
    ntasks = niters//nperjob
    namps = 20
    
    for fpeak in np.linspace(0.5,4.0,num=namps):
        for i in range(0,ntasks):
            ndir = "fpeak_%1.4f_%d"%(fpeak,i)
            os.system("mkdir "+ndir)
            f=open(ndir+"/runblocking","w")
            name = ndir
            txt = _BATCHSCRIPT%(name,name,name,os.getcwd()+"/"+ndir)
            txt += RUNSTRING%(nperjob,2,1.0,fpeak,name)
            f.write(txt)
            f.close()
            os.system("cp *.py "+ndir+"/")
            os.system("cd "+ndir+" && sbatch runblocking && cd ../")
                
