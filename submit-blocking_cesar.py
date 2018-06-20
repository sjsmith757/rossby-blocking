import os
import numpy as np

_BATCHSCRIPT = ("#!/bin/bash                                                  \n"+
                "#SBATCH --job-name=%s                                        \n"+
                "#SBATCH --output=%%j_%s.out                          \n"+
                "#SBATCH --error=%%j_%s.err                            \n"+
                "#SBATCH --ntasks=1                                          \n"+
                "#SBATCH --mem-per-cpu=2000M                                  \n"+
                "#SBATCH --account=rossby                                         \n"
                "#SBATCH --partition=broadwl                                       \n"+
                "#SBATCH --time=16:00:00                                      \n"+
                "module load python                   \n"+
                "cd %s                                                        \n")

RUNSTRING = "python noisyblocking.py %d %d %f %f %s  \n"

if __name__=="__main__":
    
    niters = 96
    nperjob = 16
    ntasks = niters//nperjob
    namps = 10
    
    for cpeak in np.logspace(-1,np.log10(2.0),num=namps):
        for i in range(0,ntasks):
            ndir = "cstrength_%1.4f_%d"%(cpeak,i)
            os.system("mkdir "+ndir)
            f=open(ndir+"/runblocking","w")
            name = ndir
            txt = _BATCHSCRIPT%(name,name,name,os.getcwd()+"/"+ndir)
            txt += RUNSTRING%(nperjob,2,cpeak,1.8,name)
            f.write(txt)
            f.close()
            os.system("cp *.py "+ndir+"/")
            os.system("cd "+ndir+" && sbatch runblocking && cd ../")
                