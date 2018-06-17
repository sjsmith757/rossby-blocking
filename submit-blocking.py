import os
import numpy as np

_BATCHSCRIPT = ("#!/bin/bash                                                  \n"+
                "#SBATCH --job-name=%s                                        \n"+
                "#SBATCH --ntasks=1                                          \n"+
                "#SBATCH --mem-per-cpu=2000M                                  \n"+
                "#SBATCH --account=rossby                                         \n"
                "#SBATCH --partition=broadwl                                       \n"+
                "#SBATCH --time=16:00:00                                      \n"+
                "module load python                   \n"+
                "cd %s                                                        \n")

RUNSTRING = "python runblocking_forcing.py %f %f %f %s  \n"

if __name__=="__main__":
    
    x1 = 50.0
    x2 = 5000.0

    t1 = 0.5
    t2 = 10.0
    
    p1 = 0.5
    p2 = 10.0
    
    for t in np.logspace(np.log10(t1),np.log10(t2),num=5):
        for x in np.logspace(np.log10(x1),np.log10(x2),num=10):
            for p in np.logspace(np.log10(p1),np.log10(p2),num=5):
                ndir = "forcingsweep_%02.1f_%04.1f_%02.1f"%(t,x,p)
                os.system("mkdir "+ndir)
                f=open(ndir+"/runblocking","w")
                name="block%02.1f_%04.1f_%02.1f"%(t,x,p)
                print ndir,name
                txt = _BATCHSCRIPT%(name,os.getcwd()+"/"+ndir)
                txt += RUNSTRING%(t,x,p,name)
                f.write(txt)
                f.close()
                os.system("cp *.py "+ndir+"/")
                os.system("cd "+ndir+" && sbatch runblocking && cd ../")
                
                