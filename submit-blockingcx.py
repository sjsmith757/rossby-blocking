import os
import numpy as np

_BATCHSCRIPT = ("#!/bin/bash                                                  \n"+
                "#SBATCH --job-name=%s                                        \n"+
                "#SBATCH --ntasks=1                                          \n"+
                "#SBATCH --mem-per-cpu=2000M                                  \n"+
                "#SBATCH --account=rossby                                         \n"
                "#SBATCH --partition=broadwl                                       \n"+
                "#SBATCH --time=8:00:00                                      \n"+
                "module load python                   \n"+
                "cd %s                                                        \n")

RUNSTRING = "python runblocking_cx.py %d %f %s  \n"

if __name__=="__main__":
    
    x1 = 50.0
    x2 = 5000.0

    t1 = 0.5
    t2 = 10.0
    
    p1 = 0.5
    p2 = 10.0
    
    for n in range(1,13):
        for Y in np.logspace(0,2,num=10):
            ndir = "cxsweep_%02.4f_%04.5f_"%(n,Y)
            os.system("mkdir "+ndir)
            f=open(ndir+"/runblocking","w")
            name="cxblock%02d_%04.1f_%02.1f"%(n,Y)
            print ndir,name
            txt = _BATCHSCRIPT%(name,ndir)
            txt += RUNSTRING%(n,Y,name)
            f.write(txt)
            f.close()
            os.system("cp *.py "+ndir+"/")
            os.system("cd "+ndir+" && sbatch runblocking && cd ../")
                
                