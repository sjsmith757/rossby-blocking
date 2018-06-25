import os
import numpy as np
import cx_hold_blockinggrid as cx

_BATCHSCRIPT = ("#!/bin/bash                                                  \n"+
                "#SBATCH --job-name=%s                                        \n"+
                "#SBATCH --output=%%j_%s.out                          \n"+
                "#SBATCH --error=%%j_%s.err                            \n"+
                "#SBATCH --ntasks=1                                          \n"+
                "#SBATCH --mem-per-cpu=2000M                                  \n"+
                "#SBATCH --account=rossby                                         \n"
                "#SBATCH --partition=broadwl                                       \n"+
                "#SBATCH --mail-user=paradise@astro.utoronto.ca \n"+
                "#SBATCH --mail-type=END \n"+
                "#SBATCH --time=16:00:00                                      \n"+
                "module load python                   \n"+
                "cd %s                                                        \n")

RUNSTRING = "python cx_hold_blockinggrid.py %s %s %f \n"

if __name__=="__main__":
    
    niters = 1
    nperjob = 16
    ntasks = niters//nperjob
    namps = 16
    
    initc = cx.noisyconditions(cfunc=cx.noisybackground,cxpeak=1.0,peak=1.7,
                               background=True,forcing=True,beta=60,Y=10)
    np.save("bconditions.npy",initc,fix_imports=True)
    
    for beta in np.linspace(30.0,90.0,num=16):
        ndir = "cx_bygrid_%2.1f"%beta
        os.system("mkdir "+ndir)
        f=open(ndir+"/runblocking","w")
        name = ndir
        txt = _BATCHSCRIPT%(name,name,name,os.getcwd()+"/"+ndir)
        txt += RUNSTRING%(name,"bconditions.npy",beta)
        f.write(txt)
        f.close()
        os.system("cp *.py "+ndir+"/")
        os.system("cp bconditions.npy "+ndir+"/")
        os.system("cd "+ndir+" && sbatch runblocking && cd ../")
                