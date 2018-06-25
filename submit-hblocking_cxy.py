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
                "#SBATCH --mail-user=paradise@astro.utoronto.ca \n"+
                "#SBATCH --mail-type=END \n"+
                "#SBATCH --time=16:00:00                                      \n"+
                "module load python                   \n"+
                "cd %s                                                        \n")

RUNSTRING = "python cx_hold_noisyblocking.py %s  \n"

if __name__=="__main__":
    
    niters = 240
    nperjob = 60
    ntasks = niters//nperjob
    namps = 16
    
    ndir = "cx_a0y"
    os.system("mkdir "+ndir)
    f=open(ndir+"/runblocking","w")
    name = ndir
    txt = _BATCHSCRIPT%(name,name,name,os.getcwd()+"/"+ndir)
    txt += RUNSTRING%(name)
    f.write(txt)
    f.close()
    os.system("cp *.py "+ndir+"/")
    os.system("cd "+ndir+" && sbatch runblocking && cd ../")
                