import numpy as np
import h5py
import sys,os
import glob
import matplotlib.cm as cm
import pylab as py
import logging


def CC(decadal=0,year=0, field='F'):
    year_units = 360*4
    Field = 0
    t=[]; beta=[]
    for fni in fnis[decadal*year_units:(decadal+10)*year_units][360*4*(year):360*4*(year+1):4]:
        snap = h5py.File(fni)
        t.append(snap['t'][()])
        beta.append(snap['beta'][()])
        try:
            Field = np.vstack([Field, snap[field][:]])
        except:
            Field = snap[field][:]
        snap.close()
    t = np.array(t)
    beta=np.array(beta)    
    return Field, beta, t


def count_blocks(mask,dx,dt):
    dsh = mask.shape
    nt = dsh[0]
    nx = dsh[1]
    dmask = np.zeros(np.array(mask.shape)+[2*dt,2*dx])
    dmask[dt:-dt,dx:-dx] = mask[:,:]
    dmask[dt:-dt,0:dx] = mask[:,-dx:]
    dmask[dt:-dt,-dx:] = mask[:,0:dx]
    
    ict = 0
    for it in range(nt+dt,dt,-1):
        for ix in range(dx,nx+dx):
            if dmask[it,ix]==1:
                if np.sum(dmask[it-dt:it+dt,ix-dx:ix+dx])>1:
                    dmask[it,ix]=0
    
    ict = np.sum(dmask[dt:-dt,dx:-dx])
    return ict,dmask[dt:-dt,dx:-dx]

def plot_in_matrix(figs=0, axs=0, decadal=0, year=0,  variable='speed'):    
    titles=""
    if (variable=='speed'):
        Ahat, beta, t = CC(decadal,year, 'A') 
        u = (40)-alpha*(A0+ Ahat)
        Field = u 
        h = 2.5
        contour_range = np.arange(-5,40+h,h)
        cmap1 = cm.Spectral
        titles = "%1.2f [m/s]"%(beta.mean())

    if (variable=='LWA'):
        Ahat, beta, t = CC(decadal,year, 'A') 
        A = (A0+ Ahat)
        Field = A 
        contour_range = np.arange(0,85,5)
        cmap1= cm.Spectral_r
        titles = "%1.2f [m/s]"%(beta.mean())
        
    if (variable=='LWA flux'):
        F, beta, t = CC(decadal,year, 'F')
        Field = F
        contour_range = np.arange(200,1100,50)
        cmap1 = cm.afmhot_r
        titles = r"%1.2f [$m^{2}/s^{2}$]"%(beta.mean())
        
    if (variable=='Forcing'):
        S, beta, t = CC(decadal,year, 'S')
        Field = np.sqrt(S)  
        contour_range = np.linspace(0,0.0205,20)
        cmap1= 'viridis'
        titles = r"%1.2f [$m^{2}/s^{2}$]"%(beta.mean())
        
    if (variable=='blocks'):
        Ahat, beta, t = CC(decadal,year, 'A') 
        u = (40)-alpha*(A0+ Ahat)
        thresu = 10
        ugrad = -(np.gradient(u, axis=1))
        blocks = ((ugrad-np.mean(ugrad))/np.std(ugrad) > threshu)*1.0
        uct,umask = count_blocks(blocks,80,10)
        uts = t[np.where(umask>0.5)[0]]/86400
        uxs = x[np.where(umask>0.5)[1]]/1e3        
        
        Field  =  blocks
        cmap1   = 'Reds' 
        contour_range = np.arange(0.6,1.1,0.1)
       
    row=int(decadal/10)
    col=int(year)
    im = axs[row,col].contourf(x/1000, (t/(86400)), Field, contour_range, cmap=cmap1);  
    if (variable=='blocks'):    
        axs[row,col].scatter(uxs,uts,s=180,marker='*',color='b')
    axs[row,col].set_title(titles, fontsize=60)   
    axs[row,col].tick_params(axis='y', length=10, width=5, labelsize=60)
    axs[row,col].tick_params(axis='x', length=10, width=5, labelsize=60)
    axs[row,col].set_ylabel('decade %i \n[days]'%(int(decadal/10)+1), fontsize=60)
    axs[row,col].set_xlabel('[km]', fontsize=60)
    axs[row,col].set_yticklabels(np.arange(0,300+100,100))
    axs[row,col].set_xticks(np.arange(5000,30000,10000))
    axs[row,col].set_xticklabels(np.arange(5000,30000,10000))
#     axs[row,col].set_ylim(0,360)
    cbaxes = figs.add_axes([0.1, 1.01, 0.8, 0.005]) 
    if (decadal+year)==N1:
        cbar=figs.colorbar(im, cax = cbaxes, orientation='horizontal')
        cbar.ax.tick_params(labelsize=70)
            
    for ax in axs.flat:
       ax.label_outer()

    
if __name__=="__main__":

    setup        =  h5py.File("/project2/tas1/pragallva/rossby-blocking/shell_scripts/output_slow_change_Uref/setup.h5",'r')
    x            =  setup['grid/x'][:]
    noise_params =  h5py.File("/project2/tas1/pragallva/rossby-blocking/shell_scripts/noise_params/parameters.h5",'r')
    parameters   =  h5py.File("/project2/tas1/pragallva/rossby-blocking/shell_scripts/output_slow_change_Uref/parameters.h5",'r')

    fnis = (np.array(sorted(glob.glob("/project2/tas1/pragallva/rossby-blocking/shell_scripts/output_slow_change_Uref/snapshots/*.h5"))))

    Y  = 10
    n  = 2
    Lx = parameters['Lx'][()]
    A0 = Y*(1-np.cos(2*np.pi*n*x/Lx))
    alpha = parameters['alpha'][()]

    setup.close()
    noise_params.close()
    parameters.close()

    
    inputs  = int(sys.argv[1]) #90
    
    for handler in logging.root.handlers[:]:
       logging.root.removeHandler(handler)
        
    if (inputs==1):
        variable='speed'

    elif (inputs==2):
        variable='LWA'

    elif (inputs==3):
        variable='LWA flux'

    elif (inputs==4):
        variable='Forcing'

    else:
        variable='blocks'
        
    log_directory='/project2/tas1/pragallva/rossby-blocking/shell_scripts/logs/slow_uref'+variable
    logging.basicConfig(filename=log_directory+'.log',level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    
    logging.debug("============================="+variable+"=============================")
    N1=0;
    N2=200;
    fig, axs = py.subplots(20, 10, sharex=True, sharey=False, figsize=(100,200))
    for year in range(0,10): 
        for decadal in range(N1,N2,10):
           plot_in_matrix(fig, axs, decadal, year, variable)
           logging.debug("YEARS = %i"%(decadal+year+1))
           #print ("YEARS = %i"%(decadal+year+1))

    py.suptitle(variable, fontsize=70, y=1.025) 
    py.tight_layout()
    fig.savefig("/project2/tas1/pragallva/rossby-blocking/shell_scripts/figures/"+variable+".pdf",bbox_inches="tight")
