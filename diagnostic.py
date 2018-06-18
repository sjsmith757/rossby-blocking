import numpy as np

# Calling: call nblocks = diagnostic.count(flux,it0) to get number of blocks after time t0,
# where it0 is the index of time t0 (i.e., t0 = time[it0]).

def count_blocks(mask,dx,dt):
    #Count number of contiguous patches on binary map
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

def count(flux,t0):
    #Get number of blocks
    grad = (np.gradient(flux[it0:],axis=1))
    fmask = ((grad-np.mean(grad))/np.std(grad) > 5)*1.0
    ict,dmask = count_blocks(fmask,10,10)
    return ict

def size(flux,t0):
    #Get total area of map which is a steepened block edge
    grad = (np.gradient(flux[it0:],axis=1))
    fmask = ((grad-np.mean(grad))/np.std(grad) > 5)*1.0
    return np.sum(fmask)
    
def blockstarts(flux,t0):
    #Get coordinates of start of blocking events
    grad = (np.gradient(flux[it0:],axis=1))
    fmask = ((grad-np.mean(grad))/np.std(grad) > 5)*1.0
    ict,dmask = count_blocks(fmask,10,10)
    ts = t[it0:][np.where(dmask>0.5)[0]]/86400
    xs = x[np.where(dmask>0.5)[1]]/1e3
    return xs,ts
