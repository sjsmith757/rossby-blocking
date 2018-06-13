import numpy as np
import matplotlib.pyplot as plt

import AtmosphericBlocking

plt.close("all")

model = AtmosphericBlocking.Model(nx=2048,Lx = 28000e3,dt=.01*86400,alpha=0.55,
                                        tmax=3.5*86400,D=3.26e5,tau=10*86400)

#model.run()
tmaxes = np.linspace(1,30,50)*86400


A = 0
S = 0
for tmax in tmaxes:

    model.tmax = tmax
    model.run()

    try:
        A = np.vstack([A,model.A[np.newaxis]])
        S = np.vstack([S,model.S[np.newaxis]])
    except:
        A = model.A
        S = model.S

lons = model.x/1e3*360/28000. + 100.0
lons[np.where(lons>180)] -= 360.0

plt.figure(figsize=(4.,8))
plt.contourf(model.x/1e3,tmaxes/86400,A+model.A0[np.newaxis,:],np.linspace(0,64,30))
plt.colorbar()
plt.contour(model.x/1e3,tmaxes/86400,S,np.linspace(3e-5,6e-5,5),colors='0.5')
plt.xlabel(r"Distance [km]")
plt.ylabel(r"Day")
plt.title("A")
plt.ylim(5,30)
plt.xlim(13500,model.Lx/1e3)


#
# plt.figure(figsize=(5,8))
# plt.contourf(lons,tmaxes/86400,S)
# plt.xlabel(r"Longitude")
# plt.ylabel(r"Day")
# plt.title("S")
#
# plt.ylim(1,27)
# plt.xlim(-100,100)
