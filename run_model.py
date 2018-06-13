import numpy as np
import matplotlib.pyplot as plt

import AtmosphericBlocking

plt.close("all")

model = AtmosphericBlocking.Model(nx=2048,Lx = 28000e3,dt=.001*86400,alpha=0.55,
                                        tmax=3.5*86400,D=3.26e5,tau=10*86400)

#model.run()
tmaxes = np.linspace(1,30,50)*86400


A = 0

for tmax in tmaxes:

    model.tmax = tmax
    model.run()

    try:
        A = np.vstack([A,model.A[np.newaxis]])
    except:
        A = model.A

plt.figure(figsize=(5,8))
plt.contourf(model.x/1e3,tmaxes/86400,A)
plt.xlabel(r"x")
plt.ylabel(r"Day")
