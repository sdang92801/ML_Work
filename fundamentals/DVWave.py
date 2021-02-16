import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,6.2,100)
gamma = [0.0,0.125,0.25,0.375,0.5]
phi = [0.0,0.125,0.25,0.375,0.5]

def sinwave(gamma,x,phi):
    return gamma*np.sin(x+phi)

for i,gam in enumerate(gamma):
    plt.plot(x,sinwave(gam,x,phi[i]))

plt.xlabel(r'$\theta$(rad)')
plt.legend()
plt.show()
