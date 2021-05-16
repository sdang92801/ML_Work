import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pylab as plt
x = np.linspace(-np.pi, np.pi, 201)
p=[0,.125,.25,.375,.5]
# A=[0,.125,.25,.375,.5]
for t in p:
    plt.plot(x, t*(np.sin(x+t)))
plt.show()