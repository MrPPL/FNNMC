#libraries
import numpy as np
import matplotlib.pyplot as plt
import os

#!python
import pylab
plt.style.use('ggplot')
fig_width_pt = 290.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': fig_size}
pylab.rcParams.update(params)
plt.axes([0.145,0.15,0.95-0.145,0.95-0.15])

#Data
patience= np.linspace(1,7,7)
CallMax2PriceErrors = np.array([0.00912,	0.00768,	0.00670,	0.00835,	0.01004,	0.00994,	0.00851])
GeometricPut15PriceErrors= np.array([0.01705,  0.01346,	0.01199,	0.01037,	0.01191,	0.01098,	0.01232])
meanPriceErrors= np.add(CallMax2PriceErrors, GeometricPut15PriceErrors)/2

plt.plot(patience, CallMax2PriceErrors,       marker="o", label='Call Max')
plt.plot(patience, GeometricPut15PriceErrors, marker="o", label='Geometric Average')
plt.plot(patience, meanPriceErrors,           marker="o", label='Average')
#plt.title("Price Grid Search Patience")
plt.xlabel("Patience")
plt.legend(loc="upper right")
plt.ylabel("Price Error")
plt.savefig(os.path.join("..", "..", "..", "My_Papers", "FNNMC_v2", "Illustration", "patienceGridSearch.eps"))
plt.show()