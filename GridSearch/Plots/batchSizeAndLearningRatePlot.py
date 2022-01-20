#libraries
from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
import os

#!python
import pylab
plt.style.use('ggplot')
fig_width_pt = 290.0  # Get this from LaTeX using \showthe\columnwidth  483.69687pt x 290.57135pt.
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 10,
          'legend.fontsize': 7,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': fig_size}
pylab.rcParams.update(params)
plt.axes([0.145,0.15,0.95-0.145,0.95-0.15])

#Data
batchSize= [2**i for i in range(5,11,1)]

learningRate3 = [0.0290, 0.0129, 0.0155, 0.0355, 0.0577, 0.0638]
learningRate4 = [0.0047,	0.0079,	0.0089,	0.0195,	0.0271,	0.0181]
learningRate5 = [0.0073,	0.0090,	0.0111,	0.0042,	0.0111,	0.0135]
learningRate6 = [0.0108,	0.0141,	0.0134,	0.0170,	0.0187,	0.0154]

plt.plot(range(len(batchSize)), learningRate3, label='$\eta=10^{-3}$', marker="o")
plt.plot(range(len(batchSize)), learningRate4, label='$\eta=10^{-4}$', marker="o")
plt.plot(range(len(batchSize)), learningRate5, label='$\eta=10^{-5}$', marker="o")
plt.plot(range(len(batchSize)), learningRate6, label='$\eta=10^{-6}$', marker="o")
x_labels = ['$2^5$', '$2^6$', '$2^7$', '$2^8$', '$2^9$', '$2^{10}$']
plt.xticks(ticks=range(len(batchSize)), labels=x_labels)
#plt.title("Price Grid Search Learning Rate and Batch Size")
plt.xlabel("Batch Size")
plt.legend(loc='upper left', title='Learning rates')
plt.ylabel("Price Error")
x1,x2,y1,y2 = plt.axis()  
plt.axis((x1,x2,y1,0.04))
plt.savefig(os.path.join("..", "..", "..", "My_Papers", "FNNMC_v2", "Illustration", "batchSizeLearningGridSearch.eps"))
plt.show()