
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

learningRate = np.array([10**(-1), 10**(-2), 10**(-3), 10**(-4), 10**(-5), 10**(-6)])

mixEpochCallMax2= np.array([13.2422, 13.7349, 13.8901, 13.9158, 13.9048, 13.6400])
CallMax2= np.array([13.2664, 13.7723, 13.8944, 13.9090, 13.9183, 13.9054])
priceCallMax2 = 13.902
errorMixEpochCallMax2 = mixEpochCallMax2-priceCallMax2
errorCallMax2 = CallMax2-priceCallMax2

mixEpochCallMax5= np.array([24.1913, 25.7052, 26.0977, 26.1077, 26.0539, 25.8636])
CallMax5= np.array([24.2262, 26.0542, 25.9720, 26.1098, 26.1170, 26.0227])
priceCallMax5 = 26.1395
errorMixEpochCallMax5 = mixEpochCallMax5-priceCallMax5
errorCallMax5 = CallMax5-priceCallMax5

mixEpochGeometricPut15 = np.array([1.0843, 1.0882, 1.0955, 1.0512, 1.0874, 1.0665])
GeometricPut15 = np.array([1.0880, 1.1009, 1.1029, 1.0952, 1.1002, 1.0880])
priceGeometricPut15 = 1.1190
errorMixEpochGeometricPut15 = mixEpochGeometricPut15-priceGeometricPut15
errorGeometricPut15 = GeometricPut15-priceGeometricPut15



plt.style.use('ggplot')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

learningRate = np.array([10**(-2), 10**(-3), 10**(-4), 10**(-5), 10**(-6)])
plt.plot(learningRate, errorMixEpochCallMax2[1:], '*', color='Blue', label='Mix Epoch Call Max 2')
#plt.plot(learningRate, errorMixEpochCallMax5[1:], 'v', color='Red', label='Mix Epoch Call Max 5')
plt.plot(learningRate, errorMixEpochGeometricPut15[1:], '+', color='Green', label='Mix Epoch Geometric Put 15')
plt.legend()
plt.title('Error from True Price', fontsize=11)
plt.xlabel('Learning Rate', fontsize=11)
plt.xscale('log',base=10) 
plt.ylabel('Error', fontsize=11)
plt.grid(True)
pathToSave = f'C:\\Users\\HY34WN\\OneDrive - Aalborg Universitet\\Documents\\PhD\\My_Papers\\FNNMC_v2\\Illustration\\GridSearch\\LearningRate\\mixEpochFigure.eps'
plt.savefig(pathToSave)
plt.show()


learningRate = np.array([10**(-2), 10**(-3), 10**(-4), 10**(-5), 10**(-6)])
plt.plot(learningRate, errorCallMax2[1:], '*', color='Blue', label='Call Max 2')
#plt.plot(learningRate, errorCallMax5[1:], 'v', color='Red', label='Call Max 5')
plt.plot(learningRate, errorGeometricPut15[1:], '+', color='Green', label='Geometric Put 15')
plt.legend()
plt.title('Error from True Price', fontsize=11)
plt.xlabel('Learning Rate', fontsize=11)
plt.xscale('log',base=10) 
plt.ylabel('Error', fontsize=11)
plt.grid(True)
pathToSave = f'C:\\Users\\HY34WN\\OneDrive - Aalborg Universitet\\Documents\\PhD\\My_Papers\\FNNMC_v2\\Illustration\\GridSearch\\LearningRate\\figure.eps'
plt.savefig(pathToSave)
plt.show()