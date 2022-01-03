import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

data = pd.read_excel(r'C:\Users\HY34WN\OneDrive - Aalborg Universitet\Documents\PhD\My_Papers\FFNNMC\gridSearch.xlsx', sheet_name='reProduceGeometricPut')
Diff = pd.DataFrame(data, columns= ['Diff']).to_numpy()
width = pd.DataFrame(data, columns= ['Width']).to_numpy()
hiddenLayers = pd.DataFrame(data, columns=['Hidden Layers']).to_numpy()
activationFunc = pd.DataFrame(data, columns=['Activation Function']).to_numpy()

#Grouping data
diff1 = Diff[width==1]
diff2 = Diff[width==10]
diff3 = Diff[width==100]
diff4 = Diff[width==1000]
hiddenLayers1 = hiddenLayers[width==1]
hiddenLayers2 = hiddenLayers[width==10]
hiddenLayers3 = hiddenLayers[width==100]
hiddenLayers4 = hiddenLayers[width==1000]
plt.style.use('ggplot')
plt.plot(hiddenLayers1, diff1, '*', color='Red', label='Width d+1')
plt.plot(hiddenLayers2, diff2, '+', color='Blue', label='Width d+10')
plt.plot(hiddenLayers3, diff3, 'x', color='Green', label='Width d+100')
plt.plot(hiddenLayers4, diff4, '_', color='Black', label='Width d+1000')
plt.plot(hiddenLayers,[0]*48, '-')
plt.legend()
plt.title('Grid Seach Geometric 15')
plt.xlabel('Hidden Layers')
plt.ylabel('Difference from True Price')
pathToSave = f'C:\\Users\\HY34WN\\OneDrive - Aalborg Universitet\\Documents\\PhD\\My_Papers\\FNNMC_v2\\Illustration\\GridSearch\\GeometricPut15\\diffPlot.eps'
plt.savefig(pathToSave)
plt.show()
