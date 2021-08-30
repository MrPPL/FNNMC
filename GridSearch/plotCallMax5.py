
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

data = pd.read_excel(r'C:\Users\HY34WN\OneDrive - Aalborg Universitet\Documents\PhD\My_Papers\FFNNMC\gridSearch.xlsx', sheet_name='CallMax5Asset')
price = pd.DataFrame(data, columns= ['Price']).to_numpy()
width = pd.DataFrame(data, columns= ['Width']).to_numpy()
hiddenLayers = pd.DataFrame(data, columns=['Hidden Layers']).to_numpy()
activationFunc = pd.DataFrame(data, columns=['Activation Function']).to_numpy()
price1 = price[np.logical_and(activationFunc=='Relu', width==1)]
price2 = price[np.logical_and(activationFunc=='Relu', width==10)]
price3 = price[np.logical_and(activationFunc=='Relu', width==100)]
price4 = price[np.logical_and(activationFunc=='Relu', width==1000)]
price5 = price[np.logical_and(activationFunc=='Leaky (a=0.01)', width==1)]
price6 = price[np.logical_and(activationFunc=='Leaky (a=0.01)', width==10)]
price7 = price[np.logical_and(activationFunc=='Leaky (a=0.01)', width==100)]
price8 = price[np.logical_and(activationFunc=='Leaky (a=0.01)', width==1000)]
price9 = price[np.logical_and(activationFunc=='Leaky (a=0.3)', width==1)]
price10= price[np.logical_and(activationFunc=='Leaky (a=0.3)', width==10)]
price11= price[np.logical_and(activationFunc=='Leaky (a=0.3)', width==100)]
price12= price[np.logical_and(activationFunc=='Leaky (a=0.3)', width==1000)]
width1 = width[np.logical_and(activationFunc=='Relu', width==1)]
width2 = width[np.logical_and(activationFunc=='Relu', width==10)]
width3 = width[np.logical_and(activationFunc=='Relu', width==100)]
width4 = width[np.logical_and(activationFunc=='Relu', width==1000)]
width5 = width[np.logical_and(activationFunc=='Leaky (a=0.01)', width==1)]
width6 = width[np.logical_and(activationFunc=='Leaky (a=0.01)', width==10)]
width7 = width[np.logical_and(activationFunc=='Leaky (a=0.01)', width==100)]
width8 = width[np.logical_and(activationFunc=='Leaky (a=0.01)', width==1000)]
width9 = width[np.logical_and(activationFunc=='Leaky (a=0.3)', width==1)]
width10= width[np.logical_and(activationFunc=='Leaky (a=0.3)', width==10)]
width11= width[np.logical_and(activationFunc=='Leaky (a=0.3)', width==100)]
width12= width[np.logical_and(activationFunc=='Leaky (a=0.3)', width==1000)]
hiddenLayers1 = hiddenLayers[np.logical_and(activationFunc=='Relu', width==1)]
hiddenLayers2 = hiddenLayers[np.logical_and(activationFunc=='Relu', width==10)]
hiddenLayers3 = hiddenLayers[np.logical_and(activationFunc=='Relu', width==100)]
hiddenLayers4 = hiddenLayers[np.logical_and(activationFunc=='Relu', width==1000)]
hiddenLayers5 = hiddenLayers[np.logical_and(activationFunc=='Leaky (a=0.01)', width==1)]
hiddenLayers6 = hiddenLayers[np.logical_and(activationFunc=='Leaky (a=0.01)', width==10)]
hiddenLayers7 = hiddenLayers[np.logical_and(activationFunc=='Leaky (a=0.01)', width==100)]
hiddenLayers8 = hiddenLayers[np.logical_and(activationFunc=='Leaky (a=0.01)', width==1000)]
hiddenLayers9 = hiddenLayers[np.logical_and(activationFunc=='Leaky (a=0.3)', width==1)]
hiddenLayers10= hiddenLayers[np.logical_and(activationFunc=='Leaky (a=0.3)', width==10)]
hiddenLayers11= hiddenLayers[np.logical_and(activationFunc=='Leaky (a=0.3)', width==100)]
hiddenLayers12= hiddenLayers[np.logical_and(activationFunc=='Leaky (a=0.3)', width==1000)]

from mpl_toolkits import mplot3d
plt.style.use('ggplot')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xticks([0,1,2,3])
ax.set_yticks([1,2,3,4])

ax.set_xlabel("log(width)", fontsize=11)
ax.set_ylabel("L", fontsize=11)
ax.set_zlabel('Price', rotation = 0, fontsize=11)

# Data for a three-dimensional line
ax.plot3D(np.log10(width1), hiddenLayers1, price1,  "o", color="Black", label="a=0")
ax.plot3D(np.log10(width5), hiddenLayers5, price5,  "v", color="Black", label="a=0.01")
ax.plot3D(np.log10(width9), hiddenLayers9, price9,  "*", color="Black", label="a=0.3")
ax.plot3D(np.log10(width2), hiddenLayers2, price2,  "o", color="grey", label="a=0")
ax.plot3D(np.log10(width6), hiddenLayers6, price6,  "v", color="grey", label="a=0.01")
ax.plot3D(np.log10(width10),hiddenLayers10,price10, "*", color="grey", label="a=0.3")
ax.plot3D(np.log10(width3), hiddenLayers3, price3,  "o", color="Blue",label="a=0")
ax.plot3D(np.log10(width7), hiddenLayers7, price7,  "v", color="Blue",label="a=0.01")
ax.plot3D(np.log10(width11),hiddenLayers11,price11, "*", color="Blue",label="a=0.3")
ax.plot3D(np.log10(width4), hiddenLayers4, price3,  "o", color="Red", label="a=0")
ax.plot3D(np.log10(width8), hiddenLayers8, price8,  "v", color="Red", label="a=0.01")
ax.plot3D(np.log10(width12),hiddenLayers12,price12, "*", color="Red", label="a=0.3")
plt.grid(True)
#for ii in range(0,360,1):
#        ax.view_init(elev=10., azim=ii)
#        pathToSave = f'C:\\Users\\HY34WN\\OneDrive - Aalborg Universitet\\Documents\\PhD\\My_Papers\\FFNNMC\\Illustration\\GridSearch\\CallMax5\\Trials\\movie{ii}.png'
#        plt.savefig(pathToSave)
ax.view_init(elev=10., azim=340)
pathToSave = f'C:\\Users\\HY34WN\\OneDrive - Aalborg Universitet\\Documents\\PhD\\My_Papers\\FFNNMC\\Illustration\\GridSearch\\CallMax5\\3DWidthL.png'
plt.savefig(pathToSave)
plt.show()
