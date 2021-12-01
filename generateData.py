#!/usr/bin/env python
# PyTorch 1.8.1-CPU virtual env.
# Python 3.9.4 Windows 10
# -*- coding: utf-8 -*-

"""The script implement the classical longstaff-schwartz algorithm for pricing american options.
This script focus on the multidimensional case for rainbow option
"""
# reproducablity
seed = 3
import random
import numpy as np
import torch

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import Products
import time
import SimulationPaths.GBM
import h5py
#############
# American Put
############
timeStepsTotal = 1000
normalizeStrike=40
spot = 36
putOption = Products.Option(timeToMat=1, strike=1, typeOfContract="Put")
marketVariables = Products.MarketVariables(r=0.06,vol=0.2, spot=spot/normalizeStrike)
learningPaths= SimulationPaths.GBM.generateSDEStockPaths(pathTotal=10**6, timeStepsPerYear=timeStepsTotal, timeToMat=putOption.timeToMat, MarketVariables=marketVariables)
##Safe data
f = h5py.File('data/AmericanPut/1MPut1000K.hdf5', 'w')
f.create_dataset('RND', data = learningPaths)
f.close()
#
for i in range(100):
    # create empirical estimations
    pricingPaths = SimulationPaths.GBM.generateSDEStockPaths(pathTotal=10**4, timeStepsPerYear=timeStepsTotal, timeToMat=putOption.timeToMat, MarketVariables=marketVariables)
    g = h5py.File(f"data/AmericanPut/PricePaths1000K/PricePath{i}.hdf5", 'w')
    g.create_dataset('RND', data = pricingPaths)
    g.close()
    print(i)
######################
# American Call Max option on two stocks
########################
import SimulationPaths.GBMMultiDim
#timeStepsTotal = 9
#normalizeStrike=100
#callMax = Products.Option(timeToMat=3, strike=1, typeOfContract="CallMax")
#underlyingsTotal = 2
#marketVariables = Products.MarketVariables(r=0.05, dividend=0.10, vol=0.2, spot=[100/normalizeStrike]*underlyingsTotal, correlation=0.0)
#
#timeSimPathsStart = time.time()
#learningPaths = SimulationPaths.GBMMultiDim.simulatePaths(timeStepsTotal=timeStepsTotal,pathsTotal=10**6, marketVariables=marketVariables, timeToMat=callMax.timeToMat)
#timeSimPathsEnd = time.time()
#print(f"Time taken to simulate paths is: {timeSimPathsEnd-timeSimPathsStart:f}")
###Safe data
#f = h5py.File('data/MaxCall/1MPut.hdf5', 'w')
#f.create_dataset('RND', data = learningPaths)
#f.close()
#
#estimates = np.zeros(100)
#for i in range(100):
#    # create empirical estimations
#    pricingPaths = SimulationPaths.GBMMultiDim.simulatePaths(timeStepsTotal=timeStepsTotal,pathsTotal=10**4, marketVariables=marketVariables, timeToMat=callMax.timeToMat)
#    g = h5py.File(f"data/MaxCall/PricePaths/PricePath{i}.hdf5", 'w')
#    g.create_dataset('RND', data = pricingPaths)
#    g.close()
#    print(i)

####################
# American Geometric Average option on 7 stocks
#####################
#import SimulationPaths.GBMMultiDim
#timeStepsTotal = 10
#normalizeStrike=100
#geometricCall = Products.Option(timeToMat=1, strike=1, typeOfContract="CallGeometricAverage")
#underlyingsTotal = 7
#marketVariables = Products.MarketVariables(r=0.03, dividend=0.05, vol=0.4, spot=[100/normalizeStrike]*underlyingsTotal, correlation=0.0)
#
#timeSimPathsStart = time.time()
#learningPaths = SimulationPaths.GBMMultiDim.simulatePaths(timeStepsTotal=timeStepsTotal,pathsTotal=10**4, marketVariables=marketVariables, timeToMat=geometricCall.timeToMat)
#timeSimPathsEnd = time.time()
#print(f"Time taken to simulate paths is: {timeSimPathsEnd-timeSimPathsStart:f}")
###Safe data
#f = h5py.File('data/GeometricCall/10KAssets7.hdf5', 'w')
#f.create_dataset('RND', data = learningPaths)
#f.close()
#
#estimates = np.zeros(100)
#for i in range(100):
#    # create empirical estimations
#    pricingPaths = SimulationPaths.GBMMultiDim.simulatePaths(timeStepsTotal=timeStepsTotal,pathsTotal=10**3, marketVariables=marketVariables, timeToMat=geometricCall.timeToMat)
#    g = h5py.File(f"data/GeometricCall/PricePathsFewData/PricePath{i}.hdf5", 'w')
#    g.create_dataset('RND', data = pricingPaths)
#    g.close()
#    print(i)
