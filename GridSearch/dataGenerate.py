#!/usr/bin/env python
# PyTorch 1.8.1-CPU virtual env.
# Python 3.9.4 Windows 10
# -*- coding: utf-8 -*-

"""The script implement the classical longstaff-schwartz algorithm for pricing american options.
This script focus on the multidimensional case for rainbow option
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import Products
import SimulationPaths.GBMMultiDim
import time
import h5py

timeStepsTotal = 10
normalizeStrike=40
callGeometricAverage = Products.Option(timeToMat=1, strike=1, typeOfContract="PutGeometricAverage")
marketVariables = Products.MarketVariables(r=0.06, dividend=0.0, vol=0.2, spot=[40/normalizeStrike]*15, correlation=0.25)

timeSimPathsStart = time.time()
learningPaths = SimulationPaths.GBMMultiDim.simulatePaths(timeStepsTotal=timeStepsTotal,pathsTotal=10**6, marketVariables=marketVariables, timeToMat=callGeometricAverage.timeToMat)
timeSimPathsEnd = time.time()
print(f"Time taken to simulate paths is: {timeSimPathsEnd-timeSimPathsStart:f}")
##Safe data
f = h5py.File('Gridsearch/Data/GeometricPut/1MGeometricPut15Assets.hdf5', 'w')
f.create_dataset('RND', data = learningPaths)
f.close()

for i in range(100):
    # create empirical estimations
    pricingPaths = SimulationPaths.GBMMultiDim.simulatePaths(timeStepsTotal=timeStepsTotal,pathsTotal=10**4, marketVariables=marketVariables, timeToMat=callGeometricAverage.timeToMat)
    g = h5py.File(f"GridSearch/Data/GeometricPut/PricePaths15Asset/PricePath{i}.hdf5", 'w')
    g.create_dataset('RND', data = pricingPaths)
    g.close()
    print(i)