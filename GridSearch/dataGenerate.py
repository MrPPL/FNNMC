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


#timeStepsTotal = 9
#normalizeStrike=100
#callGeometricAverage = Products.Option(timeToMat=3, strike=1, typeOfContract="CallMax")
#marketVariables = Products.MarketVariables(r=0.05, dividend=0.1, vol=0.2, spot=[100/normalizeStrike]*2, correlation=0.0)

timeStepsTotal = 10
normalizeStrike=40
PutGeometricAverage = Products.Option(timeToMat=1, strike=1, typeOfContract="PutGeometricAverage")
marketVariables = Products.MarketVariables(r=0.06, dividend=0.0, vol=0.2, spot=[40/normalizeStrike]*15, correlation=0.25)

timeSimPathsStart = time.time()
learningPaths = SimulationPaths.GBMMultiDim.simulatePaths(timeStepsTotal=timeStepsTotal,pathsTotal=10**2, marketVariables=marketVariables, timeToMat=PutGeometricAverage.timeToMat)
timeSimPathsEnd = time.time()
print(f"Time taken to simulate paths is: {timeSimPathsEnd-timeSimPathsStart:f}")
##Safe data
with h5py.File(os.path.join("GridSearch", "Data", "GeometricPut", "100GeometricPut15Assets.hdf5"), "w") as f:
    dset = f.create_dataset("RND", data=learningPaths)

for i in range(100):
    # create empirical estimations
    pricingPaths = SimulationPaths.GBMMultiDim.simulatePaths(timeStepsTotal=timeStepsTotal,pathsTotal=10**2, marketVariables=marketVariables, timeToMat=PutGeometricAverage.timeToMat)
    g = h5py.File(os.path.join("GridSearch", "Data", "GeometricPut", "SmallPricePaths15Asset", f"PricePath{i}.hdf5"), "w")
    g.create_dataset('RND', data = pricingPaths)
    g.close()
    print(i)