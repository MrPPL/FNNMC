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
import time
import FNNMCMultiDim
import h5py
import FNNMC
# load data
f = h5py.File('Gridsearch/Data/GeometricPut/1MGeometricPut15Assets.hdf5', 'r')
learningPaths = f['RND'][...]

timeStepsTotal = 10
normalizeStrike=40
underlyingsTotal = 15
callGeometricAverage = Products.Option(timeToMat=1, strike=1, typeOfContract="PutGeometricAverage")
marketVariables = Products.MarketVariables(r=0.06, dividend=0.0, vol=0.2, spot=[40/normalizeStrike]*underlyingsTotal, correlation=0.25)
hyperparameters = FNNMC.Hyperparameters(learningRate=0.001, inputSize=underlyingsTotal, hiddenlayer1=underlyingsTotal+1, epochs=10, batchSize=10**4, trainOnlyLastTimeStep=True)
timeRegressionStart = time.time()
FNNMC.findNeuralNetworkModels(simulatedPaths=learningPaths, Option=callGeometricAverage, MarketVariables=marketVariables, hyperparameters=hyperparameters)
timeRegressionEnd = time.time()
print(f"Time taken for find regressioncoefficients: {timeRegressionEnd-timeRegressionStart:f}")
estimates = np.zeros(100)
for i in range(100):
    # create empirical estimations
    g = h5py.File(f"GridSearch/Data/GeometricPut/PricePaths15Asset/PricePath{i}.hdf5", 'r')
    pricingPaths = g['RND'][...]
    timePriceStart = time.time()
    price = FNNMCMultiDim.priceAmericanOption(simulatedPaths=pricingPaths, Option=callGeometricAverage, MarketVariables=marketVariables, hyperparameters=hyperparameters)*normalizeStrike
    timePriceEnd = time.time()
    print(f"Time taken for Pricing: {timePriceEnd-timePriceStart:f}")
    print(f"The estimated price is: {price:f} and the true price is: 1.119")
    estimates[i]=price
print("Mean: ", np.mean(estimates))
print("Std Error Mean: ", np.std(estimates)/10)
