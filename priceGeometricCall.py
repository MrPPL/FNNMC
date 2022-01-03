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
import h5py
import FNNMCMultiDim

# reproducablity
seed = 3
import random
import torch
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# load data
f = h5py.File(os.path.join('.','data','GeometricCall', '1MAssets7.hdf5'), 'r')
learningPaths = f['RND'][...]

timeStepsTotal = 10
normalizeStrike= 100
geometricCall = Products.Option(timeToMat=1, strike=1, typeOfContract="CallGeometricAverage")
underlyingsTotal = 7
marketVariables = Products.MarketVariables(r=0.03, dividend=0.05, vol=0.4, spot=[100/normalizeStrike]*underlyingsTotal, correlation=0.0)

hyperparameters = FNNMCMultiDim.Hyperparameters(learningRate=10**(-5), inputSize=underlyingsTotal, 
                        hiddenlayer1=underlyingsTotal+100, hiddenlayer2=underlyingsTotal+100, hiddenlayer3=underlyingsTotal+100, 
                        hiddenlayer4=underlyingsTotal+100, epochs=10**4, batchSize=256, 
                        trainOnlyLastTimeStep=False, patience=4)
timeRegressionStart = time.time()
FNNMCMultiDim.findNeuralNetworkModels(simulatedPaths=learningPaths, Option=geometricCall, MarketVariables=marketVariables, hyperparameters=hyperparameters)
timeRegressionEnd = time.time()
print(f"Time taken for find regressioncoefficients: {timeRegressionEnd-timeRegressionStart:f}")
estimates = np.zeros(100)
for i in range(100):
    # create empirical estimations
    g = h5py.File(os.path.join(".", "data", "GeometricCall", "PricePaths", f"PricePath{i}.hdf5"), 'r')
    pricingPaths = g['RND'][...]
    timePriceStart = time.time()
    price = FNNMCMultiDim.priceAmericanOption(simulatedPaths=pricingPaths, Option=geometricCall, MarketVariables=marketVariables, hyperparameters=hyperparameters)*normalizeStrike
    timePriceEnd = time.time()
    estimates[i]=price
print("BatchSize: ", 256, " Mean: ", np.mean(estimates))
print("Std Error Mean: ", np.std(estimates)/10)
