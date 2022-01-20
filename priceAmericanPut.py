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
import FNNMC

# reproducablity
seed = 3
import random
import torch
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

putOption = Products.Option(timeToMat=1, strike=1, typeOfContract="Put")
underlyingsTotal = 1
normalizeStrike= 40
marketVariables = Products.MarketVariables(r=0.06,vol=0.2, spot=36/normalizeStrike)
hyperparameters = FNNMC.Hyperparameters(learningRate=10**(-5), inputSize=underlyingsTotal, 
                        hiddenlayer1=underlyingsTotal+100, hiddenlayer2=underlyingsTotal+100, hiddenlayer3=underlyingsTotal+100, 
                        hiddenlayer4=underlyingsTotal+100, epochs=10**4, batchSize=256, trainOnlyLastTimeStep=False, patience=4)

# load data
f = h5py.File(os.path.join('data', 'AmericanPut', '1MPut100K.hdf5'), 'r')
learningPaths = f['RND'][...]
timeRegressionStart = time.time()
FNNMC.findNeuralNetworkModels(simulatedPaths=learningPaths, Option=putOption, MarketVariables=marketVariables, hyperparameters=hyperparameters)
timeRegressionEnd = time.time()
print(f"Time taken for find regressioncoefficients: {timeRegressionEnd-timeRegressionStart:f}")

estimates = np.zeros(100)
for i in range(100):
    # create empirical estimations
    g = h5py.File(os.path.join(".", "data", "AmericanPut", "PricePaths100K", f"PricePath{i}.hdf5"), 'r')
    pricingPaths = g['RND'][...]
    timePriceStart = time.time()
    price = FNNMC.priceAmericanOption(simulatedPaths=pricingPaths, Option=putOption, MarketVariables=marketVariables, hyperparameters=hyperparameters)*normalizeStrike
    timePriceEnd = time.time()
    print("Estimate: ", i, " Price: ", price)
    estimates[i]=price
print("Mean: ", np.mean(estimates))
print("Std Error Mean: ", np.std(estimates)/10)
#
##
# LSM
######
#Price American Put
import LSM
#normalizeStrike=40
#spot = 36
#putOption = Products.Option(timeToMat=1, strike=1, typeOfContract="Put")
#MarketVariablesEx1 = Products.MarketVariables(r=0.06,vol=0.2, spot=spot/normalizeStrike, dividend=0.0)
#f = h5py.File(os.path.join('data', 'AmericanPut', '1MPut100K.hdf5'), 'r')
#learningPaths = f['RND'][...]
#timeRegressionStart = time.time()
#regressionCoefficient = LSM.findRegressionCoefficient(basisFuncTotal=3, Option=putOption, simulatedPaths=learningPaths, MarketVariables=MarketVariablesEx1)
#timeRegressionEnd = time.time()
#print(f"Time taken for regression {timeRegressionEnd-timeRegressionStart:f}")
#estimates = np.zeros(100)
#for i in range(100):
#    # create empirical estimations
#    g = h5py.File(os.path.join(".", "data", "AmericanPut", "PricePaths100K", f"PricePath{i}.hdf5"), 'r')
#    pricingPaths = g['RND'][...]
#    timePriceStart = time.time()
#    price = LSM.priceAmericanOption(coefficientMatrix=regressionCoefficient, Option=putOption , simulatedPaths=pricingPaths, MarketVariables=MarketVariablesEx1)*normalizeStrike
#    timePriceEnd = time.time()
#    estimates[i]=price
#print("Mean: ", np.mean(estimates))
#print("Std Error Mean: ", np.std(estimates)/10)
#