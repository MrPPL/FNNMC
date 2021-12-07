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
import GridSearch.FNNMC
# reproducablity
seed = 5
import random
import torch

for patience in range(1,11,1):
    for learningrate in range(2,7,1):
        for batchsize in range(1,11,1):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            # load data
            with h5py.File(os.path.join("GridSearch", "Data", "GeometricPut", "1MGeometricPut15Assets.hdf5"), "r") as f:
                learningPaths = f['RND'][...]

            timeStepsTotal = 10
            normalizeStrike=40
            underlyingsTotal = 15
            putGeometricAverage = Products.Option(timeToMat=1, strike=1, typeOfContract="PutGeometricAverage")
            marketVariables = Products.MarketVariables(r=0.06, dividend=0.0, vol=0.2, spot=[40/normalizeStrike]*underlyingsTotal, correlation=0.25)
            hyperparameters = GridSearch.FNNMC.Hyperparameters(learningRate=10**(-learningrate), inputSize=underlyingsTotal, 
                                    hiddenlayer1=underlyingsTotal+10**2, hiddenlayer2=underlyingsTotal+10**2, hiddenlayer3=underlyingsTotal+10**2, hiddenlayer4=underlyingsTotal+10**2, 
                                    epochs=10**4, batchSize=2**batchsize, trainOnlyLastTimeStep=False, patience=patience)
            timeRegressionStart = time.time()
            GridSearch.FNNMC.findNeuralNetworkModels(simulatedPaths=learningPaths, Option=putGeometricAverage, MarketVariables=marketVariables, hyperparameters=hyperparameters)
            timeRegressionEnd = time.time()
            print(f"Time taken for find regressioncoefficients: {timeRegressionEnd-timeRegressionStart:f}")
            estimates = np.zeros(100)
            for i in range(100):
                # create empirical estimations
                g = h5py.File(os.path.join("GridSearch", "Data", "GeometricPut", "PricePaths15Asset", f"PricePath{i}.hdf5"), "r")
                pricingPaths = g['RND'][...]
                timePriceStart = time.time()
                price = GridSearch.FNNMC.priceAmericanOption(simulatedPaths=pricingPaths, Option=putGeometricAverage, MarketVariables=marketVariables, hyperparameters=hyperparameters)*normalizeStrike
                timePriceEnd = time.time()
                #print(f"Time taken for Pricing: {timePriceEnd-timePriceStart:f}")
                #print(f"The estimated price is: {price:f} and the true price is: 1.119")
                estimates[i]=price
            print("patience: ", patience, " BatchSize: ", batchsize, " Learning rate: ", learningrate ," Mean: ", np.mean(estimates))
            print("Std Error Mean: ", np.std(estimates)/np.sqrt(100))
