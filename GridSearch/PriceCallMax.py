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
for patience in range(6,11,1):
    for learningrate in range(2,7,1):
        for batchsize in range(5,11,1):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            # load data
            f = h5py.File(".\Gridsearch\Data\MaxCall\1MCallMax2Assets.hdf5", "r")
            learningPaths = f['RND'][...]

            timeStepsTotal = 9
            normalizeStrike=100
            callMax = Products.Option(timeToMat=3, strike=1, typeOfContract="CallMax")
            underlyingsTotal = 2
            marketVariables = Products.MarketVariables(r=0.05, dividend=0.1, vol=0.2, spot=[100/normalizeStrike]*underlyingsTotal, correlation=0.0)

            hyperparameters = GridSearch.FNNMC.Hyperparameters(learningRate=10**(-learningrate), inputSize=underlyingsTotal, 
                                    hiddenlayer1=underlyingsTotal+10**2, hiddenlayer2=underlyingsTotal+10**2, hiddenlayer3=underlyingsTotal+10**2, hiddenlayer4=underlyingsTotal+10**2, 
                                    epochs=10**4, batchSize=2**batchsize, trainOnlyLastTimeStep=False, patience=patience)
            timeRegressionStart = time.time()
            GridSearch.FNNMC.findNeuralNetworkModels(simulatedPaths=learningPaths, Option=callMax, MarketVariables=marketVariables, hyperparameters=hyperparameters)
            timeRegressionEnd = time.time()
            print(f"Time taken for find regressioncoefficients: {timeRegressionEnd-timeRegressionStart:f}")
            estimates = np.zeros(100)
            for i in range(100):
                # create empirical estimations
                g = h5py.File(f"GridSearch\Data\MaxCall\PricePaths2Asset\PricePath{i}.hdf5", "r")
                pricingPaths = g['RND'][...]
                timePriceStart = time.time()
                price = GridSearch.FNNMC.priceAmericanOption(simulatedPaths=pricingPaths, Option=callMax, MarketVariables=marketVariables, hyperparameters=hyperparameters)*normalizeStrike
                timePriceEnd = time.time()
                estimates[i]=price
            print("patience: ", patience, " BatchSize: ", batchsize, " Learning rate: ", learningrate ," Mean: ", np.mean(estimates))
            print("Std Error Mean: ", np.std(estimates)/10)
