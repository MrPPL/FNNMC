#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The script implement the longstaff-schwartz algorithm for pricing american options
"""

import numpy as np
from numpy.random import Generator
from numpy.random import PCG64

rng = Generator(PCG64(123)) #set seed and get generator object

def simulateGaussianMatrix(pathTotal, timeStepsTotal):
    #simulate the gaussian random variables for finding the coefficients of the regression
    gaussianMatrix = rng.standard_normal(size=(pathTotal,timeStepsTotal-1))
    return gaussianMatrix

def generateTimeStepStock(spot, r, timeIncrement, vol, rNorm):
    #Use black-scholes transition probability sampling to make one time step for underlying stock
    return spot*np.exp(r*timeIncrement-timeIncrement*np.square(vol)*0.5+np.sqrt(timeIncrement)*vol*rNorm)

def generateSDEStock(spot, r, vol, pathTotal, timeStepsTotal, timeToMat):
    #Transform the simulations of gaussian random variables to paths of the underlying asset(s)
    rNorm = simulateGaussianMatrix(pathTotal, timeStepsTotal)
    paths = np.empty((pathTotal,timeStepsTotal))
    paths[:,0] = spot
    timeIncrement = timeToMat/timeStepsTotal
    for timeStep in range(1,timeStepsTotal):
        paths[:,timeStep] = generateTimeStepStock(paths[:,timeStep-1], r, timeIncrement, vol, rNorm[:,timeStep-1])

generateSDEStock(40, 0.06, 0.02, 3, 4, 1)

#Use the paths to get payoff at maturity


# Go backward recursively to find the regression coefficients all the way back to T_1


# Store the regression coefficients

# Simulate a new set a path for pricing
# Use the new path to price the american option