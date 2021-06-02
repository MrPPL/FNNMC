#!/usr/bin/env python
# PyTorch 1.8.1-CPU virtual env.
# Python 3.9.4 Windows 10
# -*- coding: utf-8 -*-

"""The script implement the classical longstaff-schwartz algorithm for pricing american options.
This script focus on the multidimensional case for rainbow option
"""
import numpy as np
from numpy.random import Generator
from numpy.random import PCG64
rng = Generator(PCG64(123)) #set seed and get generator object

#########
# Classes
#########
class MarketVariables:
    # The object holds the "observable" market variables
    def __init__(self, r=0, vol=0, spots=[0], correlation=0):
        self.r = r
        self.vol = vol
        self.spots = spots
        self.correlation = correlation

#########
# Functions for simulate paths
#########
def initialState(pathsTotal, spots, assetsTotal):
    """Args:
        pathsTotal ([int]): [Total number of simulated paths]
        spots ([vector of doubles]): [The current price of the underlying stocks]
        assetsTotal ([int]): [Total number of underlying assets]

    Returns:
        [2D Matrix]: [The initial state for markov chain at time 0]
    """
    spotMatrix = np.zeros((pathsTotal, assetsTotal))
    for assetIndex in range(assetsTotal):
        spotMatrix[:,assetIndex] = spots[assetIndex]
    return spotMatrix

def GBMUpdate(spot, marketVariables, timeIncrement, lowerTriangleMatrixRow):
    #Calculate the next step for stock
    rNormVec = rng.standard_normal(len(lowerTriangleMatrixRow))
    drift = (marketVariables.r-marketVariables.vol/2)*timeIncrement
    volTerm = np.sqrt(timeIncrement) * np.dot(lowerTriangleMatrixRow,rNormVec)
    return spot*np.exp(drift+volTerm)

def generateCovarianceMatrix(assetsTotal, vol, correlation):
    covarianceMatrix = np.zeros((assetsTotal, assetsTotal))
    covarianceMatrix[:,:] = correlation*vol**2
    np.fill_diagonal(covarianceMatrix,vol**2)
    return covarianceMatrix

def updateState(currentState, marketVariables, timeIncrement):
    """Update the state of markov chain to next timestep for all paths.

    Args:
        currentState ([2D Matrix]): [Describes the state of markov chain before time-step update]
    Returns:
        [2D Matrix]: [Describes the updated state of markov chain at current time-step]
    """
    newState = np.zeros(currentState.shape)
    pathsTotal = currentState.shape[0]
    assetsTotal = currentState.shape[1]
    covarianceMatrix = generateCovarianceMatrix(assetsTotal, marketVariables.vol, marketVariables.correlation)
    lowerTriangleMatrix = np.linalg.cholesky(covarianceMatrix)
    for path in range(pathsTotal):
        for assetIndex in range(assetsTotal):
            newState[path,assetIndex] = GBMUpdate(spot=currentState[path,assetIndex], marketVariables=marketVariables, timeIncrement=timeIncrement, 
                lowerTriangleMatrixRow=lowerTriangleMatrix[assetIndex,:])
    return newState


def simulatePaths(timeStepsTotal, pathsTotal, marketVariables, timeToMat):
    """[Simulates the paths needed for pricing American multivariate contingent claims]

    Args:
        timeStepsTotal ([int]): [Total number of exercise dates]
        pathsTotal ([int]): [Total number of simulated paths]
        marketVariables ([Object]): [Object containing the market variables]
        timeToMat ([doubles]): [Time to maturity of option]
    Returns:
        [3D Matrix: [A matrix contining all the simulated paths]
    """
    assetsTotal = len(marketVariables.spots)
    pathMatrix = np.zeros((timeStepsTotal,pathsTotal, assetsTotal))
    timeIncrement = timeToMat/timeStepsTotal
    for timeStep in range(timeStepsTotal):
        if timeStep==0:
            pathMatrix[timeStep] = initialState(pathsTotal, marketVariables.spots, assetsTotal)
        else:
            pathMatrix[timeStep] = updateState(pathMatrix[timeStep-1], marketVariables, timeIncrement)
    return pathMatrix


if __name__ == '__main__':
    marketVariables = MarketVariables(r=0.03, vol=0.2, spots=[40,50, 10], correlation=0.2)
    learningPaths = simulatePaths(timeStepsTotal=50,pathsTotal=10**3, marketVariables=marketVariables, timeToMat=1)
    pass
