#!/usr/bin/env python
# PyTorch 1.8.1-CPU virtual env.
# Python 3.9.4 Windows 10
# -*- coding: utf-8 -*-

"""The script implement the classical longstaff-schwartz algorithm for pricing american options.
This script focus on the multidimensional case for rainbow option
"""
import numpy as np
from numpy.random import Generator, seed
from numpy.random import PCG64
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Products
import time
rng = Generator(PCG64(seed=123)) #set seed and get generator object
#rng = Generator(PCG64()) #set seed and get generator object


#########
# Functions for simulate paths
#########
def initialState(pathsTotal, spot, assetsTotal):
    """Args:
        pathsTotal ([int]): [Total number of simulated paths]
        spot ([vector of doubles]): [The current price of the underlying stocks]
        assetsTotal ([int]): [Total number of underlying assets]

    Returns:
        [2D Matrix]: [The initial state for markov chain at time 0]
    """
    spotMatrix = np.zeros((pathsTotal, assetsTotal), order="F")
    for assetIndex in range(assetsTotal):
        spotMatrix[:,assetIndex] = spot[assetIndex]
    return spotMatrix

def GBMUpdate(spot, marketVariables, timeIncrement, lowerTriangleMatrixRow, normVec):
    #Calculate the next step for stock
    drift = (marketVariables.r-marketVariables.dividend-np.square(marketVariables.vol)*0.5)*timeIncrement
    volTerm = np.sqrt(timeIncrement) * np.dot(lowerTriangleMatrixRow, normVec)
    return spot*np.exp(drift+volTerm)

def choleskyLowerTriangular(assetsTotal, vol, correlation):
    """Use cholesky decomposition to generate a lower triuangular matrix
    """
    #Same volatility for each asset
    if (isinstance(vol, (int, float))):
        covarianceMatrix = np.zeros((assetsTotal, assetsTotal))
        covarianceMatrix[:,:] = correlation*vol**2
        np.fill_diagonal(covarianceMatrix,vol**2)
        lowerTriangleMatrix = np.linalg.cholesky(covarianceMatrix)
        return lowerTriangleMatrix
    #Different volatility for the assets
    else:
        correlationMatrix = np.zeros((assetsTotal, assetsTotal))
        diagonalVol = np.diag(vol)
        correlationMatrix[:,:] = correlation
        np.fill_diagonal(correlationMatrix,1)
        covarianceMatrix = np.matmul(diagonalVol, np.matmul(correlationMatrix,diagonalVol) )
        lowerTriangleMatrix = np.linalg.cholesky(covarianceMatrix)
        return lowerTriangleMatrix
        

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
    lowerTriangleMatrix = choleskyLowerTriangular(assetsTotal, marketVariables.vol, marketVariables.correlation)
    for path in range(pathsTotal):
        rNormVec = rng.standard_normal(assetsTotal)
        newState[path,:] = [GBMUpdate(spot=currentState[path,assetIndex], marketVariables=marketVariables, timeIncrement=timeIncrement,
            lowerTriangleMatrixRow=lowerTriangleMatrix[assetIndex,:], normVec=rNormVec) for assetIndex in range(assetsTotal)]

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
    assetsTotal = len(marketVariables.spot)
    pathMatrix = np.zeros((timeStepsTotal+1,pathsTotal, assetsTotal), order="F")
    timeIncrement = timeToMat/timeStepsTotal
    for timeStep in range(timeStepsTotal+1):
        if timeStep==0:
            pathMatrix[timeStep] = initialState(pathsTotal, marketVariables.spot, assetsTotal)
        else:
            pathMatrix[timeStep] = updateState(pathMatrix[timeStep-1], marketVariables, timeIncrement)
    return pathMatrix


if __name__ == '__main__':
    marketVariables = Products.MarketVariables(r=0.03, vol=0.2, spot=[40,50, 10], dividend=0.1, correlation=0.2)
    pathTotal = 10**5
    exerciseDatesTotal=3
    pathSimulationTimeStart = time.time()
    learningPaths = simulatePaths(timeStepsTotal=exerciseDatesTotal,pathsTotal=pathTotal, marketVariables=marketVariables, timeToMat=1)
    pathSimulationTimeEnd = time.time()
    print(f"Time taken to simulate {pathTotal:.2e} paths with {exerciseDatesTotal:d} exercise dates and {len(marketVariables.spot):d} underlyings: {pathSimulationTimeEnd-pathSimulationTimeStart:4.4f}")
    pass
