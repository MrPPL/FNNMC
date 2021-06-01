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
import LSM

def initialState(pathsTotal, underlyings, assetsTotal):
    spotMatrix = np.zeros((pathsTotal, assetsTotal))
    for assetIndex in range(assetsTotal):
        spotMatrix[:,assetIndex] = underlyings[assetIndex]
    return spotMatrix

rng = Generator(PCG64(123)) #set seed and get generator object
def GBMUpdate(spot, r, vol, timeIncrement, lowerTriangleMatrixRow):
    rNormVec = rng.standard_normal(lowerTriangleMatrixRow.shape[1])
    drift = (r-vol/2)*timeIncrement
    volTerm = np.sqrt(timeIncrement) * np.dot(lowerTriangleMatrixRow,rNormVec)
    return spot*np.exp(drift+volTerm)

def updateState(currentState):
    newState = np.zeros(currentState.shape)
    pathsTotal = currentState.shape[0]
    assetsTotal = currentState.shape[1]
    for path in range(pathsTotal):
        for assetIndex in range(assetsTotal):
            newState[path] = currentState[0]


def simulatePaths(timeStepsTotal, pathsTotal, spots):
    assetsTotal = len(spots)
    pathMatrix = np.zeros((timeStepsTotal,pathsTotal, assetsTotal))
    for timeStep in range(timeStepsTotal):
        if timeStep==0:
            pathMatrix[timeStep] = initialState(pathsTotal, spots, assetsTotal)
        else:
            pathMatrix[timeStep] = updateState(pathMatrix[timeStep-1])
    return pathMatrix


if __name__ == '__main__':
    print(simulatePaths(timeStepsTotal=2,pathsTotal=3, underlyings=[40,50]).shape[1])
