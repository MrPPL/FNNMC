#!/usr/bin/env python
# PyTorch 1.8.1-CPU virtual env.
# Python 3.9.4 Windows 10
# -*- coding: utf-8 -*-

"""The script implement the classical longstaff-schwartz algorithm for pricing american options.
The only difference in the regression phase
"""
import numpy as np
from numpy.random import Generator
from numpy.random import PCG64
import LSM

##########
# Contract class
##########
class Option():
    def __init__(self, timeToMaturity=0, strike=0, underlyings=[0]):
         self.timeToMaturity = timeToMaturity
         self.strike = strike
         self.underlyings = underlyings

    def __str__(self):
        return 'Time to Maturity %.d years\nStrike %.d \nValue of underlyings %s' % (self.timeToMaturity, self.strike, self.underlyings)

    def payoffFunc(self, strike, underlyings, typeOfContract):
        #TODO!
        if (typeOfContract=="Put"):
            return np.maximum(0, strike - underlyings)
        elif (typeOfContract=="Call"):
            return np.maximum(0,underlyings - strike)
        elif (typeOfContract=="Geometric Average Put Basket"):
            return np.maximum(0, strike - np.prod(underlyings)**(1/len(underlyings)))
        else:
            print("Invalid input for the payoff function")

##########
# Simulation
##########
rng = Generator(PCG64(123)) #set seed and get generator object
def generateCovarianceMatrix(underlyingsTotal, vol, correlation):
    covarianceMatrix = np.zeros((underlyingsTotal, underlyingsTotal))
    covarianceMatrix[:,:] = correlation*vol**2
    np.fill_diagonal(covarianceMatrix,vol**2)
    return covarianceMatrix

def generateTimeStepMatrix(pathTotal, marketVariables, timeIncrement, underlyingsTotal, previousTimeStepMatrix):
    covarianceMatrix = generateCovarianceMatrix(underlyingsTotal, marketVariables.vol, marketVariables.correlation)
    A = np.linalg.cholesky(covarianceMatrix)
    timeStepMatrix = np.zeros((pathTotal,underlyingsTotal))
    for path in range(pathTotal):
        for underlying in range(underlyingsTotal):
            rNormVec = rng.standard_normal(underlyingsTotal)
            brownianTerm = np.dot(A[underlying,:],rNormVec)*np.sqrt(timeIncrement)
            timeStepMatrix[path,underlying]=previousTimeStepMatrix[path,underlying]*np.exp((marketVariables.r-marketVariables.vol**2/2)*timeIncrement+brownianTerm)
    return timeStepMatrix

def simulatePaths(underlyingsTotal, timeStepsPerYear, pathTotal, marketVariables, yearToMaturity):
    """The method simulate the paths for GBM

    Args:
        underlyingsTotal ([int])
        timeStepsTotal ([int]): [Number of steps to per year approximate American Option]
        pathTotal ([int])
    """
    timeStepsTotal = timeStepsPerYear*yearToMaturity
    timeIncrement = 1/timeStepsPerYear
    for timeStep in range(timeStepsTotal):
        if timeStep==0:
            timeStepDict = dict()
            timeStepDict[timeStep] = np.full((pathTotal, underlyingsTotal),marketVariables.spot) 
        else:
            previousTimeStepMatrix = timeStepDict[timeStep-1]
            timeStepDict[timeStep] = generateTimeStepMatrix(pathTotal, marketVariables, timeIncrement, underlyingsTotal, previousTimeStepMatrix)
    return timeStepDict

marketVariables = LSM.MarketVariables(r=0.06, vol=0.2, spot=40, correlation=0.2)
Dict1 = simulatePaths(underlyingsTotal=2, timeStepsPerYear=50, pathTotal=10**5, marketVariables=marketVariables, yearToMaturity=1)


    


