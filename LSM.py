#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The script implement the longstaff-schwartz algorithm for pricing american options
"""

import numpy as np
from numpy.lib import stride_tricks
from numpy.random import Generator
from numpy.random import PCG64

rng = Generator(PCG64(123)) #set seed and get generator object

class MarketVariables:
    # The object holds the "observable" market variables
    def __init__(self, r, vol, spot):
        self.r = r
        self.vol = vol
        self.spot = spot

class Option:
    #Give the option to price either a call or put option
    def __init__(self, strike, payoffType):
        self.strike = strike
        self.payoffType = payoffType
    def payoff(self, underlyingPrice: float):
        if(self.payoffType=="Call"):
            return np.maximum(0,underlyingPrice-self.strike)
        elif(self.payoffType=="Put"):
            return np.maximum(0,self.strike-underlyingPrice)
        else:
            print("Invalid call to function, try check your spelling of Call or Put.")
            return 0

##########
# Simulation
##########
def simulateGaussianRandomVariables(pathTotal, timeStepsTotal):
    #simulate the gaussian random variables for finding the coefficients of the regression
    return rng.standard_normal(size=(pathTotal,timeStepsTotal))

def generateTimeStepStock(timeIncrement, rNorm, MarketVariables, previousPrice):
    #Use black-scholes transition probability sampling to make one time step for underlying stock
    return previousPrice*np.exp(MarketVariables.r*timeIncrement-timeIncrement*np.square(MarketVariables.vol)*0.5+np.sqrt(timeIncrement)*MarketVariables.vol*rNorm)

def generateSDEStockPaths(pathTotal, timeStepsTotal, timeToMat, MarketVariables):
    #Transform the simulations of gaussian random variables to paths of the underlying asset(s)
    rNorm = simulateGaussianRandomVariables(pathTotal, timeStepsTotal)
    paths = np.empty((pathTotal,timeStepsTotal+1))
    paths[:,0] = MarketVariables.spot
    timeIncrement = timeToMat/timeStepsTotal
    for timeStep in range(1,timeStepsTotal+1):
        paths[:,timeStep] = generateTimeStepStock(timeIncrement, rNorm[:,timeStep-1], MarketVariables, paths[:,timeStep-1])
    return paths

##########
# Regression Phase
##########

def findRegressionCoefficient(paths, timeStepsTotal, basisFuncTotal, Option):
    # Go backward recursively to find the regression coefficients all the way back to T_1
    # and then store all the regression coefficients
    coefficientMatrix = np.empty((basisFuncTotal+1,timeStepsTotal))
    for timeStep in range(timeStepsTotal,0,-1):
        if(timeStep==timeStepsTotal):
            paths[:,timeStep] = Option.payoff(paths[:,timeStep])
        else:
            coefficientMatrix[:,timeStep]=np.polyfit(paths[:,timeStep], paths[:,timeStep+1], basisFuncTotal)
            paths[:,timeStep] = Option.payoff(paths[:,timeStep])
    return coefficientMatrix

#########
# Pricing Phase
#########

# Simulate a new set a path for pricing
# Use the new path to price the american option


callOption = Option(strike=40,payoffType="Call")
MarketVariablesEx1 = MarketVariables(r=0.06,vol=0.02, spot=40)
stockMatrix = generateSDEStockPaths(pathTotal=100, timeStepsTotal=50, timeToMat=1, MarketVariables=MarketVariablesEx1)
regressionCoefficient = findRegressionCoefficient(paths=stockMatrix, timeStepsTotal=50, basisFuncTotal=2, Option=callOption)

2+2


