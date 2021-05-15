#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The script implement the longstaff-schwartz algorithm for pricing american options
"""

import numpy as np

class MarketVariables:
    # The object holds the "observable" market variables
    def __init__(self, r, vol, spot):
        self.r = r
        self.vol = vol
        self.spot = spot

class Option:
    #Give the option to price either a call or put option
    def __init__(self, strike, payoffType, timeToMat):
        self.strike = strike
        self.payoffType = payoffType
        self.timeToMat = timeToMat
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
from numpy.random import Generator
from numpy.random import PCG64

rng = Generator(PCG64(123)) #set seed and get generator object
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

def findRegressionCoefficient(simulatedPaths, basisFuncTotal, Option, MarketVariables):
    # Go backward recursively to find the regression coefficients all the way back to T_1
    # and then store all the regression coefficients
    timeStepsTotal = np.shape(simulatedPaths)[1]-1
    coefficientMatrix = np.empty((basisFuncTotal+1,timeStepsTotal))
    timeIncrement = Option.timeToMat/(timeStepsTotal)
    for timeStep in range(timeStepsTotal,0,-1):
        #Get payoff at maturity
        if(timeStep==timeStepsTotal):
            simulatedPaths[:,timeStep] = Option.payoff(simulatedPaths[:,timeStep])
        #Find regressionscoefficients at each exercise dates before maturity
        else:
            response = np.exp(-MarketVariables.r*timeIncrement)*simulatedPaths[:,timeStep+1]
            covariates = simulatedPaths[:,timeStep]
            pathsITM = np.where(Option.payoff(covariates)>0)
            if(np.shape(pathsITM)[1]):
                regressionFit = np.polyfit(covariates[pathsITM],response[pathsITM], basisFuncTotal)
                coefficientMatrix[:,timeStep]= regressionFit
                expectedContinuationValue = np.polyval(regressionFit,covariates)
                intrinsicValue = Option.payoff(simulatedPaths[:,timeStep])
                simulatedPaths[:,timeStep] = response
                #overwrite the default to keep the option alive, if it is beneficial to keep the exercise for the ITM paths.
                #CashFlow from decision wheather to stop or keep option alive
                cashFlowChoice = np.where(intrinsicValue>expectedContinuationValue, intrinsicValue, response)
                simulatedPaths[:,timeStep][pathsITM] = cashFlowChoice[pathsITM]
            else:
                coefficientMatrix[:,timeStep]= 0
                simulatedPaths[:,timeStep] = response

    return coefficientMatrix

#########
# Pricing Phase
#########

def priceAmericanOption(coefficientMatrix, simulatedPaths, Option, MarketVariables):
    timeStepsTotal = simulatedPaths.shape[1]-1 #time 0 does not count to a timestep
    timeIncrement = Option.timeToMat/timeStepsTotal
    for timeStep in range(timeStepsTotal,0,-1):
        #Get payoff at maturity
        if(timeStep==timeStepsTotal):
            simulatedPaths[:,timeStep] = Option.payoff(simulatedPaths[:,timeStep])
        #Use coefficientMatrix and paths to price american option 
        else:
            continuationValue = np.exp(-MarketVariables.r*timeIncrement)*simulatedPaths[:,timeStep+1]
            covariates = simulatedPaths[:,timeStep]
            expectedContinuationValue = np.polyval(coefficientMatrix[:,timeStep], covariates)
            intrinsicValue = Option.payoff(simulatedPaths[:,timeStep])
            #overwrite the default to keep the option alive, if it is beneficial to keep the exercise for the ITM paths.
            simulatedPaths[:,timeStep] = continuationValue #default value to keep option alive
            cashFlowChoice = np.where(intrinsicValue>expectedContinuationValue, intrinsicValue, continuationValue)

            pathsITM = np.where(intrinsicValue>0)
            simulatedPaths[:,timeStep][pathsITM] = cashFlowChoice[pathsITM]

    return simulatedPaths[:,1].mean()*np.exp(-MarketVariables.r*timeIncrement)
