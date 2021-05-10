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

def findRegressionCoefficient(basisFuncTotal, Option, pathTotal, timeStepsTotal, MarketVariables):
    # Go backward recursively to find the regression coefficients all the way back to T_1
    # and then store all the regression coefficients
    paths = generateSDEStockPaths(pathTotal, timeStepsTotal, Option.timeToMat, MarketVariables)
    coefficientMatrix = np.empty((basisFuncTotal+1,timeStepsTotal))
    timeIncrement = Option.timeToMat/timeStepsTotal
    for timeStep in range(timeStepsTotal,-1,-1):
        #Get payoff at maturity
        if(timeStep==timeStepsTotal):
            paths[:,timeStep] = Option.payoff(paths[:,timeStep])
        #Find regressionscoefficients
        else:
            response = np.exp(-MarketVariables.r*timeIncrement)*paths[:,timeStep+1]
            covariates = paths[:,timeStep]
            pathsITM = np.where(Option.payoff(covariates)>0)
            if(pathsITM[0].size):
                regressionFit = np.polyfit(covariates[pathsITM],response[pathsITM], basisFuncTotal)
                coefficientMatrix[:,timeStep]= regressionFit
                continuationValue = np.polyval(regressionFit,covariates)
                intrinsicValue = Option.payoff(paths[:,timeStep])
                #CashFlow from decision wheather to stop or keep option alive
                cashFlowChoice = np.where(continuationValue>intrinsicValue, response,intrinsicValue)
                paths[:,timeStep] = response
                #overwrite the default keep the option alive, if it is beneficial to exercise for the ITM paths.
                paths[:,timeStep][pathsITM] = cashFlowChoice[pathsITM]

            else:
                coefficientMatrix[:,timeStep]= 0
                paths[:,timeStep] = response

    return coefficientMatrix

#########
# Pricing Phase
#########

def priceAmericanOption(coefficientMatrix, pathTotal, timeStepsTotal, MarketVariables, Option):
    # Simulate a new set a path for pricing
    paths = generateSDEStockPaths(pathTotal, timeStepsTotal, Option.timeToMat, MarketVariables)
    timeIncrement = Option.timeToMat/timeStepsTotal
    for timeStep in range(timeStepsTotal,-1,-1):
        #Get payoff at maturity
        if(timeStep==timeStepsTotal):
            paths[:,timeStep] = Option.payoff(paths[:,timeStep])
        #Use coefficientMatrix and paths to price american option 
        else:
            continuationValue = np.exp(-MarketVariables.r*timeIncrement)*paths[:,timeStep+1]
            covariates = paths[:,timeStep]
            expectedContinuationValue = np.polyval(coefficientMatrix[:,timeStep], covariates)
            intrinsicValue = Option.payoff(paths[:,timeStep])
            paths[:,timeStep]= np.where(intrinsicValue>expectedContinuationValue, intrinsicValue, continuationValue)
    return paths[:,0].mean()

################
# calling functions
################
timeStepsTotal = 50
putOption = Option(strike=40,payoffType="Put", timeToMat=1)
MarketVariablesEx1 = MarketVariables(r=0.06,vol=0.2, spot=40)
regressionCoefficient = findRegressionCoefficient(basisFuncTotal=3, Option=putOption, pathTotal=10^5, timeStepsTotal=timeStepsTotal, MarketVariables=MarketVariablesEx1)
priceAmer = priceAmericanOption(coefficientMatrix=regressionCoefficient, pathTotal=1000, timeStepsTotal=timeStepsTotal, MarketVariables=MarketVariablesEx1, Option=putOption)
print(priceAmer)


