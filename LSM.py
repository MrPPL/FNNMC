#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The script implement the longstaff-schwartz algorithm for pricing american options
"""
import numpy as np
import time
import numba
import SimulationPaths.GBM

try:
    @profile
    def f(x): return x
except:
    def profile(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner

class MarketVariables:
    # The object holds the "observable" market variables
    def __init__(self, r=0, vol=0, spot=0, correlation=0):
        self.r = r
        self.vol = vol
        self.spot = spot
        self.correlation = correlation

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
            response = Option.payoff(simulatedPaths[:,timeStep])
        #Find regressionscoefficients at each exercise dates before maturity
        else:
            response = np.exp(-MarketVariables.r*timeIncrement)*response
            covariates = simulatedPaths[:,timeStep]
            pathsITM = np.where(Option.payoff(covariates)>0)
            if(np.shape(pathsITM)[1]):
                regressionFit = np.polyfit(covariates[pathsITM],response[pathsITM], basisFuncTotal)
                coefficientMatrix[:,timeStep]= regressionFit
                expectedContinuationValue = np.polyval(regressionFit,covariates)
                intrinsicValue = Option.payoff(simulatedPaths[:,timeStep])
                #overwrite the default to keep the option alive, if it is beneficial to keep the exercise for the ITM paths.
                #CashFlow from decision wheather to stop or keep option alive
                cashFlowChoice = np.where(intrinsicValue>expectedContinuationValue, intrinsicValue, response)
                response[pathsITM] = cashFlowChoice[pathsITM]
            else:
                coefficientMatrix[:,timeStep]= 0
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

##########################
# Testing
##########################
if __name__ == '__main__':
    #Price American Put
    timeStepsPerYear = 50
    normalizeStrike=40
    putOption = Option(strike=1,payoffType="Put", timeToMat=1)
    MarketVariablesEx1 = MarketVariables(r=0.06,vol=0.2, spot=36/normalizeStrike)
    pathTotal = 10**5
    timeSimulationStart = time.time()
    learningPaths= SimulationPaths.GBM.generateSDEStockPaths(pathTotal=pathTotal, timeStepsPerYear=timeStepsPerYear, timeToMat=putOption.timeToMat, MarketVariables=MarketVariablesEx1)
    timeSimulationEnd = time.time()
    print(f"Time taken for simulation {timeSimulationEnd-timeSimulationStart:f}")
    timeRegressionStart = time.time()
    regressionCoefficient = findRegressionCoefficient(basisFuncTotal=5, Option=putOption, simulatedPaths=learningPaths, MarketVariables=MarketVariablesEx1)
    timeRegressionEnd = time.time()
    print(f"Time taken for regression {timeRegressionEnd-timeRegressionStart:f}")
    pricingPaths= SimulationPaths.GBM.generateSDEStockPaths(pathTotal=pathTotal, timeStepsPerYear=timeStepsPerYear, timeToMat=putOption.timeToMat, MarketVariables=MarketVariablesEx1)
    timePriceStart = time.time()
    priceAmerPut = priceAmericanOption(coefficientMatrix=regressionCoefficient, Option=putOption , simulatedPaths=pricingPaths, MarketVariables=MarketVariablesEx1)*normalizeStrike
    timePriceEnd = time.time()
    print(f"Time taken for Pricing {timePriceEnd-timePriceStart:f}")
    print(f"Spot: {40:3d} and the price American Put: {priceAmerPut:f}")

    #for spot1 in range(36,46,2):
    #    MarketVariablesEx1 = MarketVariables(r=0.06,vol=0.2, spot=spot1/normalizeStrike)
    #    learningPaths= generateSDEStockPaths(pathTotal=10**5, timeStepsTotal=timeStepsTotal, timeToMat=putOption.timeToMat, MarketVariables=MarketVariablesEx1)
    #    regressionCoefficient = findRegressionCoefficient(basisFuncTotal=5, Option=putOption, simulatedPaths=learningPaths, MarketVariables=MarketVariablesEx1)
    #    pricingPaths= generateSDEStockPaths(pathTotal=10**5, timeStepsTotal=timeStepsTotal, timeToMat=putOption.timeToMat, MarketVariables=MarketVariablesEx1)
    #    priceAmerPut = priceAmericanOption(coefficientMatrix=regressionCoefficient, Option=putOption , simulatedPaths=pricingPaths, MarketVariables=MarketVariablesEx1)*normalizeStrike
    #    print("Spot: ", spot1, "Price American Put: ", priceAmerPut)
        

    #Price American call aka european call
    #timeStepsTotal = 50
    #normalizeStrike=40
    #putOption = Option(strike=1,payoffType="Call", timeToMat=1)
    #MarketVariablesEx1 = MarketVariables(r=0.06,vol=0.2, spot=1)
    #learningPaths= generateSDEStockPaths(pathTotal=10**5, timeStepsTotal=timeStepsTotal, timeToMat=putOption.timeToMat, MarketVariables=MarketVariablesEx1)
    #regressionCoefficient = findRegressionCoefficient(basisFuncTotal=2, Option=putOption, simulatedPaths=learningPaths, MarketVariables=MarketVariablesEx1)
    #pricingPaths= generateSDEStockPaths(pathTotal=10**4, timeStepsTotal=timeStepsTotal, timeToMat=putOption.timeToMat, MarketVariables=MarketVariablesEx1)
    #priceAmerCall = priceAmericanOption(coefficientMatrix=regressionCoefficient, Option=putOption , simulatedPaths=pricingPaths, MarketVariables=MarketVariablesEx1)*normalizeStrike
    #print("Spot: ", MarketVariablesEx1.spot*normalizeStrike, "American Call Price: ", priceAmerCall)