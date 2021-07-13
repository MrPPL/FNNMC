#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The script implement the longstaff-schwartz algorithm for pricing american options for univariate claims
"""
import numpy as np
import time
import numba
import SimulationPaths.GBM
import Products

try:
    @profile
    def f(x): return x
except:
    def profile(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner


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
            continuationValue = Option.payoff(simulatedPaths[:,timeStep])
        #Use coefficientMatrix and paths to price american option 
        else:
            continuationValue = np.exp(-MarketVariables.r*timeIncrement)*continuationValue
            covariates = simulatedPaths[:,timeStep]
            expectedContinuationValue = np.polyval(coefficientMatrix[:,timeStep], covariates)
            intrinsicValue = Option.payoff(simulatedPaths[:,timeStep])
            #overwrite the default to keep the option alive, if it is beneficial to keep the exercise for the ITM paths.
            cashFlowChoice = np.where(intrinsicValue>expectedContinuationValue, intrinsicValue, continuationValue)

            pathsITM = np.where(intrinsicValue>0)
            continuationValue[pathsITM] = cashFlowChoice[pathsITM]

    return continuationValue.mean()*np.exp(-MarketVariables.r*timeIncrement)

##########################
# Testing execution
##########################
if __name__ == '__main__':
    #Price American Put
    timeStepsPerYear = 10
    normalizeStrike=40
    spot = 36
    putOption = Products.Option(timeToMat=1, strike=1,typeOfContract="Put")
    MarketVariablesEx1 = Products.MarketVariables(r=0.06,vol=0.2, spot=spot/normalizeStrike, dividend=0.0)
    pathTotal = 10**4
    timeSimulationStart = time.time()
    learningPaths= SimulationPaths.GBM.generateSDEStockPaths(pathTotal=10**6, timeStepsPerYear=timeStepsPerYear, timeToMat=putOption.timeToMat, MarketVariables=MarketVariablesEx1)
    timeSimulationEnd = time.time()
    print(f"Time taken for simulation {timeSimulationEnd-timeSimulationStart:f}")
    timeRegressionStart = time.time()
    regressionCoefficient = findRegressionCoefficient(basisFuncTotal=3, Option=putOption, simulatedPaths=learningPaths, MarketVariables=MarketVariablesEx1)
    timeRegressionEnd = time.time()
    print(f"Time taken for regression {timeRegressionEnd-timeRegressionStart:f}")
    repeat =100
    estimates = np.zeros(repeat)
    for i in range(repeat):
        # create empirical estimations
        pricingPaths= SimulationPaths.GBM.generateSDEStockPaths(pathTotal=pathTotal, timeStepsPerYear=timeStepsPerYear, timeToMat=putOption.timeToMat, MarketVariables=MarketVariablesEx1)
        timePriceStart = time.time()
        priceAmerPut = priceAmericanOption(coefficientMatrix=regressionCoefficient, Option=putOption , simulatedPaths=pricingPaths, MarketVariables=MarketVariablesEx1)*normalizeStrike
        timePriceEnd = time.time()
        print(f"Time taken for Pricing {timePriceEnd-timePriceStart:f}")
        print(f"Spot: {spot:3d} and the price American Put: {priceAmerPut:f}")
        estimates[i]=priceAmerPut
    print("Mean: ", np.mean(estimates))
    print("Std Error Mean: ", np.std(estimates)/np.sqrt(repeat))

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