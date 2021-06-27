
#!/usr/bin/env python
# PyTorch 1.8.1-CPU virtual env.
# Python 3.9.4 Windows 10
# -*- coding: utf-8 -*-

"""The script implement the classical longstaff-schwartz algorithm for pricing american options.
This script focus on the multidimensional case for rainbow option
"""
import numpy as np

#########
# Learning
#########
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def generateDesignMatrix(currentSpots):
    #add intercept
    transformer = PolynomialFeatures(degree=1, interaction_only=False, include_bias=False)
    transformer.fit(currentSpots)
    basisfunctions = transformer.transform(currentSpots)
    #basisfunctions = np.append(basisfunctions, np.power(basisfunctions[:,1],2).reshape(-1,1), axis=1)
    #basisfunctions = np.append(basisfunctions, np.power(basisfunctions[:,2],2).reshape(-1,1), axis=1)
    #basisfunctions = np.append(basisfunctions, np.power(basisfunctions[:,1],3).reshape(-1,1), axis=1)
    #basisfunctions = np.append(basisfunctions, np.power(basisfunctions[:,2],3).reshape(-1,1), axis=1)
    return basisfunctions


def findRegressionCoefficient(simulatedPaths, basisFuncTotal, Option, MarketVariables):
    # Go backward recursively to find the regression coefficients all the way back to T_1
    # and then store all the regression coefficients
    timeStepsTotal = np.shape(simulatedPaths)[0]-1
    pathsTotal = np.shape(simulatedPaths)[1]
    coefficientMatrix = np.zeros((basisFuncTotal,timeStepsTotal))
    ValueMatrix=np.zeros((pathsTotal, timeStepsTotal+1))

    timeIncrement = Option.timeToMat/(timeStepsTotal)
    for timeStep in range(timeStepsTotal,0,-1):
        #Get payoff at maturity
        if(timeStep== (timeStepsTotal) ):
            ValueMatrix[:,timeStep] = Option.payoff(simulatedPaths[timeStep,:,:])
        #Find regressionscoefficients at each exercise dates before maturity
        else:
            response = np.exp(-MarketVariables.r*timeIncrement)*ValueMatrix[:,timeStep+1]
            currentSpots = simulatedPaths[timeStep,:,:]
            covariates = generateDesignMatrix(currentSpots)
            pathsITM = np.where(Option.payoff(currentSpots)>0)
            if(np.shape(pathsITM)[1]):
                regressionFit = LinearRegression().fit(covariates[pathsITM],response[pathsITM])
                coefficientMatrix[:,timeStep]= np.insert(regressionFit.coef_, 0, regressionFit.intercept_)
                expectedContinuationValue = regressionFit.predict(covariates)
                intrinsicValue = Option.payoff(currentSpots)

                ValueMatrix[:,timeStep] = response
                #overwrite the default to keep the option alive, if it is beneficial to keep the exercise for the ITM paths.
                #CashFlow from decision wheather to stop or keep option alive
                cashFlowChoice = np.where(intrinsicValue>expectedContinuationValue, intrinsicValue, response)
                ValueMatrix[:,timeStep][pathsITM] = cashFlowChoice[pathsITM]
            else:
                coefficientMatrix[:,timeStep]= 0
                ValueMatrix[:,timeStep] = response

    return coefficientMatrix

#########
# Pricing
#########
def priceAmericanOption(coefficientMatrix, simulatedPaths, Option, MarketVariables):
    timeStepsTotal = simulatedPaths.shape[0]-1 #time 0 does not count to a timestep
    timeIncrement = Option.timeToMat/timeStepsTotal
    ValueMatrix=np.zeros((simulatedPaths.shape[1], timeStepsTotal+1))
    for timeStep in range(timeStepsTotal,0,-1):
        #Get payoff at maturity
        if(timeStep==timeStepsTotal):
            ValueMatrix[:,timeStep] = Option.payoff(simulatedPaths[timeStep,:,:])
        #Use coefficientMatrix and paths to price american option 
        else:
            continuationValue = np.exp(-MarketVariables.r*timeIncrement)*ValueMatrix[:,timeStep+1]
            currentSpots = simulatedPaths[timeStep,:,:]
            covariates = generateDesignMatrix(currentSpots)
            expectedContinuationValue = np.matmul(covariates, coefficientMatrix[1:,timeStep])+ coefficientMatrix[0,timeStep]
            intrinsicValue = Option.payoff(currentSpots)
            #overwrite the default to keep the option alive, if it is beneficial to keep the exercise for the ITM paths.
            ValueMatrix[:,timeStep] = continuationValue #default value to keep option alive
            cashFlowChoice = np.where(intrinsicValue>expectedContinuationValue, intrinsicValue, continuationValue)

            pathsITM = np.where(intrinsicValue>0)
            ValueMatrix[:,timeStep][pathsITM] = cashFlowChoice[pathsITM]

    return ValueMatrix[:,1].mean()*np.exp(-MarketVariables.r*timeIncrement)

import Products
import SimulationPaths.GBMMultiDim
import time

if __name__ == '__main__':
    timeStepsTotal = 9
    normalizeStrike=100
    pathTotal = 10**5
    callMax = Products.Option(timeToMat=3, strike=1, typeOfContract="CallMax")
    marketVariables = Products.MarketVariables(r=0.05, dividend=0.1, vol=0.2, spot=[100/normalizeStrike,100/normalizeStrike], correlation=0.0)
    timeSimPathsStart = time.time()
    learningPaths = SimulationPaths.GBMMultiDim.simulatePaths(timeStepsTotal=timeStepsTotal,pathsTotal=pathTotal, marketVariables=marketVariables, timeToMat=callMax.timeToMat)
    timeSimPathsEnd = time.time()
    print(f"Time taken to simulate paths is: {timeSimPathsEnd-timeSimPathsStart:f}")

    timeRegressionStart = time.time()
    coefMatrix = findRegressionCoefficient(simulatedPaths=learningPaths, basisFuncTotal=7, Option=callMax, MarketVariables=marketVariables)
    timeRegressionEnd = time.time()
    print(f"Time taken for find regressioncoefficients: {timeRegressionEnd-timeRegressionStart:f}")
    pricingPaths = SimulationPaths.GBMMultiDim.simulatePaths(timeStepsTotal=timeStepsTotal,pathsTotal=pathTotal, marketVariables=marketVariables, timeToMat=callMax.timeToMat)
    timePriceStart = time.time()
    price = priceAmericanOption(coefMatrix,pricingPaths,callMax, marketVariables)*normalizeStrike
    timePriceEnd = time.time()
    print(f"Time taken for Pricing: {timePriceEnd-timePriceStart:f}")
    print(f"The estimated price is: {price:f} and the true price is: 13.9")