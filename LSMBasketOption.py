
#!/usr/bin/env python
# PyTorch 1.8.1-CPU virtual env.
# Python 3.9.4 Windows 10
# -*- coding: utf-8 -*-

"""The script implement the classical longstaff-schwartz algorithm for pricing american options.
This script focus on the multidimensional case for rainbow option
"""
import numpy as np
from numpy.core.fromnumeric import shape

#########
# Learning
#########
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def generateDesignMatrix(currentSpots, Option, regressionBasis):
    #add intercept
    transformer = PolynomialFeatures(degree=1, interaction_only=False, include_bias=False)
    transformer.fit(currentSpots)
    basisfunctions = transformer.transform(currentSpots)
    if (regressionBasis=="Base"):
        pass
    elif (regressionBasis=="secondOrderPoly"):
        basisfunctions = np.append(basisfunctions, np.power(basisfunctions[:,0],2).reshape(-1,1), axis=1)
        basisfunctions = np.append(basisfunctions, np.power(basisfunctions[:,1],2).reshape(-1,1), axis=1)
    elif (regressionBasis=="thirdOrderPoly"):
        basisfunctions = np.append(basisfunctions, np.power(basisfunctions[:,0],2).reshape(-1,1), axis=1)
        basisfunctions = np.append(basisfunctions, np.power(basisfunctions[:,1],2).reshape(-1,1), axis=1)
        basisfunctions = np.append(basisfunctions, np.power(basisfunctions[:,1],3).reshape(-1,1), axis=1)
        basisfunctions = np.append(basisfunctions, np.power(basisfunctions[:,2],3).reshape(-1,1), axis=1)
    elif (regressionBasis=="bestConfigCallMax"):
        basisfunctions = np.append(basisfunctions, np.power(basisfunctions[:,0],2).reshape(-1,1), axis=1)
        basisfunctions = np.append(basisfunctions, np.power(basisfunctions[:,1],2).reshape(-1,1), axis=1)
        basisfunctions = np.append(basisfunctions, np.multiply(basisfunctions[:,0], basisfunctions[:,1]).reshape(-1,1), axis=1)
        basisfunctions = np.append(basisfunctions, Option.payoff(basisfunctions[:,:2]).reshape(-1,1), axis=1)

    return (basisfunctions, basisfunctions.shape[1])


def findRegressionCoefficient(simulatedPaths, Option, MarketVariables, regressionBasis):
    # Go backward recursively to find the regression coefficients all the way back to T_1
    # and then store all the regression coefficients
    timeStepsTotal = np.shape(simulatedPaths)[0]-1
    covariatesTotal = generateDesignMatrix(simulatedPaths[0,:,:], Option, regressionBasis)[1] + 1
    coefficientMatrix = np.zeros((covariatesTotal, timeStepsTotal), order="F")

    timeIncrement = Option.timeToMat/(timeStepsTotal)
    for timeStep in range(timeStepsTotal,0,-1):
        #Get payoff at maturity
        if(timeStep== (timeStepsTotal) ):
            ValueVec = Option.payoff(simulatedPaths[timeStep,:,:])
        #Find regressionscoefficients at each exercise dates before maturity
        else:
            response = np.exp(-MarketVariables.r*timeIncrement)*ValueVec
            currentSpots = simulatedPaths[timeStep,:,:]
            covariates = generateDesignMatrix(currentSpots, Option, regressionBasis)[0]
            pathsITM = np.where(Option.payoff(currentSpots)>0)
            if(np.shape(pathsITM)[1]):
                regressionFit = LinearRegression().fit(covariates[pathsITM],response[pathsITM])
                coefficientMatrix[:,timeStep]= np.insert(regressionFit.coef_, 0, regressionFit.intercept_)
                expectedContinuationValue = regressionFit.predict(covariates)
                intrinsicValue = Option.payoff(currentSpots)

                ValueVec = response
                #overwrite the default to keep the option alive, if it is beneficial to keep the exercise for the ITM paths.
                #CashFlow from decision wheather to stop or keep option alive
                cashFlowChoice = np.where(intrinsicValue>expectedContinuationValue, intrinsicValue, response)
                ValueVec[pathsITM] = cashFlowChoice[pathsITM]
            else:
                coefficientMatrix[:,timeStep]= 0
                ValueVec = response

    return coefficientMatrix

#########
# Pricing
#########
def priceAmericanOption(coefficientMatrix, simulatedPaths, Option, MarketVariables, regressionBasis):
    timeStepsTotal = simulatedPaths.shape[0]-1 #time 0 does not count to a timestep
    timeIncrement = Option.timeToMat/timeStepsTotal
    for timeStep in range(timeStepsTotal,0,-1):
        #Get payoff at maturity
        if(timeStep==timeStepsTotal):
            ValueVec = Option.payoff(simulatedPaths[timeStep,:,:])
        #Use coefficientMatrix and paths to price american option 
        else:
            continuationValue = np.exp(-MarketVariables.r*timeIncrement)*ValueVec
            currentSpots = simulatedPaths[timeStep,:,:]
            covariates = generateDesignMatrix(currentSpots, Option, regressionBasis)[0]
            expectedContinuationValue = np.matmul(covariates, coefficientMatrix[1:,timeStep])+ coefficientMatrix[0,timeStep]
            intrinsicValue = Option.payoff(currentSpots)
            #overwrite the default to keep the option alive, if it is beneficial to keep the exercise for the ITM paths.
            ValueVec = continuationValue #default value to keep option alive
            cashFlowChoice = np.where(intrinsicValue>expectedContinuationValue, intrinsicValue, continuationValue)

            pathsITM = np.where(intrinsicValue>0)
            ValueVec[pathsITM] = cashFlowChoice[pathsITM]

    return ValueVec.mean()*np.exp(-MarketVariables.r*timeIncrement)

import Products
import SimulationPaths.GBMMultiDim
import time

if __name__ == '__main__':
    timeStepsTotal = 9
    normalizeStrike=100
    pathTotal = 10**1
    callMax = Products.Option(timeToMat=3, strike=1, typeOfContract="CallGeometricAverage")
    marketVariables = Products.MarketVariables(r=0.05, dividend=0.1, vol=0.2, spot=[100/normalizeStrike,100/normalizeStrike], correlation=0.0)

    # create empirical estimations
    timeSimPathsStart = time.time()
    learningPaths = SimulationPaths.GBMMultiDim.simulatePaths(timeStepsTotal=timeStepsTotal,pathsTotal=pathTotal, marketVariables=marketVariables, timeToMat=callMax.timeToMat)
    timeSimPathsEnd = time.time()
    print(f"Time taken to simulate paths is: {timeSimPathsEnd-timeSimPathsStart:f}")

    timeRegressionStart = time.time()
    coefMatrix = findRegressionCoefficient(simulatedPaths=learningPaths, Option=callMax, MarketVariables=marketVariables, regressionBasis="Base")
    timeRegressionEnd = time.time()
    print(f"Time taken for find regressioncoefficients: {timeRegressionEnd-timeRegressionStart:f}")
    pricingPaths = SimulationPaths.GBMMultiDim.simulatePaths(timeStepsTotal=timeStepsTotal,pathsTotal=pathTotal, marketVariables=marketVariables, timeToMat=callMax.timeToMat)
    timePriceStart = time.time()
    price = priceAmericanOption(coefMatrix,pricingPaths,callMax, marketVariables, "Base")*normalizeStrike
    timePriceEnd = time.time()
    print(f"Time taken for Pricing: {timePriceEnd-timePriceStart:f}")
    print(f"The estimated price is: {price:f} and the true price is: 13.9")