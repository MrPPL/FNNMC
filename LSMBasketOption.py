
#!/usr/bin/env python
# PyTorch 1.8.1-CPU virtual env.
# Python 3.9.4 Windows 10
# -*- coding: utf-8 -*-

"""The script implement the classical longstaff-schwartz algorithm for pricing american options.
This script focus on the multidimensional case for rainbow option
"""
import numpy as np
import SimGBMMultidimensions
#########
# Classes
#########

class RegressionModel():
    def linearRegression(self):
        pass

#########
# Learning
#########
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def generateDesignMatrix(currentSpots):
    #add intercept
    transformer = PolynomialFeatures(degree=1, interaction_only=False, include_bias=True)
    transformer.fit(currentSpots)
    basisfunctions = transformer.transform(currentSpots)
    basisfunctions = np.append(basisfunctions, np.power(basisfunctions[:,1],2).reshape(-1,1), axis=1)
    basisfunctions = np.append(basisfunctions, np.power(basisfunctions[:,2],2).reshape(-1,1), axis=1)
    basisfunctions = np.append(basisfunctions, np.power(basisfunctions[:,1],3).reshape(-1,1), axis=1)
    basisfunctions = np.append(basisfunctions, np.power(basisfunctions[:,2],3).reshape(-1,1), axis=1)
    return basisfunctions


def findRegressionCoefficient(simulatedPaths, basisFuncTotal, Option, MarketVariables):
    # Go backward recursively to find the regression coefficients all the way back to T_1
    # and then store all the regression coefficients
    timeStepsTotal = np.shape(simulatedPaths)[0]-1
    pathsTotal = np.shape(simulatedPaths)[1]
    coefficientMatrix = np.zeros((basisFuncTotal,timeStepsTotal))
    ValueMatrix=np.zeros((pathsTotal, timeStepsTotal+1))

    timeIncrement = Option.timeToMaturity/(timeStepsTotal)
    for timeStep in range(timeStepsTotal,0,-1):
        #Get payoff at maturity
        if(timeStep==timeStepsTotal):
            ValueMatrix[:,timeStep] = Option.payoff(simulatedPaths[timeStep,:,:])
        #Find regressionscoefficients at each exercise dates before maturity
        else:
            response = np.exp((-MarketVariables.r)*timeIncrement)*ValueMatrix[:,timeStep+1]
            currentSpots = simulatedPaths[timeStep,:,:]
            covariates = generateDesignMatrix(currentSpots)
            pathsITM = np.where(Option.payoff(currentSpots)>0)
            if(np.shape(pathsITM)[1]):
                regressionFit = LinearRegression().fit(covariates[pathsITM],response[pathsITM])
                coefficientMatrix[:,timeStep]= regressionFit.coef_
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
    timeIncrement = Option.timeToMaturity/timeStepsTotal
    ValueMatrix=np.zeros((simulatedPaths.shape[1], timeStepsTotal+1))
    for timeStep in range(timeStepsTotal,0,-1):
        #Get payoff at maturity
        if(timeStep==timeStepsTotal):
            ValueMatrix[:,timeStep] = Option.payoff(simulatedPaths[timeStep,:,:])
        #Use coefficientMatrix and paths to price american option 
        else:
            continuationValue = np.exp((-MarketVariables.r)*timeIncrement)*ValueMatrix[:,timeStep+1]
            currentSpots = simulatedPaths[timeStep,:,:]
            covariates = generateDesignMatrix(currentSpots)
            expectedContinuationValue = np.matmul(covariates, coefficientMatrix[:,timeStep])
            intrinsicValue = Option.payoff(currentSpots)
            #overwrite the default to keep the option alive, if it is beneficial to keep the exercise for the ITM paths.
            ValueMatrix[:,timeStep] = continuationValue #default value to keep option alive
            cashFlowChoice = np.where(intrinsicValue>expectedContinuationValue, intrinsicValue, continuationValue)

            pathsITM = np.where(intrinsicValue>0)
            ValueMatrix[:,timeStep][pathsITM] = cashFlowChoice[pathsITM]

    return ValueMatrix[:,1].mean()*np.exp(-MarketVariables.r*timeIncrement)


if __name__ == '__main__':
    timeStepsTotal = 9
    normalizeStrike=100
    callMax = Option(timeToMaturity=3, strike=1, typeOfContract="CallMax")
    marketVariables = SimGBMMultidimensions.MarketVariables(r=0.05, dividend=0.0, vol=0.2, spots=[100/normalizeStrike,100/normalizeStrike], correlation=0.0)
    learningPaths = SimGBMMultidimensions.simulatePaths(timeStepsTotal=timeStepsTotal,pathsTotal=1*10**4, marketVariables=marketVariables, timeToMat=callMax.timeToMaturity)
    coefMatrix = findRegressionCoefficient(simulatedPaths=learningPaths, basisFuncTotal=7, Option=callMax, MarketVariables=marketVariables)
    pricingPaths = SimGBMMultidimensions.simulatePaths(timeStepsTotal=timeStepsTotal,pathsTotal=1*10**4, marketVariables=marketVariables, timeToMat=callMax.timeToMaturity)
    price = priceAmericanOption(coefMatrix,pricingPaths,callMax, marketVariables)*normalizeStrike
    print(price)