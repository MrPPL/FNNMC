
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
class Option():
    def __init__(self, timeToMaturity=0, strike=0, typeOfContract="Put"):
         self.timeToMaturity = timeToMaturity
         self.strike = strike
         self.typeOfContract = typeOfContract

    def __str__(self):
        return 'Time to Maturity %.d years\nStrike %.d \nType of Contract %s Option' % (self.timeToMaturity, self.strike, self.typeOfContract)

    def payoff(self, spots):
        if (self.typeOfContract=="Put"):
            return np.maximum(0, self.strike - spots)
        elif (self.typeOfContract=="Call"):
            return np.maximum(0,spots - self.strike)
        elif (self.typeOfContract=="PutGeometricAverage"):
            return np.maximum(0, self.strike - np.prod(spots)**(1/len(spots)))
        elif(self.typeOfContract=="CallMax"):
            return np.maximum(0, np.amax(spots,1)-self.strike)
        else:
            print("Invalid input for the payoff function")

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
            response = np.exp(-MarketVariables.r*timeIncrement)*ValueMatrix[:,timeStep+1]
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

if __name__ == '__main__':
    callMax = Option(timeToMaturity=1, strike=40, typeOfContract="CallMax")
    marketVariables = SimGBMMultidimensions.MarketVariables(r=0, vol=0.2, spots=[40,40], correlation=0.8)
    paths = SimGBMMultidimensions.simulatePaths(timeStepsTotal=5,pathsTotal=4, marketVariables=marketVariables, timeToMat=callMax.timeToMaturity)
    coefMatrix = findRegressionCoefficient(simulatedPaths=paths, basisFuncTotal=7, Option=callMax, MarketVariables=marketVariables)
    pass