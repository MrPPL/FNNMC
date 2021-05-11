#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The script implement the feed forward monte carlo mehtod which is inspired by longstaff-schwartz algorithm for pricing american options.
The only difference in the regression phase
"""

import LSM


timeStepsTotal = 50
normalizeStrike=40
putOption = LSM.Option(strike=1,payoffType="Put", timeToMat=1)
MarketVariablesEx1 = LSM.MarketVariables(r=0.06,vol=0.2, spot=40/normalizeStrike)
learningPaths= LSM.generateSDEStockPaths(pathTotal=10**5, timeStepsTotal=timeStepsTotal, timeToMat=putOption.timeToMat, MarketVariables=MarketVariablesEx1)
import torch
tensorPaths = torch.from_numpy(learningPaths)

##########
# Model for regression
#########
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#Design model
from torch import nn
class NeuralNetwork(nn.Module):
        def __init__(self, inputSize, hiddenSize1, hiddenSize2, outputSize):
                super(NeuralNetwork, self).__init__()
                self.input_size = inputSize
                self.l1 = nn.Linear(inputSize, hiddenSize1)
                self.leaky_relu_1 = nn.LeakyReLU(negative_slope=0.3)
                self.l2 = nn.Linear(hiddenSize1,hiddenSize2)
                self.leaky_relu_2 = nn.LeakyReLU(negative_slope=0.3)
                self.l3 = nn.Linear(hiddenSize2,outputSize)
        
        def forward(self,x):    
                out = self.l1(x)
                out = self.leaky_relu_1(out)
                out = self.l2(out)
                out = self.leaky_relu_2(out)
                out = self.l3(out)
                return out


##########
# Regression Phase
##########
import numpy as np
def findRegressionCoefficient(simulatedPaths, Option, MarketVariables):
    # Go backward recursively to find the regression coefficients all the way back to T_1
    # and then store all the trained models 
    timeStepsTotal = simulatedPaths.shape[1]-1
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

                regressionModel = NeuralNetwork(np.shape(pathsITM)[1], 120, 120, 1).to(device)
                regressionFit = np.polyfit(covariates[pathsITM],response[pathsITM])
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


