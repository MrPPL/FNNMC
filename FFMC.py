#!/usr/bin/env python
# PyTorch 1.8.1-CPU virtual env.
# Python 3.9.4 Windows 10
# -*- coding: utf-8 -*-

"""The script implement the feed forward monte carlo mehtod which is inspired by longstaff-schwartz algorithm for pricing american options.
The only difference in the regression phase
"""

import LSM
import numpy as np

timeStepsTotal = 5
normalizeStrike=40
putOption = LSM.Option(strike=1,payoffType="Put", timeToMat=1)
MarketVariablesEx1 = LSM.MarketVariables(r=0.06,vol=0.2, spot=40/normalizeStrike)
learningPaths= LSM.generateSDEStockPaths(pathTotal=10**1, timeStepsTotal=timeStepsTotal, timeToMat=putOption.timeToMat, MarketVariables=MarketVariablesEx1)
#############
# Data preparation
#############
import torch
tensorPaths = torch.from_numpy(learningPaths)
train_ldr = torch.utils.data.DataLoader(tensorPaths,
  batch_size=2, shuffle=True)


class regressionDataset(torch.utils.data.Dataset):
    def __init__(self, covariates, response):
        matrixCovariates = covariates.reshape(-1,1)
        matrixResponse = response.reshape(-1,1)
        self.covariates = torch.tensor(matrixCovariates, dtype=torch.float32).to(device)
        self.response = torch.tensor(matrixResponse, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.covariates)

    def __getitem__(self, idx):
        preds = self.covariates[idx,:]  # or just [idx]
        price = self.response[idx,:] 
        return (preds, price)       # tuple of matrices


##########
# Model for regression
#########
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#Design model
from torch import nn
class Net(torch.nn.Module):
  def __init__(self, hiddenSize1, hiddenSize2, ):
    super(Net, self).__init__()
    self.hid1 = torch.nn.Linear(1, hiddenSize1)  # 8-(10-10)-1
    #self.drop1 = torch.nn.Dropout(0.50)
    self.hid2 = torch.nn.Linear(hiddenSize1, hiddenSize2)
    #self.drop2 = torch.nn.Dropout(0.25)
    self.oupt = torch.nn.Linear(hiddenSize2, 1)

    torch.nn.init.xavier_uniform_(self.hid1.weight)
    torch.nn.init.zeros_(self.hid1.bias)
    torch.nn.init.xavier_uniform_(self.hid2.weight)
    torch.nn.init.zeros_(self.hid2.bias)
    torch.nn.init.xavier_uniform_(self.oupt.weight)
    torch.nn.init.zeros_(self.oupt.bias)

  def forward(self, x):
    z = torch.relu(self.hid1(x))
    #z = self.drop1(z)
    z = torch.relu(self.hid2(z))
    #z = self.drop2(z)
    z = self.oupt(z)  # no activation
    return z

net = Net(hiddenSize1=10, hiddenSize2=20).to(device)
#Prepare the training and test data
#Implement a Dataset object to serve up the data in batches
#Design and implement a neural network
#Write code to train the network
#Write code to evaluate the model (the trained network)
#Write code to save and use the model to make predictions for new, previously unseen data

################
# Training network function
##############
def trainNetwork(trainingData, model, lrn_rate, epochs, timeStep):
    model.train()  # set mode
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrn_rate)
    for epoch in range(0, epochs):
        torch.manual_seed(1 + epoch)  # recovery reproduce
        epoch_loss = 0.0  # sum avg loss per item
        for (batch_idx, batch) in enumerate(trainingData):
            predictor = batch[0]  
            response = batch[1]  
            optimizer.zero_grad()
            output = model(predictor)            
            loss_val = loss_func(output, response)  # avg loss in batch
            epoch_loss += loss_val.item()  # a sum of averages
            loss_val.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(" epoch = %4d   loss = %0.4f" % \
            (epoch, epoch_loss))

    path = ".\\TrainedModels\\" + str("modelAtTimeStep") + \
    str(timeStep) + ".pth"
    torch.save(model.state_dict(), path)
    print("\nDone ")
 

##########
# Dynamic Regression Phase
##########
class hyperparameters:
    # The object holds the "observable" market variables
    def __init__(self, learningRate, hiddenlayer1, hiddenlayer2, epochs, batchSize):
        self.learningRate = learningRate
        self.hiddenlayer1 = hiddenlayer1
        self.hiddenlayer2 = hiddenlayer2
        self.epochs = epochs
        self.batchSize = batchSize


def findNeuralNetworkModels(simulatedPaths, Option, MarketVariables, hyperParameters):
    # Go backward recursively to find the regression coefficients all the way back to T_1
    # and then store all the trained models 
    timeStepsTotal = simulatedPaths.shape[1]-1
    timeIncrement = Option.timeToMat/(timeStepsTotal)
    for timeStep in range(timeStepsTotal,0,-1):
        #Get payoff at maturity
        if(timeStep==timeStepsTotal):
            simulatedPaths[:,timeStep] = Option.payoff(simulatedPaths[:,timeStep])
            path = ".\\TrainedModels\\" + str("modelAtTimeStep") + \
            str(timeStep) + ".pth"
            torch.save(Net(hyperParameters.hiddenlayer1, hyperParameters.hiddenlayer2).state_dict(), path)
            print("\nDone ")
        #Find regressionscoefficients at each exercise dates before maturity
        else:
            response = np.exp(-MarketVariables.r*timeIncrement)*simulatedPaths[:,timeStep+1]
            covariates = simulatedPaths[:,timeStep]
            pathsITM = np.where(Option.payoff(covariates)>0)
            if(np.shape(pathsITM)[1]):
                #create dataset for training
                trainingData = regressionDataset(covariates[pathsITM], response[pathsITM])
                iterableTrainingData = torch.utils.data.DataLoader(trainingData, batch_size=hyperParameters.batchSize, shuffle=True)
                regressionModel = Net(hyperParameters.hiddenlayer1, hyperParameters.hiddenlayer2).to(device)
                path = ".\\TrainedModels\\" + str("modelAtTimeStep") + \
                str(timeStep+1) + ".pth"
                regressionModel.load_state_dict(torch.load(path))
                trainNetwork(trainingData=iterableTrainingData, model=regressionModel, lrn_rate=hyperParameters.learningRate,
                   epochs=hyperParameters.epochs, timeStep=timeStep)

                #load model after training set
                evaluationModel = Net(hyperParameters.hiddenlayer1, hyperParameters.hiddenlayer2).to(device)
                path = ".\\TrainedModels\\" + str("modelAtTimeStep") + \
                str(timeStep) + ".pth"
                evaluationModel.load_state_dict(torch.load(path))
                with torch.no_grad():
                    expectedContinuationValue = evaluationModel(torch.tensor(covariates.reshape(-1,1), dtype=torch.float32).to(device))
                intrinsicValue = Option.payoff(simulatedPaths[:,timeStep])
                simulatedPaths[:,timeStep] = response
                #overwrite the default to keep the option alive, if it is beneficial to keep the exercise for the ITM paths.
                #CashFlow from decision wheather to stop or keep option alive
                npExpectedContinuationValue = expectedContinuationValue.numpy().flatten()
                cashFlowChoice = np.where(intrinsicValue>npExpectedContinuationValue, intrinsicValue, response)
                simulatedPaths[:,timeStep][pathsITM] = cashFlowChoice[pathsITM]
            else:
                simulatedPaths[:,timeStep] = response

def test_MLRegression():
    learningPaths = np.array([[1,1.09,1.08,1.34],
                             [1, 1.16,1.26,1.54],
                             [1, 1.22, 1.07, 1.03],
                             [1, 0.93, 0.97, 0.92],
                             [1, 1.11, 1.56, 1.52],
                             [1, 0.76, 0.77, 0.90],
                             [1, 0.92, 0.84, 1.01],
                             [1, 0.88, 1.22, 1.34]])
    marketVariablesEX = LSM.MarketVariables(r=0.06,vol=0,spot=1)
    putOption = LSM.Option(strike=1.1, payoffType="Put",timeToMat=3)
    hyperParamters = hyperparameters(learningRate=0.001, hiddenlayer1=10, hiddenlayer2=10, epochs=10, batchSize=2)
    findNeuralNetworkModels(simulatedPaths=learningPaths, Option=putOption, MarketVariables=marketVariablesEX, hyperParameters=hyperParamters) 

test_MLRegression()