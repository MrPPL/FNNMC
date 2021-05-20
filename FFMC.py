#!/usr/bin/env python
# PyTorch 1.8.1-CPU virtual env.
# Python 3.9.4 Windows 10
# -*- coding: utf-8 -*-

"""The script implement the feed forward monte carlo mehtod which is inspired by longstaff-schwartz algorithm for pricing american options.
The only difference in the regression phase
"""

import LSM
import numpy as np

#############
# Data preparation
#############
import torch
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
class Net(torch.nn.Module):
  def __init__(self, hiddenSize1, hiddenSize2):
    super(Net, self).__init__()
    self.hiddenlayer1 = torch.nn.Linear(1, hiddenSize1)  
    #self.drop1 = torch.nn.Dropout(0.50)
    self.hiddenlayer2 = torch.nn.Linear(hiddenSize1, hiddenSize2)
    #self.drop2 = torch.nn.Dropout(0.25)
    self.output = torch.nn.Linear(hiddenSize2, 1)

    torch.nn.init.xavier_uniform_(self.hiddenlayer1.weight)
    torch.nn.init.zeros_(self.hiddenlayer1.bias)
    torch.nn.init.xavier_uniform_(self.hiddenlayer2.weight)
    torch.nn.init.zeros_(self.hiddenlayer2.bias)
    torch.nn.init.xavier_uniform_(self.output.weight)
    torch.nn.init.zeros_(self.output.bias)

  def forward(self, x):
    z = torch.relu(self.hiddenlayer1(x))
    #z = self.drop1(z)
    z = torch.relu(self.hiddenlayer2(z))
    #z = self.drop2(z)
    z = self.output(z)  # no activation
    return z

################
# Training network 
##############
def trainNetwork(trainingData, model, hyperparameters, timeStep):
    """The function train the neural network model based on hyperparameters given. 
    The function saves the trained models in TrainedModels directory"""
    model.train()  # set mode
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.learningRate)
    for epoch in range(0, hyperparameters.epochs):
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
class Hyperparameters:
    # The object holds the "observable" market variables
    def __init__(self, learningRate, hiddenlayer1, hiddenlayer2, epochs, batchSize):
        self.learningRate = learningRate
        self.hiddenlayer1 = hiddenlayer1
        self.hiddenlayer2 = hiddenlayer2
        self.epochs = epochs
        self.batchSize = batchSize


def findNeuralNetworkModels(simulatedPaths, Option, MarketVariables, hyperparameters):
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
        
            torch.save(Net(hyperparameters.hiddenlayer1, hyperparameters.hiddenlayer2).state_dict(), path)
        #Find regressionsmodels at each exercise dates before maturity
        else:
            response = np.exp(-MarketVariables.r*timeIncrement)*simulatedPaths[:,timeStep+1]
            covariates = simulatedPaths[:,timeStep]
            pathsITM = np.where(Option.payoff(covariates)>0)
            if(np.shape(pathsITM)[1]):
                #create dataset for training
                trainingData = regressionDataset(covariates[pathsITM], response[pathsITM])
                iterableTrainingData = torch.utils.data.DataLoader(trainingData, batch_size=hyperparameters.batchSize, shuffle=True)
                regressionModel = Net(hyperparameters.hiddenlayer1, hyperparameters.hiddenlayer2).to(device)
                path = ".\\TrainedModels\\" + str("modelAtTimeStep") + \
                str(timeStep+1) + ".pth"
                regressionModel.load_state_dict(torch.load(path))
                trainNetwork(trainingData=iterableTrainingData, model=regressionModel, hyperparameters=hyperparameters,
                    timeStep=timeStep)

                #load model after training of model
                evaluationModel = Net(hyperparameters.hiddenlayer1, hyperparameters.hiddenlayer2).to(device)
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


#########################
# Pricing phase
########################

def priceAmericanOption(simulatedPaths, Option, MarketVariables, hyperparameters):
    timeStepsTotal = simulatedPaths.shape[1]-1 #time 0 does not count to a timestep
    timeIncrement = Option.timeToMat/timeStepsTotal
    regressionModel = Net(hyperparameters.hiddenlayer1, hyperparameters.hiddenlayer2).to(device)
    regressionModel.eval()
    for timeStep in range(timeStepsTotal,0,-1):
        #Get payoff at maturity
        if(timeStep==timeStepsTotal):
            simulatedPaths[:,timeStep] = Option.payoff(simulatedPaths[:,timeStep])
        #Use coefficientMatrix and paths to price american option 
        else:
            continuationValue = np.exp(-MarketVariables.r*timeIncrement)*simulatedPaths[:,timeStep+1]
            covariates = simulatedPaths[:,timeStep]
            path = ".\\TrainedModels\\" + str("modelAtTimeStep") + \
                str(timeStep) + ".pth"
            regressionModel.load_state_dict(torch.load(path))
            with torch.no_grad():
                expectedContinuationValue = regressionModel(torch.tensor(covariates.reshape(-1,1), dtype=torch.float32).to(device))            
            intrinsicValue = Option.payoff(simulatedPaths[:,timeStep])
            #overwrite the default to keep the option alive, if it is beneficial to keep the exercise for the ITM paths.
            simulatedPaths[:,timeStep] = continuationValue #default value to keep option alive
            npExpectedContinuationValue = expectedContinuationValue.numpy().flatten()
            cashFlowChoice = np.where(intrinsicValue>npExpectedContinuationValue, intrinsicValue, continuationValue)
            pathsITM = np.where(intrinsicValue>0)
            simulatedPaths[:,timeStep][pathsITM] = cashFlowChoice[pathsITM]
    return simulatedPaths[:,1].mean()*np.exp(-MarketVariables.r*timeIncrement)


#######################
# Test functions
#######################
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
    hyperParameters = Hyperparameters(learningRate=0.001, hiddenlayer1=100, hiddenlayer2=100, epochs=10, batchSize=2)
    findNeuralNetworkModels(simulatedPaths=learningPaths, Option=putOption, MarketVariables=marketVariablesEX, hyperparameters=hyperParameters) 

test_MLRegression()

def test_PricePhase():
    marketVariablesEX = LSM.MarketVariables(r=0.06,vol=0,spot=1)
    putOption = LSM.Option(strike=1.1, payoffType="Put",timeToMat=3)
    pricingPaths = np.array([[1,1.09,1.08,1.34],
                             [1, 1.16,1.26,1.54],
                             [1, 1.22, 1.07, 1.03],
                             [1, 0.93, 0.97, 0.92],
                             [1, 1.11, 1.56, 1.52],
                             [1, 0.76, 0.77, 0.90],
                             [1, 0.92, 0.84, 1.01],
                             [1, 0.88, 1.22, 1.34]])

    hyperparameters = Hyperparameters(learningRate=0.001, hiddenlayer1=100, hiddenlayer2=100, epochs=1, batchSize=2)
    priceAmerOption = priceAmericanOption(simulatedPaths=pricingPaths, Option=putOption, MarketVariables=marketVariablesEX, 
        hyperparameters=hyperparameters)
    return priceAmerOption

print(test_PricePhase())


#timeStepsTotal = 50
#normalizeStrike=40
#putOption = LSM.Option(strike=1,payoffType="Put", timeToMat=1)
#marketVariablesEx1 = LSM.MarketVariables(r=0.06,vol=0.2, spot=40/normalizeStrike)
#learningPaths= LSM.generateSDEStockPaths(pathTotal=1*10**5, timeStepsTotal=timeStepsTotal, timeToMat=putOption.timeToMat, MarketVariables=marketVariablesEx1)
#hyperparameters = Hyperparameters(learningRate=0.001, hiddenlayer1=10, hiddenlayer2=10, epochs=10, batchSize=10**4)
#findNeuralNetworkModels(simulatedPaths=learningPaths, Option=putOption, MarketVariables=marketVariablesEx1, hyperparameters=hyperparameters)
#pricingPaths = LSM.generateSDEStockPaths(pathTotal=1*10**5, timeStepsTotal=timeStepsTotal, timeToMat=putOption.timeToMat, MarketVariables=marketVariablesEx1)
#price = priceAmericanOption(simulatedPaths=pricingPaths, Option=putOption, MarketVariables=marketVariablesEx1, hyperparameters=hyperparameters)*normalizeStrike
#print("Price of American put: ", price)