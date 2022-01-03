#!/usr/bin/env python
# PyTorch 1.8.1-CPU virtual env.
# Python 3.9.4 Windows 10
# -*- coding: utf-8 -*-

"""The script implement the feed forward monte carlo mehtod which is inspired by longstaff-schwartz algorithm for pricing american options.
The only difference in the regression phase
"""

import LSM
import numpy as np
import os

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
  def __init__(self, hyperparameters):
    super(Net, self).__init__()
    self.hiddenlayer1 = torch.nn.Linear(hyperparameters.inputSize, hyperparameters.hiddenlayer1)  
    self.hiddenlayer2 = torch.nn.Linear(hyperparameters.hiddenlayer1, hyperparameters.hiddenlayer2)
    self.hiddenlayer3 = torch.nn.Linear(hyperparameters.hiddenlayer2, hyperparameters.hiddenlayer3)
    self.hiddenlayer4 = torch.nn.Linear(hyperparameters.hiddenlayer3, hyperparameters.hiddenlayer4)
    #self.hiddenlayer5 = torch.nn.Linear(hyperparameters.hiddenlayer4, hyperparameters.hiddenlayer5)
    #self.hiddenlayer6 = torch.nn.Linear(hyperparameters.hiddenlayer5, hyperparameters.hiddenlayer6)
    self.output = torch.nn.Linear(hyperparameters.hiddenlayer4, 1)

    torch.nn.init.xavier_uniform_(self.hiddenlayer1.weight)
    torch.nn.init.zeros_(self.hiddenlayer1.bias)
    torch.nn.init.xavier_uniform_(self.hiddenlayer2.weight)
    torch.nn.init.zeros_(self.hiddenlayer2.bias)
    torch.nn.init.xavier_uniform_(self.hiddenlayer3.weight)
    torch.nn.init.zeros_(self.hiddenlayer3.bias)
    torch.nn.init.xavier_uniform_(self.hiddenlayer4.weight)
    torch.nn.init.zeros_(self.hiddenlayer4.bias)
    #torch.nn.init.xavier_uniform_(self.hiddenlayer5.weight)
    #torch.nn.init.zeros_(self.hiddenlayer5.bias)
    #torch.nn.init.xavier_uniform_(self.hiddenlayer6.weight)
    #torch.nn.init.zeros_(self.hiddenlayer6.bias)
    torch.nn.init.xavier_uniform_(self.output.weight)
    torch.nn.init.zeros_(self.output.bias)

  def forward(self, x):
    relu=torch.nn.ReLU()
    z = relu(self.hiddenlayer1(x))
    z = relu(self.hiddenlayer2(z))
    z = relu(self.hiddenlayer3(z))
    z = relu(self.hiddenlayer4(z))
    #z = relu(self.hiddenlayer5(z))
    #z = relu(self.hiddenlayer6(z))
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
    bestEpoch_loss = 0.0 #Parameter to keep track of performance improvement
    notImprovedEpoch = 0 #count number of improved iterations
    path =  os.path.join(".", "TrainedModels", "modelAtTimeStep" + str(timeStep) + ".pth") #path to save trained model
    for epoch in range(0, hyperparameters.epochs):
        torch.manual_seed(1 + epoch)  # recovery reproduce
        epoch_loss = 0.0  # sum avg loss per item
        for (batch_idx, batch) in enumerate(trainingData):
            predictor = batch[0]
            response = batch[1]  
            optimizer.zero_grad()
            output = model(predictor)  #forward pass          
            loss_val = loss_func(output, response)  # avg loss in batch
            epoch_loss += loss_val.item()  # a sum of averages
            loss_val.backward() # Compute gradient
            optimizer.step() #parameter update

        print(" epoch = %4d   loss = %0.4f" % \
        (epoch, epoch_loss))
        #Early stopping
        if(bestEpoch_loss>epoch_loss or epoch==0):
            bestEpoch_loss = epoch_loss
            notImprovedEpoch=0
            torch.save(model.state_dict(), path)
        elif(notImprovedEpoch>=hyperparameters.patience):
            break
        else:
            notImprovedEpoch = notImprovedEpoch + 1

        if (hyperparameters.trainOnlyLastTimeStep==True):
            hyperparameters.epochs = 1

    print("\nDone ")
 
class Hyperparameters:
    # The object holds the "observable" market variables
    def __init__(self, learningRate, inputSize=0, hiddenlayer1=0, hiddenlayer2=0, hiddenlayer3=0, hiddenlayer4=0, hiddenlayer5=0, hiddenlayer6=0, epochs=10**4, batchSize=10**4, trainOnlyLastTimeStep=False, patience=5):
        self.learningRate = learningRate
        self.inputSize = inputSize
        self.hiddenlayer1 = hiddenlayer1
        self.hiddenlayer2 = hiddenlayer2
        self.hiddenlayer3 = hiddenlayer3
        self.hiddenlayer4 = hiddenlayer4
        self.hiddenlayer5 = hiddenlayer5
        self.hiddenlayer6 = hiddenlayer6
        self.epochs = epochs
        self.batchSize = batchSize
        self.trainOnlyLastTimeStep= trainOnlyLastTimeStep
        self.patience = patience

##########
# Dynamic Regression Phase
##########
def findNeuralNetworkModels(simulatedPaths, Option, MarketVariables, hyperparameters):
    # Go backward recursively to find the regression coefficients all the way back to T_1
    # and then store all the regression coefficients
    timeStepsTotal = simulatedPaths.shape[1]-1
    timeIncrement = Option.timeToMat/(timeStepsTotal)
    for timeStep in range(timeStepsTotal,0,-1):
        #Get payoff at maturity
        if(timeStep== (timeStepsTotal) ):
            ValueVec = Option.payoff(simulatedPaths[:,timeStep])
            path = os.path.join(".", "TrainedModels", str("modelAtTimeStep") + str(timeStep) + ".pth")
            torch.save(Net(hyperparameters).state_dict(), path)
        #Find regressionscoefficients at each exercise dates before maturity
        else:
            response = np.exp(-MarketVariables.r*timeIncrement)*ValueVec
            covariates = simulatedPaths[:,timeStep]
            pathsITM = np.where(Option.payoff(covariates)>0)
            if(np.shape(pathsITM)[1]):
                #create dataset for training
                trainingData = regressionDataset(covariates[pathsITM], response[pathsITM])
                iterableTrainingData = torch.utils.data.DataLoader(trainingData, batch_size=hyperparameters.batchSize, shuffle=True)
                regressionModel = Net(hyperparameters).to(device)
                path = os.path.join(".", "TrainedModels", str("modelAtTimeStep") + str(timeStep+1) + ".pth")
                regressionModel.load_state_dict(torch.load(path))

                trainNetwork(trainingData=iterableTrainingData, model=regressionModel, hyperparameters=hyperparameters,
                    timeStep=timeStep)
                #load model after training of model
                evaluationModel = Net(hyperparameters).to(device)
                path = os.path.join(".", "TrainedModels", str("modelAtTimeStep") + str(timeStep) + ".pth")
                evaluationModel.load_state_dict(torch.load(path))
                with torch.no_grad():
                    expectedContinuationValue = evaluationModel(torch.tensor(covariates.reshape(-1,1), dtype=torch.float32).to(device))
                intrinsicValue = Option.payoff(covariates)
                ValueVec = response
                #overwrite the default to keep the option alive, if it is beneficial to keep the exercise for the ITM paths.
                #CashFlow from decision wheather to stop or keep option alive
                npExpectedContinuationValue = expectedContinuationValue.numpy().flatten() #transform tensor to vector
                cashFlowChoice = np.where(intrinsicValue>npExpectedContinuationValue, intrinsicValue, response)
                ValueVec[pathsITM] = cashFlowChoice[pathsITM]
            else:
                ValueVec = response


#########################
# Pricing phase
########################
def priceAmericanOption(simulatedPaths, Option, MarketVariables, hyperparameters):
    timeStepsTotal = simulatedPaths.shape[1]-1 #time 0 does not count to a timestep
    timeIncrement = Option.timeToMat/timeStepsTotal
    regressionModel = Net(hyperparameters).to(device)
    regressionModel.eval()
    for timeStep in range(timeStepsTotal,0,-1):
        #Get payoff at maturity
        if(timeStep==timeStepsTotal):
            ValueVec = Option.payoff(simulatedPaths[:,timeStep])
        #Use coefficientMatrix and paths to price american option 
        else:
            continuationValue = np.exp(-MarketVariables.r*timeIncrement)*ValueVec
            covariates = simulatedPaths[:,timeStep]
            path = os.path.join(".", "TrainedModels", str("modelAtTimeStep") + str(timeStep) + ".pth")
            regressionModel.load_state_dict(torch.load(path))
            with torch.no_grad():
                expectedContinuationValue = regressionModel(torch.tensor(covariates.reshape(-1,1), dtype=torch.float32).to(device))            
            intrinsicValue = Option.payoff(covariates)
            #overwrite the default to keep the option alive, if it is beneficial to keep the exercise for the ITM paths.
            npExpectedContinuationValue = expectedContinuationValue.numpy().flatten()
            cashFlowChoice = np.where(intrinsicValue>npExpectedContinuationValue, intrinsicValue, continuationValue)
            pathsITM = np.where(intrinsicValue>0)
            ValueVec[pathsITM] = cashFlowChoice[pathsITM]
    return ValueVec.mean()*np.exp(-MarketVariables.r*timeIncrement)


import Products
import time
import h5py

# reproducablity
seed = 3
import random
import torch

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if __name__ == '__main__':
    timeStepsTotal = 10**3
    normalizeStrike=40
    spot = 36
    pathTotal = 10**4
    putOption = Products.Option(strike=1,typeOfContract="Put", timeToMat=1)
    marketVariablesEx1 = Products.MarketVariables(r=0.06,vol=0.2, spot=spot/normalizeStrike)
    # load data
    f = h5py.File('data/AmericanPut/1MPut1000K.hdf5', 'r')
    learningPaths = f['RND'][...]
    underlyingsTotal=1
    hyperparameters = Hyperparameters(learningRate=10**(-5), inputSize=underlyingsTotal, 
                            hiddenlayer1=underlyingsTotal+100, hiddenlayer2=underlyingsTotal+100, hiddenlayer3=underlyingsTotal+100, 
                            hiddenlayer4=underlyingsTotal+100, epochs=10**4, batchSize=64, trainOnlyLastTimeStep=False, patience=3)
    timeRegressionStart = time.time()
    findNeuralNetworkModels(simulatedPaths=learningPaths, Option=putOption, MarketVariables=marketVariablesEx1, hyperparameters=hyperparameters)
    timeRegressionEnd = time.time()
    print(f"Time taken for find regressioncoefficients: {timeRegressionEnd-timeRegressionStart:f}")

    repeat =100
    estimates = np.zeros(repeat)
    for i in range(repeat):
        # create empirical estimations
        g = h5py.File(f"data/AmericanPut/PricePaths1000K/PricePath{i}.hdf5", 'r')
        pricingPaths = g['RND'][...]
        price = priceAmericanOption(simulatedPaths=pricingPaths, Option=putOption, MarketVariables=marketVariablesEx1, hyperparameters=hyperparameters)*normalizeStrike
        print(f"Spot {spot:3d} and price of American put: {price:f}")
        estimates[i]=price
    print("Mean: ", np.mean(estimates))
    print("Std Error Mean: ", np.std(estimates)/np.sqrt(repeat))
