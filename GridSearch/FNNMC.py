
#!/usr/bin/env python
# PyTorch 1.8.1-CPU virtual env.
# Python 3.9.4 Windows 10
# -*- coding: utf-8 -*-

"""The script implement the classical longstaff-schwartz algorithm for pricing american options.
This script focus on the multidimensional case for rainbow option
"""
import numpy as np
import torch

#############
# Data preparation
#############
class regressionDataset(torch.utils.data.Dataset):
    def __init__(self, covariates, response):
        matrixCovariates = covariates
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
  def __init__(self, inputSize, hiddenSize1):
    super(Net, self).__init__()
    self.hiddenlayer1 = torch.nn.Linear(inputSize, hiddenSize1)  
    #self.drop1 = torch.nn.Dropout(0.25)
    #self.hiddenlayer2 = torch.nn.Linear(hiddenSize1, hiddenSize2)
    #self.drop2 = torch.nn.Dropout(0.25)
    self.output = torch.nn.Linear(hiddenSize1, 1)

    torch.nn.init.xavier_uniform_(self.hiddenlayer1.weight)
    torch.nn.init.zeros_(self.hiddenlayer1.bias)
    #torch.nn.init.xavier_uniform_(self.hiddenlayer2.weight)
    #torch.nn.init.zeros_(self.hiddenlayer2.bias)
    torch.nn.init.xavier_uniform_(self.output.weight)
    torch.nn.init.zeros_(self.output.bias)

  def forward(self, x):
    #relu=torch.nn.ReLU()
    #Leakyrelu=torch.nn.LeakyReLU(negative_slope=0.01)
    Leakyrelu=torch.nn.LeakyReLU(negative_slope=0.3)
    z = Leakyrelu(self.hiddenlayer1(x))
    #z = self.drop1(z)
    #z = relu(self.hiddenlayer2(z))
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

        print(" epoch = %4d   loss = %0.4f" % \
        (epoch, epoch_loss))
        if (hyperparameters.trainOnlyLastTimeStep==True):
            hyperparameters.epochs = 1

    path = ".\\TrainedModels\\" + str("modelAtTimeStep") + \
    str(timeStep) + ".pth"
    torch.save(model.state_dict(), path)
    print("\nDone ")
 

##########
# Dynamic Regression Phase
##########
class Hyperparameters:
    # The object holds the "observable" market variables
    def __init__(self, learningRate, inputSize, hiddenlayer1, epochs, batchSize, trainOnlyLastTimeStep=False):
        self.learningRate = learningRate
        self.inputSize = inputSize
        self.hiddenlayer1 = hiddenlayer1
        #self.hiddenlayer2 = hiddenlayer2
        self.epochs = epochs
        self.batchSize = batchSize
        self.trainOnlyLastTimeStep= trainOnlyLastTimeStep


##########
# Dynamic Regression Phase
##########
def findNeuralNetworkModels(simulatedPaths, Option, MarketVariables, hyperparameters):
    # Go backward recursively to find the regression coefficients all the way back to T_1
    # and then store all the regression coefficients
    timeStepsTotal = np.shape(simulatedPaths)[0]-1
    timeIncrement = Option.timeToMat/(timeStepsTotal)
    for timeStep in range(timeStepsTotal,0,-1):
        #Get payoff at maturity
        if(timeStep== (timeStepsTotal) ):
            ValueVec = Option.payoff(simulatedPaths[timeStep,:,:])
            path = ".\\TrainedModels\\" + str("modelAtTimeStep") + \
                str(timeStep) + ".pth"
            torch.save(Net(hyperparameters.inputSize, hyperparameters.hiddenlayer1).state_dict(), path)
        #Find regressionscoefficients at each exercise dates before maturity
        else:
            response = np.exp(-MarketVariables.r*timeIncrement)*ValueVec
            currentSpots = simulatedPaths[timeStep,:,:]
            pathsITM = np.where(Option.payoff(currentSpots)>0)
            if(np.shape(pathsITM)[1]):
                #create dataset for training
                trainingData = regressionDataset(currentSpots[pathsITM], response[pathsITM])
                iterableTrainingData = torch.utils.data.DataLoader(trainingData, batch_size=hyperparameters.batchSize, shuffle=True)
                regressionModel = Net(hyperparameters.inputSize ,hyperparameters.hiddenlayer1).to(device)
                path = ".\\TrainedModels\\" + str("modelAtTimeStep") + \
                str(timeStep+1) + ".pth"
                regressionModel.load_state_dict(torch.load(path))
                trainNetwork(trainingData=iterableTrainingData, model=regressionModel, hyperparameters=hyperparameters,
                    timeStep=timeStep)
                #load model after training of model
                evaluationModel = Net(hyperparameters.inputSize, hyperparameters.hiddenlayer1).to(device)
                path = ".\\TrainedModels\\" + str("modelAtTimeStep") + \
                str(timeStep) + ".pth"
                evaluationModel.load_state_dict(torch.load(path))
                with torch.no_grad():
                    expectedContinuationValue = evaluationModel(torch.tensor(currentSpots, dtype=torch.float32).to(device))
                intrinsicValue = Option.payoff(currentSpots)
                ValueVec = response
                #overwrite the default to keep the option alive, if it is beneficial to keep the exercise for the ITM paths.
                #CashFlow from decision wheather to stop or keep option alive
                npExpectedContinuationValue = expectedContinuationValue.numpy().flatten() #transform tensor to vector
                cashFlowChoice = np.where(intrinsicValue>npExpectedContinuationValue, intrinsicValue, response)
                ValueVec[pathsITM] = cashFlowChoice[pathsITM]
            else:
                ValueVec = response

#########
# Pricing
#########
def priceAmericanOption(simulatedPaths, Option, MarketVariables, hyperparameters):
    timeStepsTotal = simulatedPaths.shape[0]-1 #time 0 does not count to a timestep
    timeIncrement = Option.timeToMat/timeStepsTotal
    regressionModel = Net(hyperparameters.inputSize,hyperparameters.hiddenlayer1).to(device)
    regressionModel.eval()
    for timeStep in range(timeStepsTotal,0,-1):
        #Get payoff at maturity
        if(timeStep==timeStepsTotal):
            ValueVec = Option.payoff(simulatedPaths[timeStep,:,:])
        #Use coefficientMatrix and paths to price american option 
        else:
            continuationValue = np.exp(-MarketVariables.r*timeIncrement)*ValueVec
            currentSpots = simulatedPaths[timeStep,:,:]
            path = ".\\TrainedModels\\" + str("modelAtTimeStep") + \
                str(timeStep) + ".pth"
            regressionModel.load_state_dict(torch.load(path))
            with torch.no_grad():
                expectedContinuationValue = regressionModel(torch.tensor(currentSpots, dtype=torch.float32).to(device))            
            intrinsicValue = Option.payoff(currentSpots)
            #overwrite the default to keep the option alive, if it is beneficial to keep the exercise for the ITM paths.
            ValueVec = continuationValue #default value to keep option alive
            npExpectedContinuationValue = expectedContinuationValue.numpy().flatten()
            cashFlowChoice = np.where(intrinsicValue>npExpectedContinuationValue, intrinsicValue, continuationValue)

            pathsITM = np.where(intrinsicValue>0)
            ValueVec[pathsITM] = cashFlowChoice[pathsITM]

    return ValueVec.mean()*np.exp(-MarketVariables.r*timeIncrement)
