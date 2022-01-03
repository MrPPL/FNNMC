import numpy as np
import matplotlib.pyplot as plt
##########
# Regression Phase
##########
def PlotRegression(simulatedPaths, basisFuncTotal, Option, MarketVariables):
    # Go backward recursively to find the regression coefficients all the way back to T_1
    # and then store all the regression coefficients
    timeStepsTotal = np.shape(simulatedPaths)[1]-1
    coefficientMatrix = np.empty((basisFuncTotal+1,timeStepsTotal))
    timeIncrement = Option.timeToMat/(timeStepsTotal)
    xRangePoly = np.linspace(0.0,2.0,100)
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
                xRangePoly = np.linspace(0.0,2.0,1000)
                plt.style.use('ggplot')
                plt.scatter(covariates, response, c="black", alpha=0.1)
                polyFit = np.poly1d(regressionFit)
                #plt.plot(xRangePoly, polyFit(xRangePoly), c='r',linestyle='-')
                #plt.title('Polynomial Regression')
                plt.xlabel(f'Current stock price at timestep {timeStep}')
                plt.ylabel('Continuation Value')
                plt.axis([0.0, 2.0,-1.0,1.0])
                plt.grid(True)
                pathToSave = f'C:\\Users\\HY34WN\\OneDrive - Aalborg Universitet\\Documents\\PhD\\My_Papers\\FFNNMC\\Illustration\\AmericanPut\\Data\\Step{timeStep}.png'
                #plt.savefig(pathToSave)
                plt.show()
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
import torch
import FNNMC
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def plotNeuralNetwork(simulatedPaths, Option, MarketVariables, hyperparameters):
    # Go backward recursively to find the regression coefficients all the way back to T_1
    # and then store all the trained models 
    timeStepsTotal = simulatedPaths.shape[1]-1
    timeIncrement = Option.timeToMat/(timeStepsTotal)
    for timeStep in range(timeStepsTotal,0,-1):
        #Get payoff at maturity
        if(timeStep==timeStepsTotal):
            response = Option.payoff(simulatedPaths[:,timeStep])
            path = ".\\TrainedModels\\" + str("modelAtTimeStep") + \
                str(timeStep) + ".pth"
        
            torch.save(FNNMC.Net(hyperparameters.hiddenlayer1, hyperparameters.hiddenlayer2).state_dict(), path)
        #Find regressionsmodels at each exercise dates before maturity
        else:
            response = np.exp(-MarketVariables.r*timeIncrement)*response
            covariates = simulatedPaths[:,timeStep]
            pathsITM = np.where(Option.payoff(covariates)>0)
            if(np.shape(pathsITM)[1]):
                #create dataset for training
                trainingData = FNNMC.regressionDataset(covariates[pathsITM], response[pathsITM])
                iterableTrainingData = torch.utils.data.DataLoader(trainingData, batch_size=hyperparameters.batchSize, shuffle=True)
                regressionModel = FNNMC.Net(hyperparameters.hiddenlayer1, hyperparameters.hiddenlayer2).to(device)
                path = ".\\TrainedModels\\" + str("modelAtTimeStep") + \
                str(timeStep+1) + ".pth"
                regressionModel.load_state_dict(torch.load(path))
                FNNMC.trainNetwork(trainingData=iterableTrainingData, model=regressionModel, hyperparameters=hyperparameters,
                    timeStep=timeStep)

                #load model after training of model
                evaluationModel = FNNMC.Net(hyperparameters.hiddenlayer1, hyperparameters.hiddenlayer2).to(device)
                path = ".\\TrainedModels\\" + str("modelAtTimeStep") + \
                str(timeStep) + ".pth"
                evaluationModel.load_state_dict(torch.load(path))
                with torch.no_grad():
                    expectedContinuationValue = evaluationModel(torch.tensor(covariates.reshape(-1,1), dtype=torch.float32).to(device))
                    xRangePoly = np.linspace(0.0,2.0,1000)
                    plotRegression = evaluationModel(torch.tensor(xRangePoly.reshape(-1,1), dtype=torch.float32).to(device))
                    plt.style.use('ggplot')
                    plt.scatter(covariates , response, c="black", alpha=0.1)
                    plt.plot(xRangePoly, plotRegression, c="red")
                    plt.title('Neural Network Regression')
                    plt.xlabel(f'Current stock price at timestep {timeStep}')
                    plt.ylabel('Continuation Value')
                    plt.axis([0.0, 2.0,-1.0,1.0])
                    plt.grid(True)
                    pathToSave = f'C:\\Users\\HY34WN\\OneDrive - Aalborg Universitet\\Documents\\PhD\\My_Papers\\FFNNMC\\Illustration\\AmericanPut\\mixedEpoch\\NNRegressionStep{timeStep}.png'
                    #plt.savefig(pathToSave)
                    plt.show()
                intrinsicValue = Option.payoff(simulatedPaths[:,timeStep])
                #overwrite the default to keep the option alive, if it is beneficial to keep the exercise for the ITM paths.
                #CashFlow from decision wheather to stop or keep option alive
                npExpectedContinuationValue = expectedContinuationValue.numpy().flatten()
                cashFlowChoice = np.where(intrinsicValue>npExpectedContinuationValue, intrinsicValue, response)
                response[pathsITM] = cashFlowChoice[pathsITM]
            else:
                simulatedPaths[:,timeStep] = response


##########################
# Testing execution
##########################
import Products
import time
import SimulationPaths.GBM
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
    regressionCoefficient = PlotRegression(basisFuncTotal=3, Option=putOption, simulatedPaths=learningPaths, MarketVariables=MarketVariablesEx1)
    #hyperparameters = FNNMC.Hyperparameters(learningRate=0.001, hiddenlayer1=100, hiddenlayer2=100, epochs=10, batchSize=10**4, trainOnlyLastTimeStep=True)
    #regressionCoefficient = plotNeuralNetwork(Option=putOption, simulatedPaths=learningPaths, MarketVariables=MarketVariablesEx1, hyperparameters=hyperparameters)
    timeRegressionEnd = time.time()
    print(f"Time taken for regression {timeRegressionEnd-timeRegressionStart:f}")