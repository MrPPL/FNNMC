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
                plt.plot(covariates, response,'ro')
                polyFit = np.poly1d(regressionFit)
                plt.plot(xRangePoly, polyFit(xRangePoly), c='g',linestyle='-')
                plt.title('Polynomial')
                plt.xlabel('Current spots')
                plt.ylabel('Continuation Value')
                plt.axis([0.0, 2.0,-1.0,1.0])
                plt.grid(True)
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
    learningPaths= SimulationPaths.GBM.generateSDEStockPaths(pathTotal=10**4, timeStepsPerYear=timeStepsPerYear, timeToMat=putOption.timeToMat, MarketVariables=MarketVariablesEx1)
    timeSimulationEnd = time.time()
    print(f"Time taken for simulation {timeSimulationEnd-timeSimulationStart:f}")
    timeRegressionStart = time.time()
    regressionCoefficient = PlotRegression(basisFuncTotal=3, Option=putOption, simulatedPaths=learningPaths, MarketVariables=MarketVariablesEx1)
    timeRegressionEnd = time.time()
    print(f"Time taken for regression {timeRegressionEnd-timeRegressionStart:f}")