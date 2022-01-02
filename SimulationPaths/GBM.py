
import numpy as np
##########
# Simulation
##########
rng = np.random.Generator(np.random.PCG64(123)) #set seed and get generator object

def generateTimeStepStock(timeIncrement, rNorm, MarketVariables, previousPrice):
    #Use black-scholes transition probability sampling to make one time step for underlying stock
    return previousPrice*np.exp(((MarketVariables.r-MarketVariables.dividend)-np.square(MarketVariables.vol)*0.5)*timeIncrement+np.sqrt(timeIncrement)*MarketVariables.vol*rNorm)

def generateSDEStockPaths(pathTotal, timeStepsPerYear, timeToMat, MarketVariables):
    #Transform the simulations of gaussian random variables to paths of the underlying asset(s)
    timeStepsTotal = round(timeStepsPerYear*timeToMat)
    rNorm = rng.standard_normal(size=(timeStepsTotal, pathTotal)).T
    paths = np.empty((pathTotal,timeStepsTotal+1), order="F")
    paths[:,0] = MarketVariables.spot
    timeIncrement = timeToMat/timeStepsTotal
    for timeStep in range(1,timeStepsTotal+1):
        if(pathTotal>1):
            #Normalize random variables to mean 0 and standard diviation 1
            rNorm[:,timeStep-1] = (rNorm[:,timeStep-1] - np.mean(rNorm[:,timeStep-1])) / np.std(rNorm[:,timeStep])
        paths[:,timeStep] = generateTimeStepStock(timeIncrement, rNorm[:,timeStep-1], MarketVariables, paths[:,timeStep-1])
    return paths